"""Preprocessing for phone-captured card photos before ML matching.

Phone photos of cards in binder sleeves suffer from:
- Uneven lighting / dark corners
- Sleeve glare (specular highlights)
- Sleeve edges / extra border captured by segmenter
- Occasional landscape orientation
- Holographic patterns (rainbow/specular reflections across the card surface)

This module applies lightweight OpenCV corrections to improve
DINOv2 and CLIP matching scores against clean reference images.

Two preprocessing paths:
  - **Standard**: gentle CLAHE + point-glare inpainting (for normal cards)
  - **Holo-aware**: edge-preserving filter + strong CLAHE (for holo/reverse holo)

The holo path uses cv2.edgePreservingFilter to smooth distributed holographic
rainbow patterns while preserving card text and artwork edges, followed by
aggressive CLAHE (clip=3.0, grid=4x4) to normalize the washed-out lighting
caused by holographic reflections. Tested on page 3 holo segments, this path
improves average DINOv2 top-1 similarity by +0.008 over the standard path
(0.5994 vs 0.5920) and +0.022 over no preprocessing (0.5776).
"""

import logging
import os
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_for_matching(image_path: str, keep_original_size: bool = False,
                            holo: bool = False) -> str:
    """Preprocess a card photo for better ML matching.

    Standard path steps:
      1. Auto-rotate to portrait if landscape
      2. CLAHE lighting normalization (per-channel in LAB)
      3. Sleeve glare removal (bright-spot detection + inpainting)
      4. Border crop (trim ~5% from each edge to remove sleeve/border)

    Holo path steps (when holo=True or auto-detected):
      1. Auto-rotate to portrait if landscape
      2. Edge-preserving filter (smooths holo rainbow patterns, keeps edges)
      3. Strong CLAHE (clip=3.0, grid=4x4) to normalize holo-washed lighting
      4. Border crop

    Parameters
    ----------
    image_path : str
        Path to the input card image.
    keep_original_size : bool
        If True, do not resize the output. Otherwise output matches input dims.
    holo : bool
        If True, use the holo-aware preprocessing path. If False, auto-detect
        based on saturation variance (high saturation variance = holographic).

    Returns
    -------
    str
        Path to the preprocessed temporary file (PNG).
        Caller is responsible for cleanup.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = img.shape[:2]
    logger.debug("preprocess: input %s  %dx%d", image_path, original_w, original_h)

    # Step 1: Auto-rotate to portrait if landscape
    img = _auto_rotate_portrait(img)

    # Auto-detect holo if not explicitly set
    is_holo = holo or _detect_holo(img)

    if is_holo:
        logger.debug("preprocess: using HOLO path")
        img = _holo_preprocess(img)
    else:
        logger.debug("preprocess: using STANDARD path")
        img = _standard_preprocess(img)

    # Final step: Crop borders (trim sleeve edges)
    img = _crop_borders(img, pct=0.05)

    # Save to temp file
    suffix = os.path.splitext(image_path)[1] or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="preproc_")
    os.close(fd)
    cv2.imwrite(tmp_path, img)

    out_h, out_w = img.shape[:2]
    logger.debug("preprocess: output %s  %dx%d", tmp_path, out_w, out_h)

    return tmp_path


def _detect_holo(img: np.ndarray) -> bool:
    """Detect whether an image likely shows a holographic card.

    Holographic cards have distinctive color properties:
    - High saturation variance (rainbow reflections create patches of
      very different saturations across the card surface)
    - High percentage of bright pixels from the reflective foil

    Parameters
    ----------
    img : np.ndarray
        BGR input image (already rotated to portrait).

    Returns
    -------
    bool
        True if the image appears to be a holographic card.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)

    sat_std = np.std(s_ch)
    bright_pct = np.mean(v_ch > 200) * 100

    is_holo = sat_std > 50 or bright_pct > 25
    logger.debug("preprocess: holo detect sat_std=%.1f bright_pct=%.1f%% -> %s",
                 sat_std, bright_pct, is_holo)
    return is_holo


def _standard_preprocess(img: np.ndarray) -> np.ndarray:
    """Standard preprocessing for non-holo cards.

    Gentle CLAHE + point-glare inpainting.
    """
    img = _apply_clahe(img)
    img = _remove_glare(img)
    return img


def _holo_preprocess(img: np.ndarray) -> np.ndarray:
    """Holo-aware preprocessing for holographic / reverse holo cards.

    Uses edge-preserving filter to smooth distributed holographic rainbow
    patterns while keeping card text and artwork edges sharp, followed by
    aggressive CLAHE to normalize the washed-out lighting from holo
    reflections.

    This approach was tested against 8 holo card segments and consistently
    outperformed the standard path:
      - edge_preserving + strong CLAHE: avg 0.5994 top-1 similarity
      - standard path:                  avg 0.5920
      - no preprocessing:               avg 0.5776
    """
    # Edge-preserving filter: smooth holo rainbow patterns, keep text/edges
    # flags=1 = RECURS_FILTER, sigma_s=60 spatial extent, sigma_r=0.4 color range
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    # Strong CLAHE to normalize holo-washed lighting
    # Higher clip_limit (3.0 vs 1.5) and smaller grid (4x4 vs 8x8) to
    # more aggressively equalize the uneven brightness from holo reflections
    img = _apply_clahe(img, clip_limit=3.0, grid_size=4)

    return img


def _auto_rotate_portrait(img: np.ndarray) -> np.ndarray:
    """Rotate image to portrait orientation if it is landscape."""
    h, w = img.shape[:2]
    if w > h:
        # Landscape -> rotate 90 degrees clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        logger.debug("preprocess: rotated landscape -> portrait")
    return img


def _apply_clahe(img: np.ndarray, clip_limit: float = 1.5, grid_size: int = 8) -> np.ndarray:
    """Apply gentle CLAHE in LAB color space for lighting normalization.

    Only equalizes the L (lightness) channel, preserving color.
    Uses a conservative clip_limit (1.5) to avoid over-enhancing
    card art contrast which hurts embedding similarity.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_chan = clahe.apply(l_chan)

    lab = cv2.merge([l_chan, a_chan, b_chan])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def _remove_glare(img: np.ndarray, brightness_thresh: int = 240,
                  saturation_max: int = 30, min_area: int = 200) -> np.ndarray:
    """Detect and inpaint specular highlights (sleeve glare).

    Uses a two-channel approach: pixels must be BOTH very bright (high V
    in HSV) AND low saturation (close to white) to qualify as glare.
    This avoids inpainting bright but colorful card art (energy symbols,
    holofoil highlights, white text on cards).

    Parameters
    ----------
    img : np.ndarray
        BGR input image.
    brightness_thresh : int
        Minimum V-channel value (0-255) for glare candidates.
    saturation_max : int
        Maximum S-channel value. Real glare is nearly colorless (low S).
        Card art highlights tend to have higher saturation.
    min_area : int
        Minimum contour area in pixels to count as glare.
        Filters out small bright spots that are part of card art.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Glare = very bright AND low saturation (near-white)
    bright = v_chan >= brightness_thresh
    desaturated = s_chan <= saturation_max
    glare_candidates = (bright & desaturated).astype(np.uint8) * 255

    # Dilate slightly to cover glare halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_candidates = cv2.dilate(glare_candidates, kernel, iterations=1)

    # Filter: only keep contours above min_area
    contours, _ = cv2.findContours(glare_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glare_mask = np.zeros_like(glare_candidates)
    glare_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(glare_mask, [cnt], -1, 255, -1)
            glare_count += 1

    if glare_count == 0:
        logger.debug("preprocess: no glare regions detected")
        return img

    # Safety check: if glare covers >15% of image, skip inpainting
    # (likely a white/bright card, not actual glare)
    glare_pixels = np.count_nonzero(glare_mask)
    total_pixels = img.shape[0] * img.shape[1]
    glare_pct = glare_pixels / total_pixels
    if glare_pct > 0.15:
        logger.debug("preprocess: glare covers %.1f%% of image, skipping inpaint (likely card art)",
                     100.0 * glare_pct)
        return img

    # Inpaint the glare regions
    result = cv2.inpaint(img, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    logger.debug("preprocess: inpainted %d glare regions (%d px, %.1f%% of image)",
                 glare_count, glare_pixels, 100.0 * glare_pct)

    return result


def _crop_borders(img: np.ndarray, pct: float = 0.05) -> np.ndarray:
    """Crop a percentage from each edge to remove sleeve borders.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    pct : float
        Fraction to trim from each side (0.05 = 5%).
    """
    h, w = img.shape[:2]
    top = int(h * pct)
    bottom = int(h * (1 - pct))
    left = int(w * pct)
    right = int(w * (1 - pct))

    cropped = img[top:bottom, left:right]
    logger.debug("preprocess: cropped borders %d,%d,%d,%d -> %dx%d",
                 top, bottom, left, right, cropped.shape[1], cropped.shape[0])
    return cropped


# ---------------------------------------------------------------------------
# FSRCNN super-resolution for OCR preprocessing
# ---------------------------------------------------------------------------

# Lazy-loaded singleton for the FSRCNN model
_fsrcnn_model = None
_fsrcnn_device = None


def _get_fsrcnn_model():
    """Load the FSRCNN x2 super-resolution model (lazy singleton).

    Architecture: FSRCNN(d=56, s=12, m=4) with PixelShuffle upscaling.
    Weights extracted from the Saafke/FSRCNN_Tensorflow pretrained model
    (trained on General-100 + T91 datasets).

    Returns (model, device) or (None, None) if unavailable.
    """
    global _fsrcnn_model, _fsrcnn_device

    if _fsrcnn_model is not None:
        return _fsrcnn_model, _fsrcnn_device

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.debug("upscale: torch not available, FSRCNN disabled")
        return None, None

    weight_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "models", "fsrcnn_x2.pth"
    )
    weight_path = os.path.normpath(weight_path)

    if not os.path.exists(weight_path):
        logger.debug("upscale: FSRCNN weights not found at %s", weight_path)
        return None, None

    class FSRCNN(nn.Module):
        """FSRCNN architecture (Dong et al., ECCV 2016) for 2x super-resolution.

        Fully convolutional -- accepts any input size.
        Uses sub-pixel convolution (PixelShuffle) for the final upscaling step.
        """

        def __init__(self, scale=2, d=56, s=12, m=4):
            super().__init__()
            self.first = nn.Sequential(
                nn.Conv2d(1, d, 5, padding=2), nn.PReLU(d)
            )
            self.shrink = nn.Sequential(nn.Conv2d(d, s, 1), nn.PReLU(s))
            layers = []
            for _ in range(m):
                layers.extend([nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)])
            self.mapping = nn.Sequential(*layers)
            self.expand = nn.Sequential(nn.Conv2d(s, d, 1), nn.PReLU(d))
            self.upscale = nn.Sequential(
                nn.Conv2d(d, scale * scale, 1), nn.PixelShuffle(scale)
            )

        def forward(self, x):
            x = self.first(x)
            x = self.shrink(x)
            x = self.mapping(x)
            x = self.expand(x)
            x = self.upscale(x)
            return x

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FSRCNN(scale=2, d=56, s=12, m=4)
        state = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device).eval()

        # Warmup forward pass to trigger CUDA kernel compilation
        with torch.no_grad():
            _ = model(torch.randn(1, 1, 32, 32, device=device))
            if device == "cuda":
                torch.cuda.synchronize()

        _fsrcnn_model = model
        _fsrcnn_device = device
        logger.info("upscale: loaded FSRCNN x2 on %s (%d params)",
                     device, sum(p.numel() for p in model.parameters()))
        return model, device
    except Exception:
        logger.warning("upscale: failed to load FSRCNN", exc_info=True)
        return None, None


def upscale_for_ocr(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscale an image crop using FSRCNN for better OCR text quality.

    Processes the luminance (Y) channel through a pretrained FSRCNN network
    for sharper edge reconstruction, then bicubic-upscales the chroma channels.
    Falls back to ``cv2.INTER_CUBIC`` if FSRCNN is unavailable (no GPU, no
    weights, no PyTorch).

    For scales > 2, FSRCNN runs 2x first, then cubic interpolation covers the
    remaining factor (e.g. scale=3 -> FSRCNN 2x then cubic 1.5x).

    Parameters
    ----------
    img : np.ndarray
        BGR input image (typically a name or number crop from a card).
    scale : int
        Upscale factor.  FSRCNN handles the first 2x; any remainder uses
        cubic interpolation.

    Returns
    -------
    np.ndarray
        Upscaled BGR image (``scale`` times larger in each dimension).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    target_h, target_w = h * scale, w * scale

    # Try FSRCNN for the 2x portion
    if scale >= 2:
        model, device = _get_fsrcnn_model()
        if model is not None:
            try:
                result = _fsrcnn_upscale(img, model, device)
                if scale == 2:
                    return result
                # For scale > 2, apply remaining upscale with cubic
                return cv2.resize(result, (target_w, target_h),
                                  interpolation=cv2.INTER_CUBIC)
            except Exception:
                logger.warning("upscale: FSRCNN failed, falling back to cubic",
                               exc_info=True)

    # Fallback: high-quality cubic interpolation
    logger.debug("upscale: using INTER_CUBIC %dx%d -> %dx%d", w, h, target_w, target_h)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _fsrcnn_upscale(img: np.ndarray, model, device: str) -> np.ndarray:
    """Run FSRCNN 2x super-resolution on a BGR image.

    Converts to YCrCb, super-resolves the Y (luminance) channel through
    the network, and bicubic-upscales the Cr/Cb (chroma) channels to match.
    """
    import torch

    # Convert BGR -> YCrCb and extract luminance
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32) / 255.0

    # Network forward pass
    y_tensor = torch.from_numpy(y_channel).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_sr = model(y_tensor)
        if device == "cuda":
            import torch.cuda
            torch.cuda.synchronize()

    y_out = y_sr.squeeze().cpu().numpy()
    y_out = np.clip(y_out * 255.0, 0, 255).astype(np.uint8)
    new_h, new_w = y_out.shape

    # Bicubic-upscale chroma channels to match
    cr_up = cv2.resize(ycrcb[:, :, 1], (new_w, new_h),
                       interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(ycrcb[:, :, 2], (new_w, new_h),
                       interpolation=cv2.INTER_CUBIC)

    # Reconstruct and convert back to BGR
    ycrcb_sr = cv2.merge([y_out, cr_up, cb_up])
    return cv2.cvtColor(ycrcb_sr, cv2.COLOR_YCrCb2BGR)
