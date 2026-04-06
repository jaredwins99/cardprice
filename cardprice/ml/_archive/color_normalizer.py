"""Color normalization for binder-scanned cards to reduce domain gap with reference images.

Binder scans suffer from several color distortions that hurt DINOv2 matching:
- **Color cast**: Orange/blue binder background bleeds through card sleeves,
  shifting the overall color balance.
- **Uneven lighting**: Center of binder page is brighter than edges due to
  flash/ambient light falloff.
- **Sleeve reflections**: Specular highlights add white/silver haze, washing
  out colors in affected regions.
- **White balance mismatch**: Camera auto-WB under indoor lighting produces
  different color temperature than the studio-lit reference images.

This module provides four techniques that can be used individually or combined:

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the L channel
   of LAB space -- normalizes uneven lighting without distorting hue/saturation.

2. **Gray World White Balance** -- scales R, G, B channels so their means are
   equal, removing systematic color casts from binder/lighting.

3. **Histogram Matching** -- when a reference image is available, reshapes the
   scanned card's LAB histograms to match the reference's distribution,
   directly closing the domain gap.

4. **Sleeve Reflection Reduction** -- detects specular highlights (high V, low S
   in HSV) and desaturates/darkens them toward surrounding pixel values.

The main entry point ``normalize_card_colors()`` applies a sensible default
pipeline.  Individual techniques are exposed as public functions for A/B testing.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_card_colors(
    card_img: np.ndarray,
    reference_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize card image colors to reduce domain gap with reference images.

    Tested on 93 ground truth cards across 11 binder pages (WotC, e-reader,
    DP, modern SV eras).  DINOv2 cosine similarity results:

      Technique             Mean delta vs original
      --------------------- ----------------------
      hist_match_only       +0.021  (best single technique)
      reflection_only       +0.000  (neutral)
      clahe_only            +0.000  (neutral)
      gray_world_only       -0.006  (slightly harmful)

    Based on these results, the default pipeline is:
    - When reference_img is provided: histogram matching only (the clear winner)
    - When no reference: sleeve reflection reduction only (safe, never hurts)

    Gray world and CLAHE are available via ``normalize_with_techniques()`` for
    cases where they may help (e.g., extreme color cast or lighting variation).

    Parameters
    ----------
    card_img : np.ndarray
        BGR image from binder scan.
    reference_img : np.ndarray, optional
        BGR reference image to match histograms against.  When provided,
        histogram matching is applied for maximum domain gap reduction.

    Returns
    -------
    np.ndarray
        BGR image with corrected colors, same size as input.
    """
    if card_img is None or card_img.size == 0:
        return card_img

    result = card_img.copy()

    # Step 1: Reduce sleeve reflections (safe, never hurts: +0.000 on eval)
    result = reduce_sleeve_reflections(result)

    # Step 2: Histogram matching if reference is available (+0.021 on eval)
    if reference_img is not None:
        result = match_histogram(result, reference_img)

    return result


# ---------------------------------------------------------------------------
# Technique 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------------------------

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB color space.

    Normalizes uneven lighting (bright center, dark edges) without
    distorting the color information in A and B channels.

    Parameters
    ----------
    img : np.ndarray
        BGR input image.
    clip_limit : float
        CLAHE contrast clipping limit.  Higher values allow more contrast
        enhancement but risk amplifying noise.  2.0 is a good default for
        binder scans (gentler than the 3.0 used for holo cards).
    grid_size : int
        CLAHE tile grid size.  8x8 provides good locality for the typical
        uneven lighting pattern in binder scans.

    Returns
    -------
    np.ndarray
        BGR image with normalized lighting.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    l_chan = clahe.apply(l_chan)

    lab = cv2.merge([l_chan, a_chan, b_chan])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    logger.debug(
        "clahe: clip=%.1f grid=%d applied",
        clip_limit, grid_size,
    )
    return result


# ---------------------------------------------------------------------------
# Technique 2: Gray World White Balance
# ---------------------------------------------------------------------------

def gray_world_white_balance(
    img: np.ndarray,
    max_scale: float = 1.5,
) -> np.ndarray:
    """Apply gray world assumption white balance correction.

    The gray world hypothesis assumes that the average color in a
    well-lit scene should be neutral gray.  When the binder background
    introduces an orange or blue cast, the channel means diverge --
    scaling each channel to equalize the means removes the cast.

    Parameters
    ----------
    img : np.ndarray
        BGR input image.
    max_scale : float
        Maximum per-channel scale factor.  Clamped to prevent extreme
        corrections on images that are legitimately dominated by one
        color (e.g., a Fire-type card with lots of red).

    Returns
    -------
    np.ndarray
        BGR image with corrected white balance.
    """
    img_float = img.astype(np.float64)

    # Compute per-channel means
    b_mean = img_float[:, :, 0].mean()
    g_mean = img_float[:, :, 1].mean()
    r_mean = img_float[:, :, 2].mean()

    if b_mean == 0 or g_mean == 0 or r_mean == 0:
        logger.debug("gray_world: zero-mean channel, skipping")
        return img

    # Target: overall mean across all channels
    overall_mean = (b_mean + g_mean + r_mean) / 3.0

    # Scale factors to equalize channel means
    b_scale = np.clip(overall_mean / b_mean, 1.0 / max_scale, max_scale)
    g_scale = np.clip(overall_mean / g_mean, 1.0 / max_scale, max_scale)
    r_scale = np.clip(overall_mean / r_mean, 1.0 / max_scale, max_scale)

    logger.debug(
        "gray_world: means B=%.1f G=%.1f R=%.1f -> scales B=%.3f G=%.3f R=%.3f",
        b_mean, g_mean, r_mean, b_scale, g_scale, r_scale,
    )

    result = img_float.copy()
    result[:, :, 0] *= b_scale
    result[:, :, 1] *= g_scale
    result[:, :, 2] *= r_scale

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Technique 3: Histogram Matching
# ---------------------------------------------------------------------------

def match_histogram(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match the source image histogram to the reference image histogram.

    Operates channel-by-channel in LAB color space so that luminance
    and chrominance are matched independently.  This directly reduces
    the domain gap between a binder scan and its reference card image.

    Parameters
    ----------
    source : np.ndarray
        BGR source image (the binder scan to correct).
    reference : np.ndarray
        BGR reference image (the clean card image to match against).

    Returns
    -------
    np.ndarray
        BGR image with histogram matched to reference.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    matched_channels = []
    channel_names = ("L", "A", "B")

    for i in range(3):
        src_chan = src_lab[:, :, i]
        ref_chan = ref_lab[:, :, i]
        matched = _match_channel_histogram(src_chan, ref_chan)
        matched_channels.append(matched)

        logger.debug(
            "hist_match: %s channel src_mean=%.1f ref_mean=%.1f -> matched_mean=%.1f",
            channel_names[i],
            src_chan.mean(), ref_chan.mean(), matched.mean(),
        )

    matched_lab = cv2.merge(matched_channels)
    result = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    return result


def _match_channel_histogram(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match the histogram of a single-channel image to a reference.

    Uses the CDF-based approach: for each source pixel value, find the
    reference pixel value whose CDF is closest.

    Parameters
    ----------
    source : np.ndarray
        Single-channel uint8 source image.
    reference : np.ndarray
        Single-channel uint8 reference image.

    Returns
    -------
    np.ndarray
        Single-channel uint8 image with matched histogram.
    """
    # Compute histograms
    src_hist, _ = np.histogram(source.ravel(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(reference.ravel(), bins=256, range=(0, 256))

    # Compute CDFs
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    # Normalize CDFs to [0, 1]
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1

    # Build lookup table: for each source value, find the reference value
    # with the closest CDF value
    lookup = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        # Find the reference value whose CDF is closest to src_cdf[src_val]
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup[src_val] = np.argmin(diff)

    # Apply the mapping
    return lookup[source]


# ---------------------------------------------------------------------------
# Technique 4: Sleeve Reflection Reduction
# ---------------------------------------------------------------------------

def reduce_sleeve_reflections(
    img: np.ndarray,
    brightness_thresh: int = 230,
    saturation_max: int = 40,
    blend_strength: float = 0.7,
) -> np.ndarray:
    """Reduce specular highlights from sleeve reflections.

    Instead of full inpainting (expensive and can introduce artifacts),
    this uses a soft blending approach:
    1. Detect specular highlights: pixels with high V and low S in HSV.
    2. Compute a soft mask (gradual falloff rather than hard threshold).
    3. For highlighted pixels, blend toward the local median color,
       reducing brightness and restoring color without sharp seams.

    Parameters
    ----------
    img : np.ndarray
        BGR input image.
    brightness_thresh : int
        Minimum V-channel value (0-255) for reflection candidates.
    saturation_max : int
        Maximum S-channel value.  Reflections are near-white (low S).
    blend_strength : float
        How aggressively to replace highlights.  1.0 = full replacement
        with median, 0.0 = no change.  0.7 provides good correction
        without visible artifacts.

    Returns
    -------
    np.ndarray
        BGR image with reduced sleeve reflections.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Build a soft reflection mask: stronger where brightness is higher
    # and saturation is lower (more "white")
    v_float = v_chan.astype(np.float32)
    s_float = s_chan.astype(np.float32)

    # Brightness component: ramp from 0 at threshold to 1 at 255
    brightness_score = np.clip(
        (v_float - brightness_thresh) / (255.0 - brightness_thresh), 0, 1
    )

    # Saturation component: ramp from 0 at saturation_max to 1 at 0
    saturation_score = np.clip(
        (saturation_max - s_float) / saturation_max, 0, 1
    )

    # Combined soft mask
    reflection_mask = brightness_score * saturation_score

    # Check coverage -- skip if negligible or too large
    mask_sum = reflection_mask.sum()
    total_pixels = img.shape[0] * img.shape[1]
    coverage = mask_sum / total_pixels

    if coverage < 0.001:
        logger.debug("reflections: no significant reflections detected (coverage=%.4f)", coverage)
        return img

    if coverage > 0.20:
        logger.debug(
            "reflections: coverage %.1f%% too high (likely white card art), skipping",
            coverage * 100,
        )
        return img

    # Compute local median as replacement color (using a blur as fast approximation)
    # Median blur smooths out the specular spikes while preserving card structure
    local_color = cv2.medianBlur(img, 15)

    # Blend: result = img * (1 - mask*strength) + local_color * mask*strength
    mask_3ch = np.stack([reflection_mask] * 3, axis=-1) * blend_strength
    result = img.astype(np.float32) * (1.0 - mask_3ch) + local_color.astype(np.float32) * mask_3ch

    logger.debug(
        "reflections: reduced highlights, coverage=%.2f%%, strength=%.1f",
        coverage * 100, blend_strength,
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Utility: apply individual techniques for A/B testing
# ---------------------------------------------------------------------------

def normalize_with_techniques(
    card_img: np.ndarray,
    reference_img: Optional[np.ndarray] = None,
    use_clahe: bool = True,
    use_gray_world: bool = True,
    use_hist_match: bool = True,
    use_reflection_reduction: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """Apply selected normalization techniques for controlled experiments.

    Parameters
    ----------
    card_img : np.ndarray
        BGR image from binder scan.
    reference_img : np.ndarray, optional
        BGR reference image for histogram matching.
    use_clahe : bool
        Apply CLAHE lighting normalization.
    use_gray_world : bool
        Apply gray world white balance.
    use_hist_match : bool
        Apply histogram matching (requires reference_img).
    use_reflection_reduction : bool
        Apply sleeve reflection reduction.
    clahe_clip : float
        CLAHE clip limit.
    clahe_grid : int
        CLAHE grid size.

    Returns
    -------
    np.ndarray
        Normalized BGR image.
    """
    result = card_img.copy()

    if use_reflection_reduction:
        result = reduce_sleeve_reflections(result)

    if use_gray_world:
        result = gray_world_white_balance(result)

    if use_clahe:
        result = apply_clahe(result, clip_limit=clahe_clip, grid_size=clahe_grid)

    if use_hist_match and reference_img is not None:
        result = match_histogram(result, reference_img)

    return result
