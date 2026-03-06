"""Page-level color cast measurement and correction for binder scans.

Binder page photos suffer from a consistent warm color cast (R-B shift of
38-51 points) caused by ambient lighting reflected off the orange binder
material. The cast varies per page but is uniform within a page.

This module implements page-level correction BEFORE card-level processing:
1. Sample the binder background visible between/around card slots
2. The binder material has a known color profile under neutral lighting
3. Derive per-channel correction gains from observed vs expected binder color
4. Apply correction to all card segments extracted from that page

Two independent estimation strategies are used and averaged:
  - **Binder background**: Sample orange binder pixels between cards, compare
    to a reference binder BGR calibrated from neutral-light photos
  - **Card border**: Sample the yellow card borders visible on each card,
    compare to the known reference yellow border channel ratios

The dual-estimator approach is more robust than either alone: the binder
background estimator works even when cards are non-standard (full-art, no
yellow border), while the card border estimator works even when binder
background is occluded or cropped.

Calibration data (from 5 test binder pages + 20k reference images):
  - Reference binder BGR under neutral light: (52, 135, 210)
    Measured by averaging observed binder across pages and adjusting for
    the known card-border-derived cast.
  - Reference card yellow border: B/R=0.341, G/R=0.886
    From median of 20k reference card images.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Reference binder material BGR under neutral lighting.
# Calibrated from test pages: observed ~(33, 124, 220) with a warm cast
# that shifts B down by ~15-20 and G down by ~10-15 relative to R.
# After applying the card-border-derived correction (B*1.20, G*1.10),
# the neutral binder color is approximately:
_REF_BINDER_BGR = np.array([52.0, 135.0, 210.0], dtype=np.float64)

# Reference card yellow border channel ratios (relative to R channel).
# From median across 20k clean reference card images.
_REF_BORDER_B_RATIO = 0.341  # B/R
_REF_BORDER_G_RATIO = 0.886  # G/R

# Gain clamps to prevent extreme corrections
_MIN_GAIN = 0.70
_MAX_GAIN = 1.60


class CorrectionMatrix:
    """Per-channel color correction derived from page-level measurements.

    Stores BGR gain factors and the diagnostic measurements used to
    derive them, so callers can log/inspect the correction quality.
    """

    def __init__(
        self,
        gains_bgr: np.ndarray,
        *,
        binder_observed_bgr: Optional[np.ndarray] = None,
        border_observed_bgr: Optional[np.ndarray] = None,
        binder_gain: Optional[np.ndarray] = None,
        border_gain: Optional[np.ndarray] = None,
        saturation_boost: float = 1.15,
        n_binder_pixels: int = 0,
        n_border_pixels: int = 0,
    ):
        self.gains_bgr = gains_bgr.astype(np.float32)
        self.binder_observed_bgr = binder_observed_bgr
        self.border_observed_bgr = border_observed_bgr
        self.binder_gain = binder_gain
        self.border_gain = border_gain
        self.saturation_boost = saturation_boost
        self.n_binder_pixels = n_binder_pixels
        self.n_border_pixels = n_border_pixels

    def __repr__(self) -> str:
        g = self.gains_bgr
        return (
            f"CorrectionMatrix(B={g[0]:.3f}, G={g[1]:.3f}, R={g[2]:.3f}, "
            f"sat_boost={self.saturation_boost:.2f}, "
            f"binder_px={self.n_binder_pixels}, border_px={self.n_border_pixels})"
        )

    @property
    def is_identity(self) -> bool:
        """True if the correction is effectively a no-op."""
        return bool(np.allclose(self.gains_bgr, 1.0, atol=0.02)
                     and abs(self.saturation_boost - 1.0) < 0.02)

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "gains_bgr": [round(float(g), 4) for g in self.gains_bgr],
            "saturation_boost": round(self.saturation_boost, 3),
            "binder_observed_bgr": (
                [round(float(v), 1) for v in self.binder_observed_bgr]
                if self.binder_observed_bgr is not None else None
            ),
            "border_observed_bgr": (
                [round(float(v), 1) for v in self.border_observed_bgr]
                if self.border_observed_bgr is not None else None
            ),
            "n_binder_pixels": self.n_binder_pixels,
            "n_border_pixels": self.n_border_pixels,
        }


# ---------------------------------------------------------------------------
# Binder background sampling
# ---------------------------------------------------------------------------

def _sample_binder_background(page_image: np.ndarray) -> Optional[np.ndarray]:
    """Sample orange binder material pixels from between/around card slots.

    Samples from:
      - Page edges (top/bottom/left/right 3% strips)
      - Inter-card gaps (horizontal strips at 33%/66% height,
        vertical strips at 33%/66% width)

    Filters for orange-hued pixels (H 5-25, S > 80, V > 80 in OpenCV HSV)
    which correspond to the binder page material.

    Returns
    -------
    numpy.ndarray or None
        Mean BGR of binder pixels as float64 array, or None if too few
        binder pixels found (< 500).
    """
    h, w = page_image.shape[:2]

    strips = []

    # Edge strips (3% of each side)
    edge_h = max(int(h * 0.03), 4)
    edge_w = max(int(w * 0.03), 4)
    strips.append(page_image[:edge_h, :])          # top
    strips.append(page_image[h - edge_h:, :])      # bottom
    strips.append(page_image[:, :edge_w])           # left
    strips.append(page_image[:, w - edge_w:])       # right

    # Horizontal inter-card gaps (~33% and ~66% height)
    gap_half = max(int(h * 0.01), 4)
    for row_frac in [0.33, 0.66]:
        cy = int(h * row_frac)
        y1 = max(0, cy - gap_half)
        y2 = min(h, cy + gap_half)
        strips.append(page_image[y1:y2, int(w * 0.05):int(w * 0.95)])

    # Vertical inter-card gaps (~33% and ~66% width)
    gap_half_w = max(int(w * 0.01), 4)
    for col_frac in [0.33, 0.66]:
        cx = int(w * col_frac)
        x1 = max(0, cx - gap_half_w)
        x2 = min(w, cx + gap_half_w)
        strips.append(page_image[int(h * 0.05):int(h * 0.95), x1:x2])

    # Combine and subsample for speed
    all_pixels = np.vstack([s.reshape(-1, 3) for s in strips if s.size > 0])

    if len(all_pixels) > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_pixels), 200_000, replace=False)
        all_pixels = all_pixels[idx]

    # Convert to HSV and filter for orange binder material
    hsv = cv2.cvtColor(
        all_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    orange_mask = (
        (hsv[:, 0] >= 5) & (hsv[:, 0] <= 25) &
        (hsv[:, 1] > 80) & (hsv[:, 2] > 80)
    )

    n_orange = int(orange_mask.sum())
    if n_orange < 500:
        logger.debug("Binder background: only %d orange pixels (need 500+)", n_orange)
        return None

    binder_mean = all_pixels[orange_mask].astype(np.float64).mean(axis=0)
    logger.debug(
        "Binder background: BGR=(%.0f, %.0f, %.0f) from %d pixels",
        binder_mean[0], binder_mean[1], binder_mean[2], n_orange,
    )
    return binder_mean


def _gains_from_binder(observed_bgr: np.ndarray) -> np.ndarray:
    """Compute per-channel gains to correct binder color to reference.

    The gain for each channel = reference / observed.
    Gains are clamped to [_MIN_GAIN, _MAX_GAIN].
    """
    gains = _REF_BINDER_BGR / np.maximum(observed_bgr, 1.0)
    gains = np.clip(gains, _MIN_GAIN, _MAX_GAIN)
    return gains


# ---------------------------------------------------------------------------
# Card border sampling (from full page, not individual segments)
# ---------------------------------------------------------------------------

def _sample_card_borders_from_page(page_image: np.ndarray) -> Optional[np.ndarray]:
    """Sample card yellow border pixels from the full binder page.

    Estimates approximate card positions from a 3x3 grid layout,
    samples the outer border strip of each card, and filters for
    bright warm pixels likely to be the yellow Pokemon card border.

    Returns
    -------
    numpy.ndarray or None
        Mean BGR of card border pixels, or None if insufficient data.
    """
    h, w = page_image.shape[:2]

    # Estimate card grid positions
    # Cards typically occupy ~92-96% of the page area with small margins
    margin_x = int(w * 0.03)
    margin_y = int(h * 0.02)
    card_w = (w - 2 * margin_x) / 3
    card_h = (h - 2 * margin_y) / 3

    border_pixels = []

    for row in range(3):
        for col in range(3):
            # Card center
            cx = int(margin_x + col * card_w + card_w / 2)
            cy = int(margin_y + row * card_h + card_h / 2)
            hw = int(card_w * 0.42)
            hh = int(card_h * 0.42)

            x1 = max(0, cx - hw)
            x2 = min(w, cx + hw)
            y1 = max(0, cy - hh)
            y2 = min(h, cy + hh)

            card = page_image[y1:y2, x1:x2]
            if card.size == 0:
                continue

            ch, cw_px = card.shape[:2]
            bw_h = max(int(ch * 0.04), 3)
            bw_w = max(int(cw_px * 0.04), 3)

            # Sample border strips (top, bottom, left, right of card)
            strips = []
            strips.append(card[:bw_h, :])
            strips.append(card[ch - bw_h:, :])
            strips.append(card[:, :bw_w])
            strips.append(card[:, cw_px - bw_w:])

            px = np.vstack([s.reshape(-1, 3) for s in strips if s.size > 0])

            # Filter: bright enough and warm-ish (yellow border)
            # Yellow border pixels should be bright (mean > 100) and have R > B
            brightness = px.astype(np.float32).mean(axis=1)
            r_channel = px[:, 2].astype(np.float32)
            b_channel = px[:, 0].astype(np.float32)
            warm_bright = (brightness > 100) & (r_channel > b_channel + 10)

            if warm_bright.sum() > 50:
                border_pixels.append(px[warm_bright])

    if not border_pixels:
        logger.debug("Card borders: no bright warm border pixels found")
        return None

    all_borders = np.vstack(border_pixels).astype(np.float64)

    if len(all_borders) < 200:
        logger.debug("Card borders: only %d pixels (need 200+)", len(all_borders))
        return None

    border_mean = all_borders.mean(axis=0)
    logger.debug(
        "Card borders: BGR=(%.0f, %.0f, %.0f) from %d pixels  "
        "B/R=%.3f G/R=%.3f",
        border_mean[0], border_mean[1], border_mean[2], len(all_borders),
        border_mean[0] / max(border_mean[2], 1),
        border_mean[1] / max(border_mean[2], 1),
    )
    return border_mean


def _gains_from_border(observed_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Compute per-channel gains from card border color vs reference.

    Uses the known yellow border B/R and G/R ratios under neutral lighting.
    R channel is kept as anchor (gain=1.0), B and G are adjusted.

    Returns None if the border doesn't look like a warm yellow (which would
    mean we're sampling wrong pixels).
    """
    obs_b, obs_g, obs_r = float(observed_bgr[0]), float(observed_bgr[1]), float(observed_bgr[2])

    if obs_r < 80:
        return None  # too dark to be a card border

    obs_b_ratio = obs_b / obs_r
    obs_g_ratio = obs_g / obs_r

    # Only correct if there's a meaningful deviation
    # (if ratios are already close to reference, no correction needed)
    b_gain = _REF_BORDER_B_RATIO / max(obs_b_ratio, 0.15)
    g_gain = _REF_BORDER_G_RATIO / max(obs_g_ratio, 0.30)

    # Clamp
    b_gain = float(np.clip(b_gain, _MIN_GAIN, _MAX_GAIN))
    g_gain = float(np.clip(g_gain, _MIN_GAIN, _MAX_GAIN))

    gains = np.array([b_gain, g_gain, 1.0], dtype=np.float64)
    return gains


# ---------------------------------------------------------------------------
# Saturation estimation
# ---------------------------------------------------------------------------

def _estimate_saturation_boost(page_image: np.ndarray) -> float:
    """Estimate how much saturation boost is needed.

    Binder photos are typically desaturated compared to reference images.
    Measures mean saturation of card-area pixels and compares to the
    expected saturation of reference card images (~110-130 in OpenCV scale).

    Returns a boost factor in [1.0, 1.5].
    """
    h, w = page_image.shape[:2]

    # Sample interior of page (card area)
    interior = page_image[int(h * 0.05):int(h * 0.95),
                          int(w * 0.05):int(w * 0.95)]

    hsv = cv2.cvtColor(interior[::4, ::4], cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Exclude very dark and very bright pixels (shadows, glare)
    val = hsv[:, :, 2].astype(np.float32)
    mask = (val > 40) & (val < 240)

    if mask.sum() < 100:
        return 1.15  # default

    mean_sat = float(sat[mask].mean())

    # Reference card images typically have mean saturation ~115-130
    # for the card area. If page saturation is lower, boost it.
    target_sat = 120.0
    if mean_sat < 10:
        return 1.0  # nearly grayscale, don't try

    boost = target_sat / mean_sat
    boost = float(np.clip(boost, 1.0, 1.5))

    logger.debug("Saturation: page mean=%.0f, target=%.0f, boost=%.2f",
                 mean_sat, target_sat, boost)
    return boost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_page_cast(page_image: np.ndarray) -> CorrectionMatrix:
    """Measure the color cast of a binder page and return a correction matrix.

    Combines two independent estimators:
      1. Binder background color vs reference
      2. Card border yellow color vs reference

    When both estimators agree, the correction is highly reliable.
    When only one is available, it is used alone with slightly more
    conservative gains.

    Parameters
    ----------
    page_image : numpy.ndarray
        Full binder page image in BGR (as from cv2.imread).

    Returns
    -------
    CorrectionMatrix
        Per-channel correction gains and diagnostic data.
    """
    binder_bgr = _sample_binder_background(page_image)
    border_bgr = _sample_card_borders_from_page(page_image)

    binder_gains = None
    border_gains = None
    n_binder = 0
    n_border = 0

    if binder_bgr is not None:
        binder_gains = _gains_from_binder(binder_bgr)
        # Count is approximate from the sampling
        n_binder = 1  # flag that we have binder data

    if border_bgr is not None:
        border_gains = _gains_from_border(border_bgr)
        n_border = 1

    # Combine estimators
    if binder_gains is not None and border_gains is not None:
        # Average the two estimates, weighted toward border (more reliable)
        final_gains = 0.4 * binder_gains + 0.6 * border_gains
        logger.debug(
            "Color correction: binder gains=(%.3f, %.3f, %.3f), "
            "border gains=(%.3f, %.3f, %.3f), "
            "combined=(%.3f, %.3f, %.3f)",
            *binder_gains, *border_gains, *final_gains,
        )
    elif border_gains is not None:
        # Only border available -- use with slight damping
        final_gains = 1.0 + (border_gains - 1.0) * 0.85
        logger.debug("Color correction: border-only gains=(%.3f, %.3f, %.3f)",
                     *final_gains)
    elif binder_gains is not None:
        # Only binder available -- use with slight damping
        final_gains = 1.0 + (binder_gains - 1.0) * 0.85
        logger.debug("Color correction: binder-only gains=(%.3f, %.3f, %.3f)",
                     *final_gains)
    else:
        # No data at all -- identity correction
        final_gains = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        logger.warning("Color correction: no binder or border data, using identity")

    # Clamp final gains
    final_gains = np.clip(final_gains, _MIN_GAIN, _MAX_GAIN)

    # Estimate saturation boost
    sat_boost = _estimate_saturation_boost(page_image)

    matrix = CorrectionMatrix(
        gains_bgr=final_gains,
        binder_observed_bgr=binder_bgr,
        border_observed_bgr=border_bgr,
        binder_gain=binder_gains,
        border_gain=border_gains,
        saturation_boost=sat_boost,
        n_binder_pixels=n_binder,
        n_border_pixels=n_border,
    )

    logger.info("Page color correction: %s", matrix)
    return matrix


def apply_correction(
    card_segment: np.ndarray,
    correction: CorrectionMatrix,
) -> np.ndarray:
    """Apply page-level color correction to a single card segment.

    Parameters
    ----------
    card_segment : numpy.ndarray
        BGR card image (from cv2.imread or segmenter output).
    correction : CorrectionMatrix
        Correction derived from measure_page_cast().

    Returns
    -------
    numpy.ndarray
        Corrected BGR image (same shape/dtype as input).
    """
    if correction.is_identity:
        return card_segment

    # Apply per-channel gains
    img_f = card_segment.astype(np.float32)
    gains = correction.gains_bgr[np.newaxis, np.newaxis, :]
    corrected = img_f * gains
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Apply saturation boost
    if correction.saturation_boost > 1.01:
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * correction.saturation_boost, 0, 255
        )
        corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return corrected


def auto_correct_page_segments(
    page_image: np.ndarray,
    segments: list[np.ndarray],
) -> list[np.ndarray]:
    """Measure page cast and correct all card segments in one call.

    Convenience function that combines measure_page_cast() and
    apply_correction() for the common workflow.

    Parameters
    ----------
    page_image : numpy.ndarray
        Full binder page image in BGR.
    segments : list of numpy.ndarray
        List of BGR card segment images extracted from this page.

    Returns
    -------
    list of numpy.ndarray
        Corrected card segment images (same order as input).
    """
    correction = measure_page_cast(page_image)

    if correction.is_identity:
        logger.info("No color correction needed for this page")
        return segments

    corrected = [apply_correction(seg, correction) for seg in segments]

    logger.info(
        "Corrected %d segments: gains=(%.3f, %.3f, %.3f) sat_boost=%.2f",
        len(corrected),
        *correction.gains_bgr,
        correction.saturation_boost,
    )
    return corrected


def measure_and_save_debug(
    page_image_path: str,
    output_dir: Optional[str] = None,
) -> dict:
    """Measure color cast and save debug visualizations.

    Saves:
      - Side-by-side comparison of original vs corrected page
      - Binder background mask overlay
      - Correction diagnostics as text overlay

    Parameters
    ----------
    page_image_path : str
        Path to the binder page image.
    output_dir : str, optional
        Directory for debug output. Defaults to same directory as input.

    Returns
    -------
    dict
        Diagnostic information including correction matrix details.
    """
    from pathlib import Path

    page_path = Path(page_image_path)
    page_image = cv2.imread(str(page_path))
    if page_image is None:
        raise ValueError(f"Could not read image: {page_path}")

    if output_dir is None:
        out_dir = page_path.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    correction = measure_page_cast(page_image)

    # Apply correction to full page for visualization
    corrected_page = apply_correction(page_image, correction)

    # Create side-by-side comparison
    h, w = page_image.shape[:2]
    # Scale down for reasonable debug image size
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        orig_small = cv2.resize(page_image, None, fx=scale, fy=scale)
        corr_small = cv2.resize(corrected_page, None, fx=scale, fy=scale)
    else:
        orig_small = page_image.copy()
        corr_small = corrected_page.copy()

    sh, sw = orig_small.shape[:2]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_small, "Original", (10, 30), font, 0.8, (0, 0, 255), 2)
    cv2.putText(corr_small, "Corrected", (10, 30), font, 0.8, (0, 255, 0), 2)

    # Add correction info to corrected image
    info_lines = [
        f"B={correction.gains_bgr[0]:.3f} G={correction.gains_bgr[1]:.3f} R={correction.gains_bgr[2]:.3f}",
        f"Sat={correction.saturation_boost:.2f}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(corr_small, line, (10, 60 + i * 25), font, 0.6,
                     (0, 255, 0), 1)

    comparison = np.hstack([orig_small, corr_small])

    stem = page_path.stem
    comp_path = out_dir / f"{stem}_color_correction.jpg"
    cv2.imwrite(str(comp_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])

    logger.info("Saved debug comparison to %s", comp_path)

    return {
        "correction": correction.to_dict(),
        "comparison_path": str(comp_path),
        "page_path": str(page_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    args = sys.argv[1:]

    if not args:
        # Default: test on all binder pages
        from pathlib import Path
        test_dir = Path("data/test_binder_pages")
        pages = sorted(test_dir.glob("binder_page_*.jpg"))
        if not pages:
            print("No test binder pages found in data/test_binder_pages/")
            sys.exit(1)

        print(f"Testing color correction on {len(pages)} binder pages")
        print("=" * 80)

        for page_path in pages:
            print(f"\n--- {page_path.name} ---")
            result = measure_and_save_debug(str(page_path))
            corr = result["correction"]
            print(f"  Gains: B={corr['gains_bgr'][0]:.3f}  "
                  f"G={corr['gains_bgr'][1]:.3f}  "
                  f"R={corr['gains_bgr'][2]:.3f}")
            print(f"  Saturation boost: {corr['saturation_boost']:.2f}")
            if corr["binder_observed_bgr"]:
                b = corr["binder_observed_bgr"]
                print(f"  Binder observed BGR: ({b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f})")
            if corr["border_observed_bgr"]:
                b = corr["border_observed_bgr"]
                print(f"  Border observed BGR: ({b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f})")
            print(f"  Saved: {result['comparison_path']}")

        print("\n" + "=" * 80)
        print("Done.")
    else:
        for path in args:
            result = measure_and_save_debug(path)
            corr = result["correction"]
            print(f"{path}:")
            print(f"  Gains: B={corr['gains_bgr'][0]:.3f}  "
                  f"G={corr['gains_bgr'][1]:.3f}  "
                  f"R={corr['gains_bgr'][2]:.3f}")
            print(f"  Saturation boost: {corr['saturation_boost']:.2f}")
            print(f"  Saved: {result['comparison_path']}")
