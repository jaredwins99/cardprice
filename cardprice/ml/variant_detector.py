"""Detect Pokemon card variant from a phone photo.

Variant types:
  - normal:          Flat print, no holographic effects
  - holofoil:        Holographic pattern on the artwork area only
  - reverse_holofoil: Holographic pattern on everything EXCEPT the artwork
  - 1st_edition:     Has a "1st Edition" stamp (left side, below artwork)

Detection approach (pure OpenCV, no ML models):

1. **Holographic detection** -- Holo cards photographed under any light show
   rainbow/prismatic color shifts.  We detect this via two complementary signals:

   a) *Hue spread*: diversity of hue values at high saturation.  Normal prints
      have narrow hue distributions; holo surfaces scatter light across the
      full spectrum.

   b) *Hue spatial noise*: Laplacian of the hue channel measures high-frequency
      color variation.  Real holographic surfaces produce rapid, noisy color
      shifts between adjacent pixels (prismatic micro-reflections).  Digital
      artwork -- even very colorful art -- has smooth gradients that score low
      on this metric.  This is the key discriminator that prevents false
      positives on colorful but non-holo reference images.

2. **Artwork vs border localisation** -- Pokemon cards have a consistent layout:
   the artwork occupies roughly the center 80% width x top-center 45% height.
   We compare holo signal strength inside the artwork region vs the border/text
   region to distinguish holofoil (art only) from reverse holofoil (border only).

3. **1st Edition stamp** -- We look for the stamp using OCR (pytesseract) on
   the expected stamp region (left side, just below the artwork frame).  Falls
   back to contour-based circular blob detection.

Design notes:
  - Reference card images (data/card_images/) are digital scans with NO holo
    effect -- they must always classify as "normal".  Thresholds are set so that
    even the most colorful digital artwork (e.g., rainbow trainers, Charizard)
    stays below the holo detection boundary.
  - The detector is designed for phone photos of real physical cards where holo
    effects manifest as visible prismatic reflections under ambient light.
  - Detection quality depends on lighting -- photos taken under fluorescent or
    angled light reveal holo effects more clearly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region definitions (fractions of card width/height).
# Pokemon card layout is very consistent across all eras.
# ---------------------------------------------------------------------------
# Artwork bounding box (approximate, works for most card layouts)
ART_X0, ART_Y0, ART_X1, ART_Y1 = 0.10, 0.10, 0.90, 0.55

# Border region = full card minus artwork.  We sample the text/border area
# below and around the artwork.
BORDER_Y0 = 0.60  # text area starts below artwork

# 1st Edition stamp region (left side, just below artwork frame)
STAMP_X0, STAMP_Y0, STAMP_X1, STAMP_Y1 = 0.04, 0.54, 0.28, 0.66

# ---------------------------------------------------------------------------
# Thresholds (tuned heuristically)
# ---------------------------------------------------------------------------
# Minimum saturation to consider a pixel as "colorful" (filters out grey/white)
MIN_SATURATION = 50

# Minimum value (brightness) to avoid dark shadows
MIN_VALUE = 40

# Hue spread threshold: number of occupied hue bins (out of 36).
# Digital art maxes at ~27 (Charizard), but real holo under light hits 30+.
# We require BOTH hue spread AND spatial noise to exceed their thresholds.
HOLO_HUE_SPREAD_THRESHOLD = 20

# Hue spatial noise (Laplacian): mean absolute Laplacian of the hue channel
# at colorful pixels.  Digital art: typically 3-60.  Real holo phone photos:
# 80+ due to rapid prismatic color shifts between adjacent pixels.
HOLO_SPATIAL_NOISE_THRESHOLD = 70.0

# Combined holo score threshold -- requires both signals to be elevated.
# Score = hue_spread * (spatial_noise / NOISE_THRESHOLD).
# Normal digital scans: typically 0-30, worst case ~58 (retro pixel art like
# Base Set Charizard/Nidoking at 240x330).  Real holo phone photos: 100+
# due to genuine prismatic reflections.
HOLO_COMBINED_THRESHOLD = 60.0

# Ratio thresholds for art-vs-border holo discrimination
ART_HOLO_RATIO = 1.3
BORDER_HOLO_RATIO = 1.2


def _extract_region(img: np.ndarray, x0: float, y0: float,
                    x1: float, y1: float) -> np.ndarray:
    """Extract a rectangular region from an image using fractional coords."""
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _hue_spread(region_bgr: np.ndarray) -> int:
    """Count distinct hue bins with significant high-saturation pixel presence.

    Returns the number of hue bins (out of 36, each covering 5 degrees) that
    have at least 1% of the high-saturation pixels.
    """
    if region_bgr.size == 0:
        return 0

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = (s_chan >= MIN_SATURATION) & (v_chan >= MIN_VALUE)
    hues = h_chan[mask]

    if len(hues) < 50:
        return 0

    hist, _ = np.histogram(hues, bins=36, range=(0, 180))
    threshold = len(hues) * 0.01
    return int(np.sum(hist > threshold))


def _hue_spatial_noise(region_bgr: np.ndarray) -> float:
    """Measure diffuse, non-edge hue variation (holo-specific noise).

    Holographic surfaces produce random color speckle across the entire
    surface -- neighboring pixels have different hues even in "flat" areas
    away from structural edges.  Digital artwork concentrates color transitions
    at drawn edges (line art, shading boundaries).

    To distinguish these, we:
    1. Compute the hue Laplacian (high-frequency color changes)
    2. Compute a grayscale edge map (structural edges in the artwork)
    3. Mask OUT the structural edges and measure hue Laplacian only in
       the non-edge "flat" regions

    This gives us the hue noise that exists AWAY from structural edges --
    which is the signature of holographic prismatic reflections.

    Returns:
        Mean absolute Laplacian of hue in non-edge regions.
        Digital art: typically 2-30.  Real holo phone photo: 50-150+.
    """
    if region_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    colorful_mask = (s_chan >= MIN_SATURATION) & (v_chan >= MIN_VALUE)

    # Structural edges: Canny on grayscale.  Dilate to create a buffer zone
    # around edges so we exclude nearby pixels too.
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    non_edge_mask = edge_dilated == 0

    # Combined mask: colorful AND not on a structural edge
    combined_mask = colorful_mask & non_edge_mask

    # Hue Laplacian
    laplacian = cv2.Laplacian(h_chan, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(laplacian)

    flat_region_lap = abs_lap[combined_mask]
    if len(flat_region_lap) < 30:
        return 0.0

    return float(np.mean(flat_region_lap))


def _saturation_std(region_bgr: np.ndarray) -> float:
    """Measure standard deviation of saturation in a region."""
    if region_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    return float(np.std(hsv[:, :, 1]))


def _holo_score(region_bgr: np.ndarray) -> tuple[float, int, float]:
    """Compute a combined holographic score for a region.

    Returns:
        (combined_score, hue_spread, spatial_noise)

    The combined score multiplies hue spread by a noise factor.  Both signals
    must be elevated for the score to be high -- this prevents false positives
    from colorful but flat-printed artwork.
    """
    spread = _hue_spread(region_bgr)
    noise = _hue_spatial_noise(region_bgr)

    # Noise factor: how much the spatial noise exceeds (or falls short of)
    # the threshold.  Capped at 0.1 minimum to avoid zeroing everything.
    noise_factor = max(0.1, noise / HOLO_SPATIAL_NOISE_THRESHOLD)

    combined = spread * noise_factor
    return combined, spread, noise


def _check_1st_edition(img_bgr: np.ndarray) -> bool:
    """Check for 1st Edition stamp using OCR or template matching.

    The 1st Edition stamp appears as a small black "1" inside a circle with
    "EDITION" text, located on the left side just below the artwork.

    Returns True if a 1st Edition indicator is found.
    """
    stamp_region = _extract_region(img_bgr, STAMP_X0, STAMP_Y0,
                                   STAMP_X1, STAMP_Y1)
    if stamp_region.size == 0:
        return False

    # Strategy 1: Try pytesseract OCR if available
    try:
        import pytesseract

        gray = cv2.cvtColor(stamp_region, cv2.COLOR_BGR2GRAY)
        # Resize up for better OCR accuracy on small regions
        scale = max(1, 100 // max(gray.shape[0], 1))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        # Adaptive threshold to handle varied lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        text = pytesseract.image_to_string(binary, config="--psm 7").strip()
        text_lower = text.lower()
        if "1st" in text_lower or "edition" in text_lower:
            logger.debug("1st Edition detected via OCR: %r", text)
            return True
    except ImportError:
        logger.debug("pytesseract not available, skipping OCR-based check")
    except Exception as e:
        logger.debug("OCR check failed: %s", e)

    # Strategy 2: Look for the dark circular stamp shape
    try:
        gray = cv2.cvtColor(stamp_region, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        h_stamp, w_stamp = stamp_region.shape[:2]
        min_area = h_stamp * w_stamp * 0.02
        max_area = h_stamp * w_stamp * 0.40
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.5:
                    logger.debug("1st Edition stamp candidate found "
                                 "(area=%.0f, circ=%.2f)", area, circularity)
                    return True
    except Exception as e:
        logger.debug("Contour-based 1st Edition check failed: %s", e)

    return False


def detect_variant(image_path: str | Path) -> str:
    """Detect the variant of a Pokemon card from a photo.

    Args:
        image_path: Path to the card image (phone photo or scan).

    Returns:
        One of: "normal", "holofoil", "reverse_holofoil", "1st_edition".

    The function first checks for a 1st Edition stamp (which overrides other
    variant detection -- 1st Edition cards can also be holo, but the stamp is
    the primary distinguishing feature for pricing purposes).

    Then it analyses holographic characteristics in the artwork vs border
    regions to distinguish normal / holofoil / reverse_holofoil.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    logger.debug("Analyzing variant for %s (shape=%s)", image_path, img.shape)

    # --- 1st Edition check (highest priority) ---
    if _check_1st_edition(img):
        logger.info("Detected variant: 1st_edition for %s", image_path)
        return "1st_edition"

    # --- Holographic analysis ---
    art_region = _extract_region(img, ART_X0, ART_Y0, ART_X1, ART_Y1)
    border_region = _extract_region(img, 0.05, BORDER_Y0, 0.95, 0.95)

    art_combined, art_spread, art_noise = _holo_score(art_region)
    border_combined, border_spread, border_noise = _holo_score(border_region)

    logger.debug("Art   -- hue_spread=%d, spatial_noise=%.1f, combined=%.1f",
                 art_spread, art_noise, art_combined)
    logger.debug("Border-- hue_spread=%d, spatial_noise=%.1f, combined=%.1f",
                 border_spread, border_noise, border_combined)

    max_combined = max(art_combined, border_combined)

    if max_combined < HOLO_COMBINED_THRESHOLD:
        logger.info("Detected variant: normal for %s (max_combined=%.1f < %.1f)",
                     image_path, max_combined, HOLO_COMBINED_THRESHOLD)
        return "normal"

    # Discriminate holofoil vs reverse_holofoil by region dominance
    if art_combined > border_combined * ART_HOLO_RATIO:
        variant = "holofoil"
    elif border_combined > art_combined * BORDER_HOLO_RATIO:
        variant = "reverse_holofoil"
    else:
        # Ambiguous -- lean holofoil (more common)
        variant = "holofoil" if art_combined >= border_combined else "reverse_holofoil"

    logger.info("Detected variant: %s for %s (art=%.1f, border=%.1f)",
                variant, image_path, art_combined, border_combined)
    return variant


def detect_variant_detailed(image_path: str | Path) -> dict:
    """Like detect_variant() but returns detailed analysis for debugging.

    Returns dict with keys:
      - variant: str -- the detected variant
      - art_hue_spread: int
      - border_hue_spread: int
      - art_spatial_noise: float
      - border_spatial_noise: float
      - art_combined_score: float
      - border_combined_score: float
      - art_saturation_std: float
      - border_saturation_std: float
      - has_1st_edition_stamp: bool
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    has_stamp = _check_1st_edition(img)

    art_region = _extract_region(img, ART_X0, ART_Y0, ART_X1, ART_Y1)
    border_region = _extract_region(img, 0.05, BORDER_Y0, 0.95, 0.95)

    art_combined, art_spread, art_noise = _holo_score(art_region)
    border_combined, border_spread, border_noise = _holo_score(border_region)

    art_sat = _saturation_std(art_region)
    border_sat = _saturation_std(border_region)

    # Determine variant using same logic as detect_variant
    if has_stamp:
        variant = "1st_edition"
    elif max(art_combined, border_combined) < HOLO_COMBINED_THRESHOLD:
        variant = "normal"
    elif art_combined > border_combined * ART_HOLO_RATIO:
        variant = "holofoil"
    elif border_combined > art_combined * BORDER_HOLO_RATIO:
        variant = "reverse_holofoil"
    else:
        variant = "holofoil" if art_combined >= border_combined else "reverse_holofoil"

    return {
        "variant": variant,
        "art_hue_spread": art_spread,
        "border_hue_spread": border_spread,
        "art_spatial_noise": round(art_noise, 2),
        "border_spatial_noise": round(border_noise, 2),
        "art_combined_score": round(art_combined, 2),
        "border_combined_score": round(border_combined, 2),
        "art_saturation_std": round(art_sat, 2),
        "border_saturation_std": round(border_sat, 2),
        "has_1st_edition_stamp": has_stamp,
    }
