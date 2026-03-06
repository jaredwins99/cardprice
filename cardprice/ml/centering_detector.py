"""Measure card centering from perspective-corrected card segments.

Centering = how symmetric the colored border is around the artwork frame.
Pokemon cards typically have yellow borders; off-center printing shifts the
artwork frame relative to the card edges, making one border thicker than
its opposite.

Algorithm:
  1. Convert to HSV and mask border-colored pixels (yellow, silver, etc.)
  2. Scan horizontal/vertical lines at multiple positions to find the
     artwork frame edges (first non-border pixel from each side)
  3. Median-filter the measurements for robustness
  4. Compute left/right and top/bottom ratios
  5. Map to PSA centering grade (10 = perfect, 6 = 75/25)

Usage:
    from cardprice.ml.centering_detector import measure_centering
    result = measure_centering("data/test_binder_segments/card_03.png")
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _make_border_mask(hsv: np.ndarray) -> np.ndarray:
    """Create binary mask of border-colored pixels.

    Tries yellow first (most common). If insufficient border pixels are
    found, falls back to Canny edge detection to find the artwork frame.
    """
    h, w = hsv.shape[:2]

    # Yellow border: H 12-35, S 60-255, V 130-255
    yellow = cv2.inRange(hsv, (12, 60, 130), (35, 255, 255))

    # Check if yellow covers a reasonable fraction of the outer border
    border_strip = max(int(min(h, w) * 0.06), 4)
    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:border_strip, :] = 255
    outer[-border_strip:, :] = 255
    outer[:, :border_strip] = 255
    outer[:, -border_strip:] = 255

    yellow_in_border = cv2.bitwise_and(yellow, outer)
    coverage = np.count_nonzero(yellow_in_border) / max(np.count_nonzero(outer), 1)

    if coverage > 0.3:
        return yellow

    # Try silver/gray: low saturation, moderate-high value
    silver = cv2.inRange(hsv, (0, 0, 140), (180, 40, 255))
    silver_in_border = cv2.bitwise_and(silver, outer)
    silver_coverage = np.count_nonzero(silver_in_border) / max(np.count_nonzero(outer), 1)

    if silver_coverage > 0.3:
        return silver

    # Fallback: anything that is NOT the dark interior artwork region
    # Use value channel — borders tend to be lighter than card art
    _, bright = cv2.threshold(hsv[:, :, 2], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bright


def _scan_border_widths(mask: np.ndarray) -> dict:
    """Scan multiple lines to measure border width on each side.

    Only searches the outer 15% from each edge to avoid false matches
    from card artwork that shares the border color (e.g. yellow energy).

    Returns dict with keys left, right, top, bottom — each a median
    pixel distance from image edge to the artwork frame.
    """
    h, w = mask.shape
    max_lr = int(w * 0.15)  # max border width to consider
    max_tb = int(h * 0.15)

    # Sample scan positions: middle 60% to avoid corners
    y_positions = np.linspace(int(h * 0.2), int(h * 0.8), 15, dtype=int)
    x_positions = np.linspace(int(w * 0.2), int(w * 0.8), 15, dtype=int)

    lefts, rights, tops, bottoms = [], [], [], []

    def _first_zero_from_start(arr):
        """Index of first 0 in arr, or None."""
        idx = np.where(arr == 0)[0]
        return int(idx[0]) if len(idx) > 0 else None

    def _first_zero_from_end(arr):
        """Distance from end of arr to first 0, scanning backward."""
        idx = np.where(arr[::-1] == 0)[0]
        return int(idx[0]) if len(idx) > 0 else None

    # Horizontal scans (measure left and right borders)
    for y in y_positions:
        row = mask[y, :]
        l = _first_zero_from_start(row[:max_lr])
        if l is not None:
            lefts.append(l)
        r = _first_zero_from_end(row[-max_lr:])
        if r is not None:
            rights.append(r)

    # Vertical scans (measure top and bottom borders)
    for x in x_positions:
        col = mask[:, x]
        t = _first_zero_from_start(col[:max_tb])
        if t is not None:
            tops.append(t)
        b = _first_zero_from_end(col[-max_tb:])
        if b is not None:
            bottoms.append(b)

    # Median filter for robustness
    def safe_median(arr, fallback=0):
        return int(np.median(arr)) if arr else fallback

    return {
        "left": safe_median(lefts),
        "right": safe_median(rights),
        "top": safe_median(tops),
        "bottom": safe_median(bottoms),
    }


def _ratio_str(a: int, b: int) -> str:
    """Format a border ratio like '55/45'."""
    total = a + b
    if total == 0:
        return "50/50"
    pct_a = round(100 * a / total)
    pct_b = 100 - pct_a
    bigger = max(pct_a, pct_b)
    smaller = 100 - bigger
    return f"{bigger}/{smaller}"


def _ratio_to_grade(ratio: float) -> float:
    """Map a centering ratio (0.50-1.0) to PSA-style 1-10 grade.

    ratio = max(side_a, side_b) / (side_a + side_b)
    0.50 = perfect centering = 10
    0.55 = 10, 0.60 = 9, 0.65 = 8, 0.70 = 7, 0.75 = 6
    """
    if ratio <= 0.55:
        return 10.0
    if ratio >= 0.80:
        return max(5.0, 10.0 - (ratio - 0.50) * 20)
    # Linear interpolation: 0.55->10, 0.60->9, 0.65->8, 0.70->7, 0.75->6
    return round(10.0 - (ratio - 0.55) * 20, 1)


def measure_centering(image_path: str = "",
                      image: Optional[np.ndarray] = None) -> dict:
    """Measure card centering from a perspective-corrected card image.

    Args:
        image_path: Path to card segment image.
        image: Optional pre-loaded BGR numpy array.

    Returns:
        dict with:
            front_lr: str like "55/45" (left/right ratio)
            front_tb: str like "52/48" (top/bottom ratio)
            centering_score: float 1-10 (PSA-style scale)
            confidence: float 0-1
            borders: dict with raw pixel measurements
    """
    if image is None:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build border mask
    mask = _make_border_mask(hsv)

    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Measure border widths
    borders = _scan_border_widths(mask)
    left, right = borders["left"], borders["right"]
    top, bottom = borders["top"], borders["bottom"]

    # Compute ratios
    lr_total = left + right
    tb_total = top + bottom
    lr_ratio = max(left, right) / lr_total if lr_total > 0 else 0.5
    tb_ratio = max(top, bottom) / tb_total if tb_total > 0 else 0.5

    # Confidence: based on border size (very thin = low confidence)
    min_border = min(left, right, top, bottom)
    border_frac = min_border / min(h, w) if min(h, w) > 0 else 0
    if border_frac < 0.01:
        # Almost no visible border — full art or bad crop
        confidence = 0.2
    elif border_frac < 0.02:
        confidence = 0.5
    else:
        confidence = 0.9

    # Grade: use the worse of LR and TB
    worst_ratio = max(lr_ratio, tb_ratio)
    centering_score = _ratio_to_grade(worst_ratio)

    result = {
        "front_lr": _ratio_str(left, right),
        "front_tb": _ratio_str(top, bottom),
        "centering_score": centering_score,
        "confidence": round(confidence, 2),
        "borders": borders,
    }

    logger.info("Centering: LR=%s TB=%s score=%.1f conf=%.2f borders=%s",
                result["front_lr"], result["front_tb"],
                centering_score, confidence, borders)

    return result


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(
        glob.glob("data/test_binder_segments/card_*.png"))

    if not paths:
        print("No images found. Pass paths as arguments or run from project root.")
        sys.exit(1)

    for f in paths:
        try:
            result = measure_centering(f)
            name = Path(f).name
            print(f"{name}: LR={result['front_lr']}  TB={result['front_tb']}  "
                  f"score={result['centering_score']}  conf={result['confidence']}  "
                  f"borders={result['borders']}")
        except Exception as e:
            print(f"{f}: ERROR {e}")
