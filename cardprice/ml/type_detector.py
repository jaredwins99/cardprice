"""Pokemon type detection from card background color.

Pokemon cards have distinctive background/border colors by type:
    Fire: red/orange          Grass: green
    Water: blue               Psychic: purple/pink
    Lightning: yellow         Fighting: brown/tan
    Colorless: white/tan      Dark: dark purple/black
    Metal: silver/gray        Dragon: gold

This module samples the card text-box region (the area below the artwork and
above the bottom info bar) and classifies the Pokemon type based on the
dominant color in HSV space.

Sampling strategy:
    - Sample the text-box background (y: 58-82%, x: 38-62% of card).
      This avoids the card border (which may be obscured by binder sleeves)
      and the artwork.
    - Also sample two thin strips just inside the left and right card edges
      at the text-box height (x: 20-28% and 72-80%).
    - Convert to HSV and classify each pixel by hue + saturation.
    - Low-saturation pixels with warm hue vote for Colorless (the tan/cream
      background of Normal-type cards).
    - Very low-saturation pixels vote for Metal or Colorless.
    - Very dark pixels vote for Dark type.

HSV calibration notes (OpenCV: H in [0,179]):
    - Psychic cards read H=170-179 (desaturated magenta near red wrap-around)
      with moderate saturation (S=70-100). Distinct from Fire which has S>120.
    - Colorless/Normal cards read H=14-20 with low saturation (S=40-80).
    - Fighting cards are similar hue but slightly higher saturation.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions keyed by HSV ranges.
# Each entry: (type_name, hue_low, hue_high, sat_low, sat_high)
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255]
#
# More specific rules are checked first; pixel is assigned to the first match.
# ---------------------------------------------------------------------------
_TYPE_RULES = [
    # Psychic: near-red hue BUT low saturation (desaturated magenta/pink).
    # On e-series and some modern cards, Psychic backgrounds read H=170-179
    # at relatively low saturation (S=50-95).  Fire backgrounds in the same
    # hue range are more saturated (S>=95).
    # Must be checked BEFORE Fire to avoid misclassification.
    ("Psychic", 170, 179, 50,  95),
    ("Psychic", 0,    3,  50,  95),   # slight wrap-around

    # Fire: saturated reds/oranges (H near 0/180, high saturation)
    ("Fire",      0,   12,  95, 255),
    ("Fire",    165,  179,  95, 255),

    # Fighting: orange-brown, moderate saturation
    ("Fighting", 10,   22,  60, 180),

    # Lightning: yellow (lowered sat floor to catch pale yellow cards)
    ("Lightning", 23,   36,  30, 255),

    # Dragon: gold (deep yellow)
    ("Dragon",   37,   44,  60, 255),

    # Grass: green
    ("Grass",    45,   85,  40, 255),

    # Water: blue
    ("Water",    86,  105,  40, 255),

    # Psychic: blue-purple / lavender (older base set style, H=100-155)
    ("Psychic", 106,  155,  40, 255),

    # Dark: deep magenta toward red, moderate sat
    ("Dark",    156,  169,  40, 255),
]

# Saturation threshold: below this, pixel is "gray/tan" -> Colorless or Metal.
_SAT_THRESHOLD = 40

# Among low-saturation pixels:
_VAL_DARK_THRESHOLD = 60     # very dark -> Dark type
_VAL_METAL_THRESHOLD = 140   # mid-value gray -> Metal
# Above _VAL_METAL_THRESHOLD -> Colorless

# For "warm gray" pixels (H=10-25, S=40-70): Colorless (tan card background).
# Kept tight on saturation to avoid eating into Fighting (brown, S=70+).
_COLORLESS_HUE_LO = 10
_COLORLESS_HUE_HI = 25
_COLORLESS_SAT_LO = 40
_COLORLESS_SAT_HI = 70


def _sample_border_region(img: np.ndarray) -> np.ndarray:
    """Extract pixels from the card text-box region that carry the type color.

    Targets the text-box background area, avoiding the artwork and avoiding
    the outer card edge which may be covered by binder sleeves.

    Returns a 2D array of pixels (N, 3) in BGR.
    """
    h, w = img.shape[:2]

    regions = []

    # Region 1: text-box center (y: 58-82%, x: 35-65%)
    # The most reliable area -- squarely in the text box background.
    # Kept wide enough to capture the type-colored background but narrow enough
    # to avoid binder sleeve bleed on the sides.
    ty1 = int(h * 0.58)
    ty2 = int(h * 0.82)
    tx1 = int(w * 0.35)
    tx2 = int(w * 0.65)
    regions.append(img[ty1:ty2, tx1:tx2])

    # Region 2: left inner strip at text-box height (x: 25-33%)
    lx1 = int(w * 0.25)
    lx2 = int(w * 0.33)
    regions.append(img[ty1:ty2, lx1:lx2])

    # Region 3: right inner strip at text-box height (x: 67-75%)
    rx1 = int(w * 0.67)
    rx2 = int(w * 0.75)
    regions.append(img[ty1:ty2, rx1:rx2])

    # Region 4: name bar area (y: 3-10%, x: 30-70%)
    # The top border above the art also carries the type color.
    # On reference images this is very reliable; on binder page scans
    # the top may be cropped tight but usually present.
    ny1 = int(h * 0.03)
    ny2 = int(h * 0.10)
    nx1 = int(w * 0.30)
    nx2 = int(w * 0.70)
    regions.append(img[ny1:ny2, nx1:nx2])

    # Flatten to pixel list
    pixels = np.vstack([r.reshape(-1, 3) for r in regions if r.size > 0])
    return pixels


def _classify_pixels(pixels_hsv: np.ndarray) -> dict:
    """Classify an array of HSV pixels into type vote buckets.

    Returns dict mapping type name -> total vote weight.
    Pixels are assigned to the first matching rule, so rule order matters.
    """
    votes: dict[str, float] = {}

    h_chan = pixels_hsv[:, 0].astype(np.float32)
    s_chan = pixels_hsv[:, 1].astype(np.float32)
    v_chan = pixels_hsv[:, 2].astype(np.float32)

    n = len(pixels_hsv)
    if n == 0:
        return votes

    # Track which pixels have been assigned already
    assigned = np.zeros(n, dtype=bool)

    # --- Achromatic pixels (very low saturation) ---
    low_sat = s_chan < _SAT_THRESHOLD

    dark_mask = low_sat & (v_chan < _VAL_DARK_THRESHOLD)
    metal_mask = low_sat & (~dark_mask) & (v_chan < _VAL_METAL_THRESHOLD)
    colorless_mask = low_sat & (~dark_mask) & (v_chan >= _VAL_METAL_THRESHOLD)

    votes["Dark"] = float(np.sum(dark_mask))
    votes["Metal"] = float(np.sum(metal_mask))
    votes["Colorless"] = float(np.sum(colorless_mask))
    assigned |= low_sat

    # --- "Warm gray" pixels: Colorless (tan Normal-type background) ---
    # These have slightly above-threshold saturation but a warm hue and
    # relatively low saturation -- typical of Colorless-type card backgrounds.
    warm_gray = (~assigned
                 & (h_chan >= _COLORLESS_HUE_LO) & (h_chan <= _COLORLESS_HUE_HI)
                 & (s_chan >= _COLORLESS_SAT_LO) & (s_chan <= _COLORLESS_SAT_HI))
    votes["Colorless"] = votes.get("Colorless", 0) + float(np.sum(warm_gray))
    assigned |= warm_gray

    # --- Chromatic pixels: classify by hue+saturation rules ---
    # Use pixel count (not saturation-weighted) so that a few high-sat sleeve
    # bleed pixels don't outweigh the majority low-sat card background.
    # Require minimum brightness (V >= 30) to avoid noise from near-black pixels
    # where hue is unreliable.
    bright_enough = v_chan >= 30
    for type_name, hue_lo, hue_hi, sat_lo, sat_hi in _TYPE_RULES:
        mask = (~assigned & bright_enough
                & (h_chan >= hue_lo) & (h_chan <= hue_hi)
                & (s_chan >= sat_lo) & (s_chan <= sat_hi))
        count = float(np.sum(mask))
        if count > 0:
            votes[type_name] = votes.get(type_name, 0) + count
            assigned |= mask

    # Any remaining unassigned pixels get a small Colorless vote
    unassigned = ~assigned
    if np.any(unassigned):
        votes["Colorless"] = votes.get("Colorless", 0) + float(np.sum(unassigned))

    return votes


def detect_type(
    image_path,
    *,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Detect the Pokemon type from a card image based on border color.

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image.
    top_n : int
        Return the top N type predictions.

    Returns
    -------
    list of (type_name, confidence)
        Sorted by confidence descending.  Confidence is in [0, 1].

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image cannot be decoded.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return detect_type_from_array(img, top_n=top_n, label=image_path.name)


def detect_type_from_array(
    img: np.ndarray,
    *,
    top_n: int = 3,
    label: str = "<array>",
) -> List[Tuple[str, float]]:
    """Detect type from an already-loaded BGR image array.

    Parameters
    ----------
    img : numpy.ndarray
        BGR image (as from cv2.imread).
    top_n : int
        Number of top predictions to return.
    label : str
        Label for logging.

    Returns
    -------
    list of (type_name, confidence)
    """
    # Sample the border region
    border_pixels_bgr = _sample_border_region(img)

    if len(border_pixels_bgr) == 0:
        logger.warning("No border pixels sampled from %s", label)
        return [("Colorless", 0.0)]

    # Convert to HSV
    # Reshape to (N, 1, 3) for cvtColor, then back to (N, 3)
    border_pixels_bgr_3d = border_pixels_bgr.reshape(-1, 1, 3)
    border_pixels_hsv = cv2.cvtColor(border_pixels_bgr_3d, cv2.COLOR_BGR2HSV)
    border_pixels_hsv = border_pixels_hsv.reshape(-1, 3)

    # Classify
    votes = _classify_pixels(border_pixels_hsv)

    if not votes:
        return [("Colorless", 0.0)]

    # Normalize to confidence scores
    total = sum(votes.values())
    if total <= 0:
        return [("Colorless", 0.0)]

    scored = [(name, weight / total) for name, weight in votes.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Only keep entries with > 0 confidence
    scored = [(name, conf) for name, conf in scored if conf > 0.0]

    result = scored[:top_n]

    logger.debug(
        "Type detection for %s: %s",
        label,
        ", ".join(f"{name} ({conf:.0%})" for name, conf in result),
    )
    return result


# ---------------------------------------------------------------------------
# CLI test runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m cardprice.ml.type_detector <image> [image ...]")
        sys.exit(1)

    for p in paths:
        try:
            results = detect_type(p)
            top = results[0] if results else ("?", 0)
            alts = ", ".join(f"{n} {c:.0%}" for n, c in results[1:])
            print(f"{Path(p).name:30s}  ->  {top[0]:12s} ({top[1]:.0%})"
                  + (f"   alts: {alts}" if alts else ""))
        except Exception as e:
            print(f"{Path(p).name:30s}  ->  ERROR: {e}")
