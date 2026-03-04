"""Pokemon card border color and era analyzer using OpenCV.

Samples pixels from the card border (outer 5%) to classify the border color
and detect era-specific features like e-reader dot strips. Returns an era
estimate that narrows the search space from ~20,000 cards to ~2,000.

Border color patterns by era:
- Base Set through Neo (1999-2002): bright yellow borders
- e-Card / Expedition/Aquapolis/Skyridge (2002-2003): yellow + e-reader dots at bottom
- EX era (2003-2007): yellow (regular), silver/gray (ex Pokemon cards)
- Diamond/Pearl (2007-2009): yellow borders
- HeartGold/SoulSilver (2010): yellow borders
- Black & White (2011-2013): yellow borders (some silver full arts)
- XY (2014-2016): yellow borders (some silver/gray full arts)
- Sun & Moon (2017-2019): yellow borders (some silver/gray GX)
- Sword & Shield (2020-2022): dark/black borders (V/VMAX), yellow (regular)
- Scarlet & Violet (2023+): varied (silver, gray, colored borders)

Usage:
    from cardprice.ml.border_analyzer import analyze_border
    result = analyze_border("path/to/card.png")
    # result = {
    #     "border_color": "yellow",
    #     "era": "yellow_border",
    #     "era_sets": ["base1", "base2", ...],
    #     "has_ereader_dots": False,
    #     "confidence": 0.50,
    #     "border_hsv": (24, 150, 230),
    # }
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Set groupings by era (pokemontcg.io set IDs)
# ---------------------------------------------------------------------------
ERA_SETS = {
    "wotc": [
        "base1", "base2", "base3", "base4", "base5", "base6",
        "basep", "gym1", "gym2",
        "neo1", "neo2", "neo3", "neo4",
    ],
    "ecard": [
        "ecard1", "ecard2", "ecard3",
    ],
    "ex": [
        "ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "ex7", "ex8",
        "ex9", "ex10", "ex11", "ex12", "ex13", "ex14", "ex15", "ex16",
        "pop1", "pop2", "pop3", "pop4", "pop5",
        "tk1a", "tk1b",
    ],
    "dp": [
        "dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7", "dpp",
        "pop6", "pop7", "pop8", "pop9",
    ],
    "platinum": [
        "pl1", "pl2", "pl3", "pl4", "ru1", "si1",
    ],
    "hgss": [
        "hgss1", "hgss2", "hgss3", "hgss4", "hsp",
        "col1",  # Call of Legends
    ],
    "bw": [
        "bw1", "bw2", "bw3", "bw4", "bw5", "bw6", "bw7", "bw8",
        "bw9", "bw10", "bw11", "bwp", "dv1",
        "tk2a", "tk2b",
    ],
    "xy": [
        "xy0", "xy1", "xy2", "xy3", "xy4", "xy5", "xy6", "xy7",
        "xy8", "xy9", "xy10", "xy11", "xy12", "xyp",
        "dc1", "g1",
    ],
    "sm": [
        "sm1", "sm2", "sm3", "sm35", "sm4", "sm5", "sm6", "sm7",
        "sm75", "sm8", "sm9", "sm10", "sm11", "sm115", "sm12",
        "sma", "smp", "det1",
    ],
    "swsh": [
        "swsh1", "swsh2", "swsh3", "swsh35", "swsh4", "swsh45",
        "swsh45sv", "swsh5", "swsh6", "swsh7", "swsh8", "swsh9",
        "swsh9tg", "swsh10", "swsh10tg", "swsh11", "swsh11tg",
        "swsh12", "swsh12pt5", "swsh12pt5gg", "swsh12tg", "swshp",
        "cel25", "cel25c", "pgo",
    ],
    "sv": [
        "sv1", "sv2", "sv3", "sv3pt5", "sv4", "sv4pt5", "sv5",
        "sv6", "sv6pt5", "sv7", "sv8", "sv8pt5", "sv9", "sv10",
        "sve", "svp",
        "rsv10pt5", "zsv10pt5",
        "mcd11", "mcd12", "mcd16", "mcd19", "mcd21", "mcd22",
    ],
    # Miscellaneous promos and special sets (span multiple eras)
    "promo": [
        "np", "bp",                  # Nintendo/Best of promos (WotC era)
        "fut20",                     # Futures 2020
        "me1", "me2", "me2pt5",     # Mythical/Special collections
    ],
}

# Flat lookup: set_id -> era name
SET_TO_ERA = {}
for era, sets in ERA_SETS.items():
    for s in sets:
        SET_TO_ERA[s] = era

# ---------------------------------------------------------------------------
# Border color classification thresholds (HSV space)
# Measured from reference card images at various eras.
# H: 0-179, S: 0-255, V: 0-255 in OpenCV's HSV.
#
# Measured medians from reference images:
#   base1-4  (yellow):  H=23 S=183 V=237
#   neo1-1   (yellow):  H=27 S=254 V=245
#   ecard1-1 (yellow):  H=27 S=185 V=237
#   ex1-1    (yellow):  H=24 S=153 V=254
#   ex1-100  (silver):  H=50 S=9   V=225   <- ex Pokemon card
#   dp1-1    (yellow):  H=24 S=150 V=255
#   bw1-1    (yellow):  H=25 S=168 V=255
#   xy1-1    (mixed):   H=34 S=94  V=151   <- greenish tint from art
#   sm1-1    (yellow):  H=24 S=156 V=255
#   swsh1-1  (black):   H=21 S=5   V=37
#   sv1-1    (silver):  H=105 S=7  V=190
#   sv8-100  (silver):  H=105 S=8  V=198
# ---------------------------------------------------------------------------


def _sample_border_pixels(image: np.ndarray,
                          border_frac: float = 0.05) -> np.ndarray:
    """Sample pixels from the outer border of a card image.

    Takes pixels from the outer border_frac (default 5%) on all four edges,
    excluding the corners (where edges overlap) to avoid double-counting.

    Args:
        image: BGR card image.
        border_frac: Fraction of the smaller dimension to use as border width.

    Returns:
        Nx3 array of BGR pixel values from the border region.
    """
    h, w = image.shape[:2]
    border = max(int(min(h, w) * border_frac), 2)

    # Four edge strips, excluding corners to avoid overlap
    top = image[:border, border:-border]
    bottom = image[-border:, border:-border]
    left = image[border:-border, :border]
    right = image[border:-border, -border:]

    pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    return pixels


def _classify_border_color(hsv_pixels: np.ndarray) -> tuple[str, float]:
    """Classify border color from HSV pixel samples.

    Uses median (robust to outliers from card art bleeding into border)
    and fraction-of-matching-pixels for confidence.

    Returns (color_name, confidence) where confidence is 0.0-1.0.
    """
    if len(hsv_pixels) == 0:
        return "unknown", 0.0

    med_h = float(np.median(hsv_pixels[:, 0]))
    med_s = float(np.median(hsv_pixels[:, 1]))
    med_v = float(np.median(hsv_pixels[:, 2]))

    # Black / very dark border (SWSH V/VMAX cards)
    if med_v < 80:
        dark_frac = float(np.mean(hsv_pixels[:, 2] < 100))
        return "black", min(dark_frac + 0.3, 1.0)

    # Yellow border: hue ~14-35, saturation > 80, value > 150
    # This is the most common border color across many eras.
    # Check yellow BEFORE silver/gray since yellow is more specific.
    # Extends to H=35 to catch cards where art bleeds into thin borders.
    if 8 <= med_h <= 35 and med_s > 50 and med_v > 130:
        yellow_frac = float(np.mean(
            (hsv_pixels[:, 0] >= 6) & (hsv_pixels[:, 0] <= 35) &
            (hsv_pixels[:, 1] > 40) & (hsv_pixels[:, 2] > 110)
        ))
        return "yellow", min(yellow_frac + 0.2, 1.0)

    # Gold border: yellow hue but lower saturation/value than pure yellow
    if 12 <= med_h <= 28 and 50 <= med_s <= 150 and 130 <= med_v <= 220:
        gold_frac = float(np.mean(
            (hsv_pixels[:, 0] >= 10) & (hsv_pixels[:, 0] <= 30) &
            (hsv_pixels[:, 1] > 40) & (hsv_pixels[:, 2] < 230)
        ))
        return "gold", min(gold_frac * 0.8 + 0.2, 1.0)

    # Silver / gray border (low saturation, moderate-high value)
    # SV era: H~105, S~7, V~190 -- basically desaturated gray-blue
    # EX Pokemon: H~50, S~9, V~225 -- nearly white with low sat
    if med_s < 40 and med_v > 120:
        gray_frac = float(np.mean((hsv_pixels[:, 1] < 50) & (hsv_pixels[:, 2] > 100)))
        return "silver", min(gray_frac + 0.1, 1.0)

    # White border (very high value, very low saturation)
    if med_s < 25 and med_v > 210:
        white_frac = float(np.mean((hsv_pixels[:, 1] < 40) & (hsv_pixels[:, 2] > 200)))
        return "white", min(white_frac + 0.2, 1.0)

    # Blue border (some promos, energy cards)
    if 90 <= med_h <= 130 and med_s > 50:
        return "blue", 0.6

    # Green border (some XY cards have greenish tint)
    if 33 <= med_h <= 85 and med_s > 50 and med_v > 100:
        return "green", 0.5

    return "unknown", 0.3


def _detect_ereader_dots(image: np.ndarray) -> tuple[bool, float]:
    """Detect e-reader dot strip along the bottom border of a card.

    e-Card series (Expedition, Aquapolis, Skyridge) have a horizontal strip
    of small dots encoded for the e-Reader accessory. The dots occupy the
    bottom ~5% of the card height, in the middle ~70% of the width.

    Detection strategy:
    At typical binder segment resolution (630x880) the dots are visible but
    small. We use morphological analysis: the dot region has many small dark
    blobs arranged in horizontal rows, creating a distinctive pattern unlike
    the smooth/textured borders of other eras.

    We compare the bottom border strip against side border strips. The e-reader
    strip has significantly more high-frequency content (small blobs) than a
    plain yellow border.

    Args:
        image: BGR card image (any resolution).

    Returns:
        (has_dots, confidence) tuple.
    """
    h, w = image.shape[:2]

    # Need minimum resolution for meaningful analysis.
    # At 240x330 (reference thumbnails), dots are ~1-2px and unreliable.
    # At 630x880 (binder segments), dots are visible.
    min_useful_height = 400
    if h < min_useful_height:
        logger.debug("e-reader dots: image too small (%dx%d), skipping", w, h)
        return False, 0.0

    # --- Sample the bottom border strip (where e-reader dots would be) ---
    # The dot strip is in the very bottom of the card, within the border.
    # Use bottom 3-6% height, inner 60% width to avoid corner rounding.
    border_h = int(h * 0.05)
    bot_y1 = h - int(h * 0.06)
    bot_y2 = h - int(h * 0.02)
    x_margin = int(w * 0.20)
    bot_strip = image[bot_y1:bot_y2, x_margin:w - x_margin]

    # --- Sample a side border strip (reference, should NOT have dots) ---
    side_x = int(w * 0.02)
    side_strip = image[int(h * 0.3):int(h * 0.7), :side_x]

    if bot_strip.size == 0 or side_strip.size == 0:
        return False, 0.0

    bot_gray = cv2.cvtColor(bot_strip, cv2.COLOR_BGR2GRAY)
    side_gray = cv2.cvtColor(side_strip, cv2.COLOR_BGR2GRAY)

    bh, bw = bot_gray.shape

    # --- Measure high-frequency content using Laplacian variance ---
    # The Laplacian highlights edges/dots. A dot pattern has much higher
    # Laplacian variance than a smooth border.
    bot_lap = cv2.Laplacian(bot_gray, cv2.CV_64F)
    side_lap = cv2.Laplacian(side_gray, cv2.CV_64F)

    bot_lap_var = bot_lap.var()
    side_lap_var = side_lap.var()

    # --- Count small contours in the bottom strip ---
    # Threshold to find dark blobs on the lighter border
    blurred = cv2.GaussianBlur(bot_gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Dots are small: area between 2px and 1% of strip area
    strip_area = bh * bw
    max_dot_area = max(strip_area * 0.01, 20)
    min_dot_area = 2

    dot_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_dot_area <= area <= max_dot_area:
            dot_count += 1

    # --- Decision logic ---
    # e-reader dot strips have:
    # 1. Higher Laplacian variance in bottom vs sides (ratio > 2)
    # 2. Many small contours (scaled by strip width)
    # 3. The bottom strip should NOT be uniformly dark or light

    lap_ratio = bot_lap_var / max(side_lap_var, 1.0)

    # Scale dot threshold by strip width (reference: ~380px at 630w)
    scale = bw / 380.0
    dot_threshold = max(int(20 * scale), 5)

    bot_std = float(bot_gray.std())

    logger.debug("e-reader dots: lap_ratio=%.2f (bot=%.0f side=%.0f) "
                 "dots=%d (threshold=%d) bot_std=%.1f strip=%dx%d",
                 lap_ratio, bot_lap_var, side_lap_var,
                 dot_count, dot_threshold, bot_std, bw, bh)

    # Require BOTH high laplacian ratio AND many small dots
    # This prevents false positives from text/logos in the bottom border
    has_dots = (lap_ratio > 2.0 and dot_count >= dot_threshold and bot_std > 15)

    if has_dots:
        # Confidence increases with stronger signal
        conf = min(0.5 + (lap_ratio - 2.0) * 0.1 + (dot_count / (dot_threshold * 4)), 1.0)
    else:
        conf = 0.0

    return has_dots, conf


def _estimate_era(border_color: str, has_ereader: bool,
                  border_hsv: tuple[float, float, float]) -> tuple[str, list[str], float]:
    """Estimate the card era from border color and features.

    Returns (era_name, list_of_set_ids, confidence).
    """
    mean_h, mean_s, mean_v = border_hsv

    # e-reader dots are the strongest single signal
    if has_ereader:
        return "ecard", ERA_SETS["ecard"], 0.90

    if border_color == "yellow":
        # Yellow border is the most common across many eras.
        # Saturation helps narrow down but we must be inclusive:
        # - Very high sat (>200): strongly WotC/ecard era, but include
        #   all yellow-border eras to avoid false exclusion
        # - High sat (140-200): standard yellow, most eras
        # - Moderate sat (80-140): can include EX, DP, XY, some modern
        #
        # All yellow-border eras are always candidates. Saturation just
        # affects which eras are prioritized (confidence).
        yellow_eras = ["wotc", "ecard", "ex", "dp", "platinum", "hgss",
                       "bw", "xy", "sm", "promo"]
        if mean_s > 200:
            # Very saturated yellow = likely WotC/ecard/HGSS
            # Exclude modern eras that rarely have this saturated yellow
            yellow_eras = ["wotc", "ecard", "hgss", "bw", "ex", "dp",
                           "platinum", "sm", "promo"]
            conf = 0.55
        elif mean_s > 140:
            # Standard yellow = any yellow-bordered era including some SWSH
            yellow_eras = ["wotc", "ecard", "ex", "dp", "platinum", "hgss",
                           "bw", "xy", "sm", "swsh", "promo"]
            conf = 0.45
        else:
            # Lower saturation yellow = broader range
            yellow_eras = ["wotc", "ecard", "ex", "dp", "platinum", "hgss",
                           "bw", "xy", "sm", "swsh", "promo"]
            conf = 0.40

        all_sets = []
        for era in yellow_eras:
            all_sets.extend(ERA_SETS.get(era, []))
        return "yellow_border", all_sets, conf

    if border_color == "silver":
        # Silver/gray borders: modern SV era and some full-art cards
        candidate_eras = ["ex", "xy", "sm", "swsh", "sv", "promo"]
        all_sets = []
        for era in candidate_eras:
            all_sets.extend(ERA_SETS.get(era, []))
        return "modern_silver", all_sets, 0.55

    if border_color == "black":
        # Black borders are strongly modern (SWSH V/VMAX, some SV)
        candidate_eras = ["swsh", "sv", "promo"]
        all_sets = []
        for era in candidate_eras:
            all_sets.extend(ERA_SETS.get(era, []))
        return "modern_dark", all_sets, 0.75

    if border_color == "gold":
        # Gold = EX Pokemon cards, some modern alt arts
        candidate_eras = ["ex", "xy", "sm", "swsh", "sv"]
        all_sets = []
        for era in candidate_eras:
            all_sets.extend(ERA_SETS.get(era, []))
        return "gold_border", all_sets, 0.50

    if border_color == "white":
        # White borders are rare -- some promos, energy cards, ex Pokemon
        candidate_eras = ["ex", "xy", "sm", "swsh", "sv"]
        all_sets = []
        for era in candidate_eras:
            all_sets.extend(ERA_SETS.get(era, []))
        return "white_border", all_sets, 0.35

    # Unknown -- return everything
    all_sets = []
    for era in ERA_SETS:
        all_sets.extend(ERA_SETS[era])
    return "unknown", all_sets, 0.10


def analyze_border(image_path: str | Path = "",
                   image: Optional[np.ndarray] = None,
                   border_frac: float = 0.05) -> dict:
    """Analyze a card image's border to estimate its era.

    Can accept either a file path or a pre-loaded BGR numpy array.

    Args:
        image_path: Path to the card image (used if image is None).
        image: Optional pre-loaded BGR image array.
        border_frac: Fraction of card dimension to sample as border.

    Returns:
        Dict with keys:
            border_color: str - "yellow", "gold", "silver", "black", etc.
            era: str - era label like "wotc", "ecard", "modern_dark", etc.
            era_sets: list[str] - set IDs that match this era.
            has_ereader_dots: bool - True if e-reader dot strip detected.
            ereader_confidence: float - confidence of e-reader detection.
            confidence: float - overall confidence of era estimate.
            border_hsv: tuple - median (H, S, V) of border pixels.
    """
    if image is None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

    # Sample border pixels
    border_pixels_bgr = _sample_border_pixels(image, border_frac)

    # Convert to HSV for color classification
    border_pixels_hsv = cv2.cvtColor(
        border_pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # Classify border color
    border_color, color_conf = _classify_border_color(border_pixels_hsv)

    # Check for e-reader dots
    has_ereader, ereader_conf = _detect_ereader_dots(image)

    # Compute median HSV for diagnostics
    median_hsv = tuple(int(x) for x in np.median(border_pixels_hsv, axis=0))

    # Estimate era
    era, era_sets, era_conf = _estimate_era(
        border_color, has_ereader,
        (float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2]))
    )

    result = {
        "border_color": border_color,
        "era": era,
        "era_sets": era_sets,
        "has_ereader_dots": has_ereader,
        "ereader_confidence": round(ereader_conf, 3),
        "confidence": round(era_conf, 3),
        "border_hsv": median_hsv,
    }

    logger.info("Border analysis: color=%s era=%s sets=%d has_ereader=%s "
                "confidence=%.2f hsv=%s",
                border_color, era, len(era_sets), has_ereader,
                era_conf, median_hsv)

    return result


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print(f"Usage: python -m cardprice.ml.border_analyzer <image_path> [...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"  {path}")
        print(f"{'='*60}")
        try:
            result = analyze_border(path)
            print(f"  Border color : {result['border_color']}")
            print(f"  Era          : {result['era']}")
            print(f"  # sets       : {len(result['era_sets'])}")
            print(f"  e-reader dots: {result['has_ereader_dots']} "
                  f"(conf={result['ereader_confidence']:.2f})")
            print(f"  Confidence   : {result['confidence']:.2f}")
            print(f"  Border HSV   : {result['border_hsv']}")
            print(f"  Sets         : {result['era_sets'][:10]}"
                  f"{'...' if len(result['era_sets']) > 10 else ''}")
        except Exception as e:
            print(f"  ERROR: {e}")
