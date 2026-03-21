"""Era-gated stamp detection pipeline.

After a card is identified (card_id is known), this module checks for
physical stamps in the correct region based on the card's era and set.

Stamp types by era:
  - WotC (era 1): 1st_edition stamp, black_star_promo, prerelease
  - EX (era 2): ex_set_stamp (ex7-ex16), black_star_promo (np), prerelease
  - DP (era 3): promo stamp (dpp), prerelease (text-based)
  - HGSS+ (era 4-9): prerelease (set logo stamp), promo stamps
  - SWSH/SV (era 8-9): modern promo pokeball stamp (swshp/svp sets)

Each stamp type has a FIXED position on the card -- we go straight to
the expected region instead of scanning the whole card.

Prerelease detection:
  - WotC through DP (era 1-3): gold foil "PRERELEASE" text on artwork
    bottom-right. Multi-preprocessing OCR with fuzzy matching.
  - HGSS onward (era 4-9): expansion set logo stamp on artwork bottom-right.
    OCR for set name text in the stamp region.

Usage::

    from cardprice.ml.stamp_detection import detect_stamps

    result = detect_stamps("path/to/card.jpg", "base1-4/holofoil")
    print(result["stamps_detected"])   # ["1st_edition"]
    print(result["stamp_details"])     # {"1st_edition": {"confidence": 0.92, ...}}
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stamp type definitions with fixed regions (normalized x0, y0, x1, y1)
# ---------------------------------------------------------------------------
# Each stamp appears in a predictable position on the card.  Coordinates are
# fractions of card width/height.

STAMP_REGIONS = {
    # 1st Edition stamp: left side, between artwork and text box.
    # Small black circle with "1" and "EDITION" text.
    "1st_edition": {
        "wide": (0.02, 0.44, 0.24, 0.65),
        "tight": (0.03, 0.53, 0.15, 0.67),
        "position": "left",
    },
    # EX-era set logo stamp: bottom-right of artwork area.
    # Semi-transparent set name text overlaid on card art.
    "ex_set_stamp": {
        "wide": (0.50, 0.30, 0.90, 0.58),
        "tight": (0.55, 0.35, 0.88, 0.55),
        "position": "artwork_bottom_right",
    },
    # WotC Black Star promo star: bottom-right, near card number.
    # Black star with "PROMO" text.
    "black_star_promo": {
        "wide": (0.60, 0.88, 0.95, 0.98),
        "tight": (0.65, 0.90, 0.92, 0.97),
        "position": "bottom_right",
    },
    # Modern promo pokeball stamp: bottom-left corner, small black pokeball.
    # Used on SWSH and SV era promo cards.
    "modern_promo": {
        "wide": (0.02, 0.88, 0.20, 0.98),
        "tight": (0.04, 0.90, 0.16, 0.97),
        "position": "bottom_left",
    },
    # General promo stamp for DP/BW/XY/SM eras: bottom-left or bottom-right.
    # Various promo indicators (star, text, pokeball).
    "promo_stamp": {
        "wide": (0.02, 0.86, 0.25, 0.98),
        "tight": (0.04, 0.88, 0.20, 0.97),
        "position": "bottom_left",
    },
    # Pokemon Center exclusive stamp: bottom-right of artwork area.
    # Small red/white circular pokeball-derived logo on ETB promo cards.
    # SWSH/SV era SVP promos only.
    "pokemon_center": {
        "wide": (0.75, 0.50, 1.00, 0.75),
        "tight": (0.80, 0.55, 0.95, 0.70),
        "position": "artwork_bottom_right",
    },
    # Prerelease stamp: right half of artwork area.
    # Gold "PRERELEASE" text overlaid on card art.
    "prerelease": {
        "wide": (0.40, 0.14, 0.95, 0.55),
        "tight": (0.55, 0.30, 0.95, 0.55),
        "position": "artwork_bottom_right",
    },
    # Staff stamp: upper-right artwork area, near/above prerelease stamp.
    # Gold "STAFF" text, much rarer and more valuable than standard prerelease.
    "staff_stamp": {
        "wide": (0.55, 0.20, 0.95, 0.45),
        "position": "artwork_upper_right",
    },
    # Retailer-exclusive stamp: artwork area, similar to prerelease position.
    # Toys R Us (2016-2018) and Build-A-Bear Workshop stamps appear on the
    # card artwork, typically bottom-right or center-right of the art box.
    "retailer_stamp": {
        "wide": (0.10, 0.14, 0.95, 0.58),
        "tight": (0.40, 0.20, 0.95, 0.55),
        "position": "artwork",
    },
    # Build & Battle box promo stamp: rounded rectangle with pokeball graphic
    # (left half) and trainer silhouette (right half).  Located in the
    # bottom-left of the artwork area on SWSH/SV era SVP promo cards.
    # Tight position from N's Zoroark ex analysis: x:4.5-16.5%, y:41.5-47.5%.
    "build_battle": {
        "wide": (0.03, 0.38, 0.20, 0.52),
        "tight": (0.045, 0.415, 0.165, 0.475),
        "position": "artwork_bottom_left",
    },
}

# ---------------------------------------------------------------------------
# Era-to-stamp mapping: which stamps are possible for each era/set
# ---------------------------------------------------------------------------

# Sets that had 1st Edition print runs
_FIRST_EDITION_SETS = frozenset({
    "base1", "base2", "base3", "base5",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
})

# EX-era sets with set logo stamps on reverse holos
_EX_STAMPED_SETS = frozenset({
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
})

# Known stamp text for EX-era sets
_EX_STAMP_TEXT: dict[str, list[str]] = {
    "ex7":  ["team", "rocket", "returns"],
    "ex8":  ["deoxys"],
    "ex9":  ["emerald"],
    "ex10": ["unseen", "forces"],
    "ex11": ["delta", "species"],
    "ex12": ["legend", "maker"],
    "ex13": ["holon", "phantoms"],
    "ex14": ["crystal", "guardians"],
    "ex15": ["dragon", "frontiers"],
    "ex16": ["power", "keepers"],
}

_ALL_EX_STAMP_WORDS = frozenset({
    "team", "rocket", "returns", "deoxys", "emerald", "unseen", "forces",
    "delta", "species", "legend", "maker", "holon", "phantoms", "crystal",
    "guardians", "dragon", "frontiers", "power", "keepers",
})

# Black Star Promo sets by era
_BLACK_STAR_PROMO_SETS = frozenset({
    "basep",  # WotC Black Star Promos
    "np",     # Nintendo Black Star Promos (EX era)
})

# Promo sets by era (DP through SM)
_PROMO_SETS = frozenset({
    "dpp",    # DP promos
    "hsp",    # HGSS promos
    "bwp",    # BW promos
    "xyp",    # XY promos
    "smp",    # SM promos
})

# Modern promo sets (SWSH/SV era)
_MODERN_PROMO_SETS = frozenset({
    "swshp",  # Sword & Shield promos
    "svp",    # Scarlet & Violet promos
})

# ---------------------------------------------------------------------------
# Prerelease-eligible sets by era
# ---------------------------------------------------------------------------
# WotC/EX/DP: gold foil "PRERELEASE" text on artwork (text-based detection)
_PRERELEASE_TEXT_SETS = frozenset({
    # WotC era (1)
    "base2", "base3", "base5",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
    "ecard1", "ecard2", "ecard3",
    # EX era (2)
    "ex1", "ex2", "ex3", "ex4", "ex5", "ex6",
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
    # DP era (3) -- last era with text-based prerelease stamps
    "dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7",
    "pl1", "pl2", "pl3", "pl4",
})

# HGSS onward: set logo stamp on artwork (logo-based detection).
# The stamp shows the expansion set name/logo in the artwork area.
_PRERELEASE_LOGO_SETS = frozenset({
    # HGSS era (4)
    "hgss1", "hgss2", "hgss3", "hgss4", "col1",
    # BW era (5)
    "bw1", "bw2", "bw3", "bw4", "bw5", "bw6",
    "bw7", "bw8", "bw9", "bw10", "bw11",
    # XY era (6)
    "xy1", "xy2", "xy3", "xy4", "xy5", "xy6",
    "xy7", "xy8", "xy9", "xy10", "xy11", "xy12",
    # SM era (7)
    "sm1", "sm2", "sm3", "sm35", "sm4", "sm5", "sm6",
    "sm7", "sm75", "sm8", "sm9", "sm10", "sm11", "sm115", "sm12",
    # SWSH era (8)
    "swsh1", "swsh2", "swsh3", "swsh35", "swsh4", "swsh45",
    "swsh5", "swsh6", "swsh7", "swsh8", "swsh9", "swsh10",
    "swsh11", "swsh12", "swsh12pt5",
    # SV era (9)
    "sv1", "sv2", "sv3", "sv3pt5", "sv4", "sv4pt5",
    "sv5", "sv6", "sv6pt5", "sv7", "sv8", "sv8pt5", "sv9",
})

# Known set names for HGSS+ prerelease logo stamp OCR matching.
_PRERELEASE_LOGO_SET_TEXT: dict[str, list[str]] = {
    "hgss1": ["heartgold", "soulsilver"],
    "hgss2": ["unleashed"],
    "hgss3": ["undaunted"],
    "hgss4": ["triumphant"],
    "col1":  ["call", "legends"],
    "bw1":  ["black", "white"],
    "bw2":  ["emerging", "powers"],
    "bw3":  ["noble", "victories"],
    "bw4":  ["next", "destinies"],
    "bw5":  ["dark", "explorers"],
    "bw6":  ["dragons", "exalted"],
    "bw7":  ["boundaries", "crossed"],
    "bw8":  ["plasma", "storm"],
    "bw9":  ["plasma", "freeze"],
    "bw10": ["plasma", "blast"],
    "bw11": ["legendary", "treasures"],
    "xy1":  ["xy"],
    "xy2":  ["flashfire"],
    "xy3":  ["furious", "fists"],
    "xy4":  ["phantom", "forces"],
    "xy5":  ["primal", "clash"],
    "xy6":  ["roaring", "skies"],
    "xy7":  ["ancient", "origins"],
    "xy8":  ["breakthrough"],
    "xy9":  ["breakpoint"],
    "xy10": ["fates", "collide"],
    "xy11": ["steam", "siege"],
    "xy12": ["evolutions"],
    "sm1":  ["sun", "moon"],
    "sm2":  ["guardians", "rising"],
    "sm3":  ["burning", "shadows"],
    "sm35": ["shining", "legends"],
    "sm4":  ["crimson", "invasion"],
    "sm5":  ["ultra", "prism"],
    "sm6":  ["forbidden", "light"],
    "sm7":  ["celestial", "storm"],
    "sm75": ["dragon", "majesty"],
    "sm8":  ["lost", "thunder"],
    "sm9":  ["team", "up"],
    "sm10": ["unbroken", "bonds"],
    "sm11": ["unified", "minds"],
    "sm115": ["hidden", "fates"],
    "sm12": ["cosmic", "eclipse"],
    "swsh1":  ["sword", "shield"],
    "swsh2":  ["rebel", "clash"],
    "swsh3":  ["darkness", "ablaze"],
    "swsh35": ["champion", "path"],
    "swsh4":  ["vivid", "voltage"],
    "swsh45": ["shining", "fates"],
    "swsh5":  ["battle", "styles"],
    "swsh6":  ["chilling", "reign"],
    "swsh7":  ["evolving", "skies"],
    "swsh8":  ["fusion", "strike"],
    "swsh9":  ["brilliant", "stars"],
    "swsh10": ["astral", "radiance"],
    "swsh11": ["lost", "origin"],
    "swsh12": ["silver", "tempest"],
    "swsh12pt5": ["crown", "zenith"],
    "sv1":    ["scarlet", "violet"],
    "sv2":    ["paldea", "evolved"],
    "sv3":    ["obsidian", "flames"],
    "sv3pt5": ["151"],
    "sv4":    ["paradox", "rift"],
    "sv4pt5": ["paldean", "fates"],
    "sv5":    ["temporal", "forces"],
    "sv6":    ["twilight", "masquerade"],
    "sv6pt5": ["shrouded", "fable"],
    "sv7":    ["stellar", "crown"],
    "sv8":    ["surging", "sparks"],
    "sv8pt5": ["prismatic", "evolutions"],
    "sv9":    ["journey", "together"],
}

# ---------------------------------------------------------------------------
# Retailer-exclusive stamp definitions
# ---------------------------------------------------------------------------
# Toys R Us exclusive promos (2016-2018): stamped with "Toys R Us" text.
# Build-A-Bear Workshop exclusives: stamped with "Build-A-Bear Workshop".
# These stamps appear on the artwork area, similar to prerelease stamps.
# Era range: XY through SM (eras 6-7), roughly 2014-2019.

_RETAILER_STAMP_KEYWORDS: dict[str, list[list[str]]] = {
    # retailer_name -> list of keyword groups (any group matching = detection)
    "Toys R Us": [
        ["toys", "us"],        # "Toys R Us" or "TOYS 'R' US"
        ["toys", "r"],         # partial read: "TOYS R"
        ["toysrus"],           # run-together OCR
    ],
    "Build-A-Bear": [
        ["build", "bear"],           # "Build-A-Bear Workshop"
        ["build", "workshop"],       # partial read
        ["bear", "workshop"],        # partial read
        ["buildabear"],              # run-together OCR
    ],
}

# OCR confusion substitutions specific to retailer stamps
_RETAILER_OCR_SUBS: dict[str, str] = {
    "ioys": "toys",
    "t0ys": "toys",
    "bui1d": "build",
    "bu1ld": "build",
    "w0rkshop": "workshop",
    "worksho9": "workshop",
}

# Eras where retailer stamps are possible (BW=5, XY=6, SM=7)
_RETAILER_STAMP_ERAS = frozenset({5, 6, 7})


def _extract_set_id(card_id: str) -> str:
    """Extract set prefix from card_id like 'base1-4/holofoil' -> 'base1'."""
    bare = card_id.split("/")[0]
    return bare.rsplit("-", 1)[0] if "-" in bare else bare


def _get_era(card_id: str) -> int:
    """Get era number for a card_id (1-9, 0 if unknown)."""
    try:
        from cardprice.ml.era_detector import get_card_era
        return get_card_era(card_id)
    except Exception:
        return 0


def _extract_region(img: np.ndarray, x0: float, y0: float,
                    x1: float, y1: float) -> np.ndarray:
    """Extract a rectangular region from an image using fractional coords."""
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _ocr_region(region_bgr: np.ndarray) -> str:
    """Run RapidOCR on a region and return concatenated lowercase text.

    Upscales small regions and adds padding for better OCR accuracy.
    Returns empty string on any failure.
    """
    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        engine = get_rapid_engine()

        h, w = region_bgr.shape[:2]
        if h < 10 or w < 10:
            return ""

        # Upscale small regions -- OCR struggles below ~150px
        scale = max(1, 150 // max(h, 1))
        if scale > 1:
            region_up = cv2.resize(region_bgr, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
        else:
            region_up = region_bgr

        # Add padding so text isn't at the edge
        region_up = cv2.copyMakeBorder(region_up, 20, 20, 20, 20,
                                       cv2.BORDER_REPLICATE)

        # Upscale 3x for reliable detection on small stamp text
        region_up = cv2.resize(region_up, None, fx=3, fy=3,
                               interpolation=cv2.INTER_CUBIC)

        result, _ = engine(region_up)
        if not result:
            return ""

        texts = []
        for box, text, conf in result:
            if text and float(conf) > 0.3:
                texts.append(text.strip())

        return " ".join(texts).lower()
    except Exception as e:
        logger.debug("OCR stamp region failed: %s", e)
        return ""


def _has_dark_circular_blob(region_bgr: np.ndarray,
                            min_area_frac: float = 0.03,
                            max_area_frac: float = 0.30,
                            min_circularity: float = 0.65) -> bool:
    """Check if the region contains a dark circular blob."""
    try:
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        h, w = region_bgr.shape[:2]
        min_area = h * w * min_area_frac
        max_area = h * w * max_area_frac
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > min_circularity:
                    return True
    except Exception as e:
        logger.debug("Dark circular blob check failed: %s", e)
    return False


def _has_dark_circle_hough(region_bgr: np.ndarray) -> bool:
    """Detect dark circles using HoughCircles."""
    try:
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        h, w = region_bgr.shape[:2]

        # Blur and invert for dark circle detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        min_radius = max(3, int(w * 0.05))
        max_radius = max(10, int(w * 0.40))
        min_dist = max(5, int(w * 0.10))

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_dist, param1=80, param2=30,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None and len(circles[0]) > 0:
            return True
    except Exception as e:
        logger.debug("HoughCircles check failed: %s", e)
    return False



# ---------------------------------------------------------------------------
# 1st Edition thick vs thin stamp sub-variant
# ---------------------------------------------------------------------------
# Base Set holos exist with two stamp variants:
#   - "thick": bolder "1" numeral, ~60% of print run
#   - "thin":  thinner "1" numeral, ~40% of print run
# CGC and PSA note this distinction on slabs.
#
# Detection approach:
#   1. Crop the tight stamp region and convert to grayscale
#   2. Binarize with adaptive threshold (handles uneven lighting)
#   3. Find the largest dark contour that looks like the "1" numeral
#      (tall, narrow bounding box with aspect ratio > 1.5)
#   4. Measure stroke width as contour_area / contour_perimeter
#      (thicker strokes have higher area:perimeter ratio)
#   5. Compare against empirical threshold
#
# Feasibility caveat: binder scans typically yield stamp crops of ~30-60px
# wide.  At that resolution the "1" numeral is only 5-15px wide, making
# reliable thick/thin separation marginal.  The function reports "unknown"
# when the crop is too small or the measurement is ambiguous.

# Empirical stroke-width ratio threshold separating thick from thin.
# After adaptive threshold + morphological close + upscaling, the "1"
# contour area/perimeter ratio is typically:
#   - Thick stamps: 4.0-8.0 (wider stroke, serif, more area per perimeter)
#   - Thin stamps:  1.5-3.5 (narrower stroke, less area per perimeter)
# The threshold of 3.5 sits in the gap between these distributions.
# On low-res binder scans, morphological close inflates thin strokes,
# compressing the range -- so some "unknown" results are expected.
_THICK_THIN_STROKE_THRESHOLD = 3.5

# Minimum stamp crop dimensions (pixels) for reliable measurement.
# Below this, the "1" numeral is too small to measure stroke width.
_MIN_STAMP_CROP_PX = 25

# Confidence zones: how far from threshold counts as confident.
_STROKE_CONFIDENCE_MARGIN = 0.6


def _check_stamp_thickness(
    img_bgr: np.ndarray, stamp_region_crop: np.ndarray,
) -> tuple[str, float]:
    """After 1st edition detected, classify stamp as thick vs thin.

    Examines the "1" numeral in the 1st Edition stamp circle to determine
    whether it is the "thick" (bolder) or "thin" (lighter) print variant.
    This sub-variant distinction is noted by CGC/PSA for Base Set holos.

    Args:
        img_bgr: Full card image in BGR format (unused currently, reserved
            for future multi-region analysis).
        stamp_region_crop: Tight crop of the 1st Edition stamp area in BGR.
            Expected to contain the circular stamp with "1" and "EDITION".

    Returns:
        (classification, confidence) where classification is one of
        "thick", "thin", or "unknown", and confidence is 0.0-1.0.
    """
    if stamp_region_crop is None or stamp_region_crop.size == 0:
        return ("unknown", 0.0)

    h, w = stamp_region_crop.shape[:2]
    if h < _MIN_STAMP_CROP_PX or w < _MIN_STAMP_CROP_PX:
        logger.debug(
            "Stamp crop too small for thickness (%dx%d < %dpx)",
            w, h, _MIN_STAMP_CROP_PX,
        )
        return ("unknown", 0.0)

    try:
        # --- Step 1: Convert to grayscale ---
        gray = cv2.cvtColor(stamp_region_crop, cv2.COLOR_BGR2GRAY)

        # Upscale small crops for better contour detection.
        # Target at least 80px on the short side.
        scale = max(1, 80 // min(h, w))
        if scale > 1:
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # --- Step 2: Adaptive threshold (handles uneven binder lighting) ---
        # The stamp is dark ink on a lighter card surface, so THRESH_BINARY_INV
        # makes the stamp pixels white (255) and background black (0).
        block_size = max(11, (min(gray.shape) // 4) | 1)  # must be odd
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 5,
        )

        # Mild morphological close to connect broken strokes from low-res scans
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # --- Step 3: Find contours and identify the "1" numeral ---
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            logger.debug("No contours found in stamp crop")
            return ("unknown", 0.0)

        bh, bw = binary.shape[:2]
        total_area = bh * bw

        # Filter for contours that could be the "1" numeral:
        # - Tall and narrow (aspect ratio height/width > 1.5)
        # - Not too large (< 40% of crop) or too small (< 1% of crop)
        # - Located in the central area of the stamp
        numeral_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * 0.01 or area > total_area * 0.40:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0 or cw == 0:
                continue

            aspect = ch / cw
            if aspect < 1.5:
                # "1" is tall and narrow; skip wide/squat shapes
                continue

            # The "1" should be roughly in the center-left of the stamp circle
            cx = x + cw / 2
            cy = y + ch / 2
            if cx < bw * 0.15 or cx > bw * 0.70:
                continue
            if cy < bh * 0.10 or cy > bh * 0.75:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue

            stroke_ratio = area / perimeter
            numeral_candidates.append({
                "contour": cnt,
                "area": area,
                "perimeter": perimeter,
                "stroke_ratio": stroke_ratio,
                "aspect": aspect,
                "bbox": (x, y, cw, ch),
            })

        if not numeral_candidates:
            # Fallback: try the largest tall contour anywhere in the crop
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                if area < total_area * 0.005:
                    break
                x, y, cw, ch = cv2.boundingRect(cnt)
                if ch == 0 or cw == 0:
                    continue
                aspect = ch / cw
                if aspect < 1.3:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1.0:
                    continue
                stroke_ratio = area / perimeter
                numeral_candidates.append({
                    "contour": cnt,
                    "area": area,
                    "perimeter": perimeter,
                    "stroke_ratio": stroke_ratio,
                    "aspect": aspect,
                    "bbox": (x, y, cw, ch),
                })
                break  # take just the largest fallback

        if not numeral_candidates:
            logger.debug("No '1' numeral candidate found in stamp crop")
            return ("unknown", 0.0)

        # Pick the best candidate: tallest aspect ratio among the larger ones
        # (prefer the one that looks most like a "1")
        best = max(
            numeral_candidates,
            key=lambda c: c["aspect"] * (c["area"] / total_area),
        )

        stroke_ratio = best["stroke_ratio"]
        logger.debug(
            "Stamp '1' numeral: stroke_ratio=%.3f, area=%d, perimeter=%.1f, "
            "aspect=%.2f, bbox=%s",
            stroke_ratio, best["area"], best["perimeter"],
            best["aspect"], best["bbox"],
        )

        # --- Step 4: Classify based on stroke width ratio ---
        delta = stroke_ratio - _THICK_THIN_STROKE_THRESHOLD

        if abs(delta) < _STROKE_CONFIDENCE_MARGIN * 0.3:
            # Too close to threshold -- ambiguous
            confidence = 0.3
            classification = "thick" if delta >= 0 else "thin"
            logger.debug(
                "Stamp thickness ambiguous (delta=%.3f): %s @ %.2f",
                delta, classification, confidence,
            )
        else:
            # Confidence scales with distance from threshold, capped at 0.85
            # (never 1.0 because binder scans are inherently noisy)
            confidence = min(
                0.85,
                0.5 + abs(delta) / (_STROKE_CONFIDENCE_MARGIN * 2),
            )
            classification = "thick" if delta >= 0 else "thin"
            logger.debug(
                "Stamp thickness: %s (ratio=%.3f, delta=%.3f, conf=%.2f)",
                classification, stroke_ratio, delta, confidence,
            )

        return (classification, confidence)

    except Exception as e:
        logger.debug("Stamp thickness detection failed: %s", e)
        return ("unknown", 0.0)

# ---------------------------------------------------------------------------
# Individual stamp detection functions
# ---------------------------------------------------------------------------

def _check_1st_edition(img_bgr: np.ndarray) -> dict:
    """Check for 1st Edition stamp.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    When detected, also includes 'stamp_thickness' and
    'thickness_confidence' from thick/thin sub-variant analysis.
    """
    regions = STAMP_REGIONS["1st_edition"]

    # Wide region OCR
    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return {"detected": False, "confidence": 0.0, "position": "left"}

    ocr_text = _ocr_region(wide)
    has_1st = "1st" in ocr_text
    has_edition = "edition" in ocr_text

    # Extract tight crop early (used for OCR fallback and thickness analysis)
    tight = _extract_region(img_bgr, *regions["tight"])

    def _add_thickness(result: dict) -> dict:
        """Append stamp thickness sub-variant to a detected result."""
        crop = tight if tight.size > 0 else wide
        thickness, thick_conf = _check_stamp_thickness(img_bgr, crop)
        result["stamp_thickness"] = thickness
        result["thickness_confidence"] = thick_conf
        return result

    if has_1st and has_edition:
        return _add_thickness({
            "detected": True, "confidence": 0.95,
            "position": "left", "evidence": "ocr_both_tokens",
            "ocr_text": ocr_text,
        })
    if has_1st or has_edition:
        return _add_thickness({
            "detected": True, "confidence": 0.85,
            "position": "left", "evidence": "ocr_one_token",
            "ocr_text": ocr_text,
        })

    # Tight region OCR
    if tight.size > 0:
        ocr_tight = _ocr_region(tight)
        has_1st_t = "1st" in ocr_tight
        has_edition_t = "edition" in ocr_tight

        if has_1st_t and has_edition_t:
            return _add_thickness({
                "detected": True, "confidence": 0.95,
                "position": "left", "evidence": "tight_ocr_both",
                "ocr_text": ocr_tight,
            })
        if has_1st_t or has_edition_t:
            return _add_thickness({
                "detected": True, "confidence": 0.85,
                "position": "left", "evidence": "tight_ocr_one",
                "ocr_text": ocr_tight,
            })
        combined_ocr = ocr_text + " " + ocr_tight
    else:
        combined_ocr = ocr_text

    # Circle detection + partial OCR evidence
    has_blob = _has_dark_circular_blob(wide)
    has_hough = _has_dark_circle_hough(tight if tight.size > 0 else wide)
    has_circle = has_blob or has_hough

    if has_circle and "1" in combined_ocr:
        method = "blob" if has_blob else "hough"
        return _add_thickness({
            "detected": True, "confidence": 0.70,
            "position": "left", "evidence": f"circle_{method}_plus_digit",
            "ocr_text": combined_ocr,
        })

    return {"detected": False, "confidence": 0.0, "position": "left"}


def _check_grey_stamp(img_bgr: np.ndarray, stamp_region_crop: np.ndarray
                      ) -> tuple[str, float]:
    """After 1st edition detected, check if stamp ink is grey vs black.

    Grey stamps have lighter ink -- the 1st Edition stamp appears grey instead
    of solid black.  CGC recognizes these as a separate sub-variant.

    Method:
      1. Convert the stamp region crop to grayscale.
      2. Find the darkest pixels (the ink) by taking the lowest 5th percentile.
      3. Classify based on the mean intensity of those darkest pixels:
         - darkest_mean < 60  -> black stamp (solid ink)
         - darkest_mean > 100 -> grey stamp (faded/light ink)
         - in between         -> unknown (needs manual review)

    Args:
        img_bgr: Full card image (unused, kept for API consistency).
        stamp_region_crop: BGR crop of the detected 1st Edition stamp area.

    Returns:
        Tuple of (ink_color, confidence) where ink_color is one of
        'black', 'grey', or 'unknown'.
    """
    if stamp_region_crop is None or stamp_region_crop.size == 0:
        return ("unknown", 0.0)

    gray = cv2.cvtColor(stamp_region_crop, cv2.COLOR_BGR2GRAY)

    # Flatten and find the darkest 5% of pixels (the ink strokes)
    pixels = gray.flatten()
    if len(pixels) == 0:
        return ("unknown", 0.0)

    threshold_count = max(1, len(pixels) // 20)  # 5th percentile
    darkest = np.partition(pixels, threshold_count)[:threshold_count]
    darkest_mean = float(np.mean(darkest))

    logger.debug("Grey stamp check: darkest_mean=%.1f (5th pct of %d pixels)",
                 darkest_mean, len(pixels))

    if darkest_mean < 60:
        # Strong dark ink -> black stamp
        # Confidence scales: darker = more confident it's black
        confidence = min(0.95, 0.70 + (60 - darkest_mean) / 100)
        return ("black", confidence)

    if darkest_mean > 100:
        # Light ink -> grey stamp
        # Confidence scales: lighter = more confident it's grey
        confidence = min(0.95, 0.70 + (darkest_mean - 100) / 100)
        return ("grey", confidence)

    # Ambiguous zone (60-100): could be either, needs calibration
    # Report which side it leans toward with low confidence
    midpoint = 80.0
    if darkest_mean < midpoint:
        lean = "black"
        confidence = 0.40 + (midpoint - darkest_mean) / 100
    else:
        lean = "grey"
        confidence = 0.40 + (darkest_mean - midpoint) / 100
    return (lean, confidence)


def _check_ex_set_stamp(img_bgr: np.ndarray, set_id: str) -> dict:
    """Check for EX-era set logo stamp on artwork.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["ex_set_stamp"]

    stamp_region = _extract_region(img_bgr, *regions["wide"])
    if stamp_region.size == 0:
        return {"detected": False, "confidence": 0.0,
                "position": "artwork_bottom_right"}

    ocr_text = _ocr_region(stamp_region)
    if not ocr_text:
        return {"detected": False, "confidence": 0.0,
                "position": "artwork_bottom_right"}

    logger.debug("EX stamp OCR: %r (set=%s)", ocr_text, set_id)

    # Check for set-specific stamp text
    if set_id in _EX_STAMP_TEXT:
        expected_words = _EX_STAMP_TEXT[set_id]
        matches = sum(1 for w in expected_words if w in ocr_text)
        if matches >= 1:
            conf = 0.90 if matches >= 2 else 0.80
            return {
                "detected": True, "confidence": conf,
                "position": "artwork_bottom_right",
                "evidence": f"set_specific_{matches}_words",
                "ocr_text": ocr_text, "set_id": set_id,
            }

    # Check for any known stamp text (set-agnostic)
    found_words = [w for w in _ALL_EX_STAMP_WORDS if w in ocr_text]
    if len(found_words) >= 2:
        return {
            "detected": True, "confidence": 0.75,
            "position": "artwork_bottom_right",
            "evidence": "generic_stamp_words",
            "ocr_text": ocr_text, "matched_words": found_words,
        }

    return {"detected": False, "confidence": 0.0,
            "position": "artwork_bottom_right"}


def _check_black_star_promo(img_bgr: np.ndarray) -> dict:
    """Check for Black Star Promo stamp (bottom-right, near card number).

    The black star promo stamp is a small black 5-pointed star, sometimes
    accompanied by "PROMO" text.  Located near the card number in the
    bottom-right area.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["black_star_promo"]
    result_base = {"detected": False, "confidence": 0.0, "position": "bottom_right"}

    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return result_base

    # OCR check for "PROMO" text
    ocr_text = _ocr_region(wide)
    has_promo = "promo" in ocr_text

    if has_promo:
        return {
            "detected": True, "confidence": 0.90,
            "position": "bottom_right",
            "evidence": "ocr_promo_text",
            "ocr_text": ocr_text,
        }

    # Look for a dark star shape (approximated as blob with moderate circularity)
    # Stars have lower circularity than circles (~0.3-0.5) but higher area
    has_blob = _has_dark_circular_blob(
        wide, min_area_frac=0.02, max_area_frac=0.40, min_circularity=0.25,
    )
    if has_blob:
        return {
            "detected": True, "confidence": 0.60,
            "position": "bottom_right",
            "evidence": "dark_star_blob",
        }

    return result_base


def _check_modern_promo(img_bgr: np.ndarray) -> dict:
    """Check for modern pokeball promo stamp (SWSH/SV era).

    The modern promo stamp is a small black pokeball icon in the bottom-left
    corner of the card, often near the set symbol area.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["modern_promo"]
    result_base = {"detected": False, "confidence": 0.0, "position": "bottom_left"}

    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return result_base

    # OCR for "PROMO" text (some modern promos have text)
    ocr_text = _ocr_region(wide)
    has_promo = "promo" in ocr_text

    if has_promo:
        return {
            "detected": True, "confidence": 0.85,
            "position": "bottom_left",
            "evidence": "ocr_promo_text",
            "ocr_text": ocr_text,
        }

    # Check for dark circular blob (pokeball is roughly circular)
    has_circle = _has_dark_circular_blob(
        wide, min_area_frac=0.02, max_area_frac=0.35, min_circularity=0.50,
    )

    # Also check with HoughCircles
    if not has_circle:
        has_circle = _has_dark_circle_hough(wide)

    if has_circle:
        return {
            "detected": True, "confidence": 0.65,
            "position": "bottom_left",
            "evidence": "pokeball_circle",
        }

    return result_base


def _check_copyright_year(img_bgr: np.ndarray, set_id: str) -> dict:
    """Check copyright line for 1999-2000 (4th print) vs 1999 (standard).

    Base Set had multiple print runs. The 4th print run (and later) updated
    the copyright line from "1999" to "1999-2000". This is a collector
    distinction with low price impact but notable for completeness.

    Detection region: x 5-50%, y 93-100% (bottom-left copyright text).
    Upscale 4x for reliable OCR on small footer text.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (only 'base1' is relevant).

    Returns:
        dict with 'detected', 'variant' ('4th_print'|'standard'|'unknown'),
        'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False, "variant": "unknown",
        "confidence": 0.0, "position": "bottom_center",
    }

    if set_id != "base1":
        return result_base

    # Crop the copyright footer: x 5-50%, y 93-100%
    h, w = img_bgr.shape[:2]
    y0, y1 = int(0.93 * h), h
    x0, x1 = int(0.05 * w), int(0.50 * w)
    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        return result_base

    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        engine = get_rapid_engine()

        # Upscale 4x for small copyright text
        crop_up = cv2.resize(crop, None, fx=4, fy=4,
                             interpolation=cv2.INTER_CUBIC)

        # Pad so text isn't at the edge
        crop_up = cv2.copyMakeBorder(crop_up, 10, 10, 10, 10,
                                     cv2.BORDER_REPLICATE)

        result_ocr, _ = engine(crop_up)
        if not result_ocr:
            return result_base

        # Concatenate all OCR text
        full_text = ""
        for box, text, conf in result_ocr:
            if text and float(conf) > 0.3:
                full_text += " " + text.strip()
        full_text = full_text.lower().strip()

        if not full_text:
            return result_base

        logger.debug("Copyright OCR text: %r", full_text)

        # Check for "1999-2000" or "2000" (4th print indicator)
        if "1999-2000" in full_text or "1999- 2000" in full_text:
            return {
                "detected": True, "variant": "4th_print",
                "confidence": 0.95, "position": "bottom_center",
                "evidence": "ocr_1999_2000", "ocr_text": full_text,
            }

        if "2000" in full_text:
            return {
                "detected": True, "variant": "4th_print",
                "confidence": 0.85, "position": "bottom_center",
                "evidence": "ocr_2000_only", "ocr_text": full_text,
            }

        # "1999" without "2000" means standard (1st-3rd print)
        if "1999" in full_text:
            return {
                "detected": True, "variant": "standard",
                "confidence": 0.90, "position": "bottom_center",
                "evidence": "ocr_1999_only", "ocr_text": full_text,
            }

        return result_base

    except Exception as e:
        logger.debug("Copyright year OCR failed: %s", e)
        return result_base



def _check_shadowless(img_bgr: np.ndarray, set_id: str) -> dict:
    """Check if a Base Set card is Shadowless (no right/bottom border shadow).

    Only applies to base1 cards.  Unlimited cards have a dark drop shadow
    along the right and bottom edges of the card frame.  Shadowless cards
    have a uniform, bright border with no shadow gradient.

    Method: compare mean brightness of an "outer" strip (near the card edge)
    to an "inner" strip (slightly inward).  On Unlimited cards the inner strip
    is darker (shadow), producing a positive outer-inner delta.  On Shadowless
    cards both strips are similarly bright.

    Returns dict with 'detected' (True = shadowless), 'confidence',
    'position', 'evidence', and diagnostic values.
    """
    result_base = {
        "detected": False, "confidence": 0.0,
        "position": "right_edge",
    }

    if set_id != "base1":
        return result_base

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return result_base

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Right edge analysis ---
    # Inner strip: x 87-90% (just inside the border, where shadow would be)
    # Outer strip: x 93-97% (the bright border edge itself)
    r_inner_x0 = int(0.87 * w)
    r_inner_x1 = int(0.90 * w)
    r_outer_x0 = int(0.93 * w)
    r_outer_x1 = int(0.97 * w)
    # Use middle 60% of height to avoid corners
    y_start = int(0.20 * h)
    y_end = int(0.80 * h)

    right_inner = gray[y_start:y_end, r_inner_x0:r_inner_x1]
    right_outer = gray[y_start:y_end, r_outer_x0:r_outer_x1]

    if right_inner.size == 0 or right_outer.size == 0:
        return result_base

    right_inner_mean = float(np.mean(right_inner))
    right_outer_mean = float(np.mean(right_outer))
    right_delta = right_outer_mean - right_inner_mean

    # --- Bottom edge analysis ---
    # Inner strip: y 87-90%
    # Outer strip: y 93-97%
    b_inner_y0 = int(0.87 * h)
    b_inner_y1 = int(0.90 * h)
    b_outer_y0 = int(0.93 * h)
    b_outer_y1 = int(0.97 * h)
    # Use middle 60% of width to avoid corners
    x_start = int(0.20 * w)
    x_end = int(0.80 * w)

    bottom_inner = gray[b_inner_y0:b_inner_y1, x_start:x_end]
    bottom_outer = gray[b_outer_y0:b_outer_y1, x_start:x_end]

    if bottom_inner.size == 0 or bottom_outer.size == 0:
        return result_base

    bottom_inner_mean = float(np.mean(bottom_inner))
    bottom_outer_mean = float(np.mean(bottom_outer))
    bottom_delta = bottom_outer_mean - bottom_inner_mean

    # Average the right and bottom deltas for a combined signal
    avg_delta = (right_delta + bottom_delta) / 2.0

    diagnostics = {
        "right_inner_mean": round(right_inner_mean, 1),
        "right_outer_mean": round(right_outer_mean, 1),
        "right_delta": round(right_delta, 1),
        "bottom_inner_mean": round(bottom_inner_mean, 1),
        "bottom_outer_mean": round(bottom_outer_mean, 1),
        "bottom_delta": round(bottom_delta, 1),
        "avg_delta": round(avg_delta, 1),
    }

    # Decision thresholds:
    # avg_delta < 5  => shadowless (uniform border, no shadow)
    # avg_delta > 15 => unlimited (clear shadow present)
    # 5-15 => ambiguous
    if avg_delta < 5:
        # Shadowless: uniform border brightness
        confidence = min(0.95, 0.70 + (5 - avg_delta) * 0.05)
        logger.info(
            "Shadowless detected: avg_delta=%.1f (right=%.1f, bottom=%.1f)",
            avg_delta, right_delta, bottom_delta,
        )
        return {
            "detected": True, "confidence": round(confidence, 2),
            "position": "right_edge",
            "evidence": "uniform_border",
            **diagnostics,
        }
    elif avg_delta > 15:
        # Unlimited: clear shadow
        logger.debug(
            "Unlimited (shadow): avg_delta=%.1f (right=%.1f, bottom=%.1f)",
            avg_delta, right_delta, bottom_delta,
        )
        return {
            "detected": False, "confidence": 0.0,
            "position": "right_edge",
            "evidence": "shadow_present",
            **diagnostics,
        }
    else:
        # Ambiguous zone (5-15)
        logger.debug(
            "Shadowless ambiguous: avg_delta=%.1f (right=%.1f, bottom=%.1f)",
            avg_delta, right_delta, bottom_delta,
        )
        if avg_delta < 10:
            confidence = round(0.50 - (avg_delta - 5) * 0.05, 2)
            return {
                "detected": True, "confidence": confidence,
                "position": "right_edge",
                "evidence": "ambiguous_lean_shadowless",
                **diagnostics,
            }
        return {
            "detected": False, "confidence": 0.0,
            "position": "right_edge",
            "evidence": "ambiguous_lean_unlimited",
            **diagnostics,
        }


def _check_promo_stamp(img_bgr: np.ndarray) -> dict:
    """Check for general promo stamp (DP/BW/XY/SM eras).

    Promo cards from these eras typically have a small star or "PROMO"
    marking in the bottom-left or bottom-right area near the card number.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["promo_stamp"]
    result_base = {"detected": False, "confidence": 0.0, "position": "bottom_left"}

    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return result_base

    ocr_text = _ocr_region(wide)
    has_promo = "promo" in ocr_text

    if has_promo:
        return {
            "detected": True, "confidence": 0.85,
            "position": "bottom_left",
            "evidence": "ocr_promo_text",
            "ocr_text": ocr_text,
        }

    # Also check the bottom-right for promo text (varies by era/set)
    br_region = _extract_region(img_bgr, 0.60, 0.86, 0.95, 0.98)
    if br_region.size > 0:
        br_ocr = _ocr_region(br_region)
        if "promo" in br_ocr:
            return {
                "detected": True, "confidence": 0.85,
                "position": "bottom_right",
                "evidence": "ocr_promo_text_br",
                "ocr_text": br_ocr,
            }

    # Check for dark star/blob shapes
    has_blob = _has_dark_circular_blob(
        wide, min_area_frac=0.02, max_area_frac=0.40, min_circularity=0.25,
    )
    if has_blob:
        return {
            "detected": True, "confidence": 0.55,
            "position": "bottom_left",
            "evidence": "dark_promo_blob",
        }

    return result_base


def _apply_retailer_ocr_subs(text: str) -> str:
    """Apply OCR confusion substitutions for retailer stamp keywords."""
    for wrong, right in _RETAILER_OCR_SUBS.items():
        text = text.replace(wrong, right)
    return text


def _check_retailer_stamp(img_bgr: np.ndarray, set_id: str,
                          era: int) -> dict:
    """Detect retailer-exclusive stamps (Toys R Us, Build-A-Bear, etc.).

    Retailer stamps are text overlays on the card artwork, similar in
    position to prerelease stamps.  They were used on exclusive promo
    cards distributed through specific retailers, primarily during the
    XY and Sun & Moon eras (2014-2019).

    Detection strategy:
      1. Crop the artwork region (wide and tight variants).
      2. Run OCR on the crop.
      3. Apply OCR confusion substitutions (common misreads).
      4. Match against known retailer keyword groups.
      5. Require at least one keyword group to fully match.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. "smp", "xyp").
        era: Era number (5-7 for BW/XY/SM).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence',
        and 'retailer' (the matched retailer name) when detected.
        Returns (retailer_name|None, confidence) is the simplified
        signature -- but we return a full dict for consistency with
        the other stamp checkers.
    """
    result_base = {
        "detected": False, "confidence": 0.0,
        "position": "artwork", "retailer": None,
    }

    # Era gate: retailer stamps only existed in BW/XY/SM eras
    if era not in _RETAILER_STAMP_ERAS and era != 0:
        return result_base

    regions = STAMP_REGIONS["retailer_stamp"]

    # --- Wide region scan (covers most of artwork area) ---
    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return result_base

    ocr_text_wide = _ocr_region(wide)
    ocr_text_wide = _apply_retailer_ocr_subs(ocr_text_wide)

    # --- Tight region scan (bottom-right artwork, where stamps usually are) ---
    tight = _extract_region(img_bgr, *regions["tight"])
    ocr_text_tight = ""
    if tight.size > 0:
        ocr_text_tight = _ocr_region(tight)
        ocr_text_tight = _apply_retailer_ocr_subs(ocr_text_tight)

    # Combine both OCR results for matching
    combined_text = ocr_text_wide + " " + ocr_text_tight

    logger.debug("Retailer stamp OCR (wide): %r", ocr_text_wide)
    if ocr_text_tight:
        logger.debug("Retailer stamp OCR (tight): %r", ocr_text_tight)

    if not combined_text.strip():
        return result_base

    # --- Match against retailer keyword groups ---
    best_retailer = None
    best_confidence = 0.0
    best_evidence = ""
    best_source = ""

    for retailer_name, keyword_groups in _RETAILER_STAMP_KEYWORDS.items():
        for group in keyword_groups:
            # Check tight region first (higher confidence -- less noise)
            tight_match = all(kw in ocr_text_tight for kw in group)
            wide_match = all(kw in ocr_text_wide for kw in group)

            if tight_match:
                # Full group match in tight region = high confidence
                n_keywords = len(group)
                conf = 0.90 if n_keywords >= 2 else 0.80
                if conf > best_confidence:
                    best_retailer = retailer_name
                    best_confidence = conf
                    best_evidence = f"tight_ocr_{n_keywords}_keywords"
                    best_source = ocr_text_tight
            elif wide_match:
                # Full group match in wide region = moderate confidence
                # (wider crop may pick up card text that coincidentally matches)
                n_keywords = len(group)
                conf = 0.80 if n_keywords >= 2 else 0.65
                if conf > best_confidence:
                    best_retailer = retailer_name
                    best_confidence = conf
                    best_evidence = f"wide_ocr_{n_keywords}_keywords"
                    best_source = ocr_text_wide

    if best_retailer is not None:
        logger.info(
            "Retailer stamp detected: %s (conf=%.2f, evidence=%s)",
            best_retailer, best_confidence, best_evidence,
        )
        return {
            "detected": True,
            "confidence": best_confidence,
            "position": "artwork",
            "evidence": best_evidence,
            "retailer": best_retailer,
            "ocr_text": best_source,
        }

    return result_base




def _check_pokemon_center_stamp(img_bgr: np.ndarray, set_id: str,
                                 era: int) -> dict:
    """Detect Pokemon Center exclusive stamp on ETB promo cards.

    The Pokemon Center stamp is a small (~5% card width) circular logo
    derived from the pokeball design, printed in red and white.  It appears
    in the bottom-right of the artwork area on SVP promos distributed
    through Pokemon Center Elite Trainer Boxes (SWSH/SV era, 2020+).

    Detection strategy:
      1. Crop the expected region (x:80-95%, y:55-70%).
      2. Build an HSV red mask (hue 0-10 or 170-180, sat > 80, val > 80).
      3. Find contours on the red mask; check for a circular blob of the
         right size (~0.5-15% of crop area).
      4. If a red circle is found, look for a white interior (pokeball
         split) as confirmation.
      5. OCR fallback for "POKEMON CENTER" text in the wider region.
      6. HoughCircles fallback on the red mask if contour method fails.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (only 'svp' is relevant).
        era: Card era number (8 = SWSH, 9 = SV).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False, "confidence": 0.0,
        "position": "artwork_bottom_right",
    }

    # Gate: only SVP promos in SWSH/SV era (era 0 = unknown, allow it)
    if set_id != "svp" or era not in (8, 9, 0):
        return result_base

    regions = STAMP_REGIONS["pokemon_center"]

    # --- Step 1: crop the tight region (higher signal) ---
    tight = _extract_region(img_bgr, *regions["tight"])
    if tight.size == 0:
        return result_base

    h_crop, w_crop = tight.shape[:2]
    crop_area = h_crop * w_crop

    # --- Step 2: red mask in HSV ---
    hsv = cv2.cvtColor(tight, cv2.COLOR_BGR2HSV)

    # Red wraps around hue 0/180 in OpenCV HSV (0-180 range).
    mask_lo = cv2.inRange(hsv, np.array([0, 80, 80]),
                          np.array([10, 255, 255]))
    mask_hi = cv2.inRange(hsv, np.array([170, 80, 80]),
                          np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # --- Step 3: find circular red blobs ---
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    best_circularity = 0.0
    best_contour = None
    min_blob_area = crop_area * 0.005   # 0.5% of crop
    max_blob_area = crop_area * 0.15    # 15% of crop

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area or area > max_blob_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt

    red_circle_found = best_circularity >= 0.55

    if red_circle_found and best_contour is not None:
        # --- Step 4: white interior check (pokeball split) ---
        x_b, y_b, w_b, h_b = cv2.boundingRect(best_contour)
        pad_x = max(1, int(w_b * 0.2))
        pad_y = max(1, int(h_b * 0.2))
        x0 = max(0, x_b + pad_x)
        y0 = max(0, y_b + pad_y)
        x1 = min(w_crop, x_b + w_b - pad_x)
        y1 = min(h_crop, y_b + h_b - pad_y)

        has_white = False
        white_ratio = 0.0
        if x1 > x0 and y1 > y0:
            interior = tight[y0:y1, x0:x1]
            gray_int = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
            white_pixels = int(np.sum(gray_int > 200))
            total_pixels = max(gray_int.size, 1)
            white_ratio = white_pixels / total_pixels
            has_white = white_ratio > 0.10

        if has_white:
            logger.debug("Pokemon Center stamp: red circle (circ=%.2f) "
                         "with white interior (%.1f%%)",
                         best_circularity, white_ratio * 100)
            return {
                "detected": True, "confidence": 0.90,
                "position": "artwork_bottom_right",
                "evidence": "red_circle_white_interior",
                "circularity": round(best_circularity, 3),
                "white_ratio": round(white_ratio, 3),
            }

        # Red circle without confirmed white interior
        logger.debug("Pokemon Center stamp: red circle (circ=%.2f) "
                     "no white interior", best_circularity)
        return {
            "detected": True, "confidence": 0.70,
            "position": "artwork_bottom_right",
            "evidence": "red_circle_only",
            "circularity": round(best_circularity, 3),
        }

    # --- Step 5: OCR fallback on wider region ---
    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size > 0:
        ocr_text = _ocr_region(wide)
        has_pokemon = "pokemon" in ocr_text or "pokémon" in ocr_text
        has_center = "center" in ocr_text or "centre" in ocr_text

        if has_pokemon and has_center:
            return {
                "detected": True, "confidence": 0.85,
                "position": "artwork_bottom_right",
                "evidence": "ocr_pokemon_center",
                "ocr_text": ocr_text,
            }
        if has_pokemon or has_center:
            return {
                "detected": True, "confidence": 0.60,
                "position": "artwork_bottom_right",
                "evidence": "ocr_partial",
                "ocr_text": ocr_text,
            }

    # --- Step 6: HoughCircles fallback on red mask ---
    red_pixels = cv2.countNonZero(red_mask)
    red_density = red_pixels / max(crop_area, 1)
    if red_density > 0.005:
        try:
            blurred_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
            min_r = max(3, int(min(h_crop, w_crop) * 0.03))
            max_r = max(10, int(min(h_crop, w_crop) * 0.25))
            circles = cv2.HoughCircles(
                blurred_mask, cv2.HOUGH_GRADIENT, dp=1.5,
                minDist=max(5, min_r * 2),
                param1=50, param2=20,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is not None and len(circles[0]) > 0:
                return {
                    "detected": True, "confidence": 0.55,
                    "position": "artwork_bottom_right",
                    "evidence": "hough_red_circle",
                    "red_density": round(red_density, 4),
                }
        except Exception as e:
            logger.debug("Pokemon Center HoughCircles fallback failed: %s", e)

    return result_base


# ---------------------------------------------------------------------------
# Prerelease and Staff stamp detection
# ---------------------------------------------------------------------------

# Known OCR misreadings of "PRERELEASE" -- fuzzy match handles most, but
# these exact substitutions catch the worst garbles.
_PRERELEASE_VARIANTS = frozenset({
    "prerelease", "pre-release",
    "prenelemee", "prereleas", "prereiease", "prerlease",
})


def _fuzzy_match_prerelease(text: str) -> tuple[bool, float]:
    """Check if text fuzzy-matches 'PRERELEASE'.

    Returns (is_match, score).  Uses the same logic as the standalone
    detect_prerelease.py script: length guard + fuzzy ratio >= 70.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return False, 0.0

    clean = text.strip().lower().replace(" ", "").replace("-", "")

    # Exact match on known variants
    if clean in _PRERELEASE_VARIANTS or text.strip().lower() in _PRERELEASE_VARIANTS:
        return True, 1.0

    # Length guard: "PRERELEASE" is 10 chars; partial reads need >= 6
    if len(clean) < 6 or len(clean) > 18:
        return False, 0.0

    score = fuzz.ratio(clean, "prerelease") / 100.0
    if score >= 0.70:
        return True, score

    return False, score


def _check_prerelease(img_bgr: np.ndarray, set_id: str = "",
                      era: int = 0) -> dict:
    """Check for PRERELEASE stamp on card artwork (era-gated dispatcher).

    Delegates to the era-appropriate detection method:
    - WotC/EX/DP (era 1-3): Multi-preprocessing OCR for gold foil
      "PRERELEASE" text on artwork. 6 preprocessing strategies (raw,
      unsharp, CLAHE, adaptive threshold, inverted, Otsu) with fuzzy
      matching including garbled OCR variants.
    - HGSS+ (era 4-9): Multi-preprocessing OCR for set logo stamp.
      Looks for both "PRERELEASE" text and set name words.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    if set_id in _PRERELEASE_LOGO_SETS:
        return _check_prerelease_logo(img_bgr, set_id)
    else:
        return _check_prerelease_text(img_bgr)


def _check_build_battle_stamp(img_bgr: np.ndarray, set_id: str,
                              era: int) -> dict:
    """Detect Build & Battle box promo stamp (pokeball + trainer silhouette).

    The stamp is a rounded rectangle in the bottom-left of the artwork area.
    Left half: pokeball graphic.  Right half: trainer silhouette.
    Rendered in red/white/dark colors against the card artwork.

    Detection uses pixel-level color analysis rather than OCR:
      1. Red channel analysis: stamp creates elevated red (R/max(G,B) ~1.87x)
      2. Mixed color profile: red + white + dark pixels co-occurring
      3. Pokeball blob: dark circular outline in the tight region
      4. Horizontal structure: left half redder than right
      5. Context comparison: red channel spike vs artwork above

    Calibrated from stamp_full_10x.png vs ref_same_spot_10x.png:
      Red ratio ~1.87x, red pixel fraction 0.41, white 0.025, dark 0.062.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (typically 'svp').
        era: Era number (8 = SWSH, 9 = SV).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["build_battle"]
    result_base = {
        "detected": False, "confidence": 0.0,
        "position": "artwork_bottom_left",
    }

    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return result_base
    h_w, w_w = wide.shape[:2]
    if h_w < 10 or w_w < 10:
        return result_base

    # --- Color analysis on wide region ---
    hsv = cv2.cvtColor(wide, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Red pixels: H < 15 or H > 165 (OpenCV 0-180), min sat/val
    red_mask = ((hue < 15) | (hue > 165)) & (sat > 40) & (val > 80)
    red_frac = float(red_mask.astype(np.float32).mean())
    # White pixels: low saturation, high value
    white_mask = (sat < 30) & (val > 180)
    white_frac = float(white_mask.astype(np.float32).mean())
    # Dark pixels: pokeball outline, trainer silhouette
    dark_mask = val < 60
    dark_frac = float(dark_mask.astype(np.float32).mean())

    # Red channel dominance
    b_ch, g_ch, r_ch = cv2.split(wide)
    r_mean = float(r_ch.mean())
    g_mean = float(g_ch.mean())
    b_mean = float(b_ch.mean())
    red_dominance = r_mean / max(g_mean, b_mean, 1.0)

    diagnostics = {
        "red_frac": round(red_frac, 4),
        "white_frac": round(white_frac, 4),
        "dark_frac": round(dark_frac, 4),
        "red_dominance": round(red_dominance, 3),
        "r_mean": round(r_mean, 1),
        "g_mean": round(g_mean, 1),
        "b_mean": round(b_mean, 1),
    }

    # --- Tight region analysis ---
    tight = _extract_region(img_bgr, *regions["tight"])
    tight_red_frac = 0.0
    tight_white_frac = 0.0
    tight_dark_frac = 0.0
    has_pokeball_blob = False

    if tight.size > 0 and tight.shape[0] >= 5 and tight.shape[1] >= 5:
        t_hsv = cv2.cvtColor(tight, cv2.COLOR_BGR2HSV)
        t_hue, t_sat, t_val = cv2.split(t_hsv)

        t_red = (((t_hue < 15) | (t_hue > 165))
                 & (t_sat > 40) & (t_val > 80))
        tight_red_frac = float(t_red.astype(np.float32).mean())
        t_white = (t_sat < 30) & (t_val > 180)
        tight_white_frac = float(t_white.astype(np.float32).mean())
        t_dark = t_val < 60
        tight_dark_frac = float(t_dark.astype(np.float32).mean())

        has_pokeball_blob = _has_dark_circular_blob(
            tight, min_area_frac=0.01, max_area_frac=0.25,
            min_circularity=0.45,
        )
        diagnostics.update({
            "tight_red_frac": round(tight_red_frac, 4),
            "tight_white_frac": round(tight_white_frac, 4),
            "tight_dark_frac": round(tight_dark_frac, 4),
            "has_pokeball_blob": has_pokeball_blob,
        })

    # --- Horizontal structure: pokeball (left) vs silhouette (right) ---
    left_redder = False
    if tight.size > 0 and tight.shape[1] >= 4:
        mid_x = tight.shape[1] // 2
        left_r = float(tight[:, :mid_x, 2].mean())
        right_r = float(tight[:, mid_x:, 2].mean())
        left_redder = left_r > right_r * 0.9
        diagnostics["left_r"] = round(left_r, 1)
        diagnostics["right_r"] = round(right_r, 1)

    # --- Context comparison: stamp region vs artwork above ---
    control_y0 = max(0.0, regions["wide"][1] - 0.14)
    control_y1 = regions["wide"][1]
    control = _extract_region(img_bgr, regions["wide"][0], control_y0,
                              regions["wide"][2], control_y1)
    red_ratio_vs_context = 1.0
    if control.size > 0:
        control_r = float(control[:, :, 2].mean())
        if control_r > 0:
            red_ratio_vs_context = r_mean / control_r
        diagnostics["control_r_mean"] = round(control_r, 1)
        diagnostics["red_ratio_vs_context"] = round(red_ratio_vs_context, 3)

    # --- Multi-signal scoring ---
    score = 0.0
    evidence_parts = []

    # Signal 1: Red fraction in tight region (stamp ~0.41)
    if tight_red_frac > 0.25:
        score += 0.20
        evidence_parts.append("tight_red_high")
    elif tight_red_frac > 0.15:
        score += 0.10
        evidence_parts.append("tight_red_moderate")

    # Signal 2: White pixels (stamp ~0.025)
    if tight_white_frac > 0.01:
        score += 0.10
        evidence_parts.append("white_pixels")

    # Signal 3: Mixed color profile (red + white + dark)
    mixed = tight_red_frac + tight_white_frac + tight_dark_frac
    if mixed > 0.35:
        score += 0.15
        evidence_parts.append("mixed_color_profile")
    elif mixed > 0.20:
        score += 0.08
        evidence_parts.append("moderate_mixed")

    # Signal 4: Red dominance (stamp ~1.87, artwork ~1.0)
    if red_dominance > 1.5:
        score += 0.15
        evidence_parts.append("strong_red_dominance")
    elif red_dominance > 1.3:
        score += 0.10
        evidence_parts.append("red_dominance")

    # Signal 5: Pokeball blob
    if has_pokeball_blob:
        score += 0.20
        evidence_parts.append("pokeball_blob")

    # Signal 6: Red ratio vs surrounding context
    if red_ratio_vs_context > 1.5:
        score += 0.15
        evidence_parts.append("red_spike_vs_context")
    elif red_ratio_vs_context > 1.2:
        score += 0.08
        evidence_parts.append("elevated_red_vs_context")

    # Signal 7: Left-right structure with pokeball
    if left_redder and has_pokeball_blob:
        score += 0.05
        evidence_parts.append("lr_structure")

    score = min(score, 1.0)
    detected = score >= 0.40
    evidence = "+".join(evidence_parts) if evidence_parts else "none"

    if detected:
        logger.info(
            "Build & Battle stamp detected (set=%s): score=%.2f evidence=%s "
            "red=%.3f white=%.3f dark=%.3f dom=%.2f",
            set_id, score, evidence, tight_red_frac, tight_white_frac,
            tight_dark_frac, red_dominance,
        )
    else:
        logger.debug(
            "Build & Battle stamp NOT detected (set=%s): score=%.2f "
            "red=%.3f dom=%.2f", set_id, score, tight_red_frac, red_dominance,
        )

    return {
        "detected": detected,
        "confidence": round(score, 3),
        "position": "artwork_bottom_left",
        "evidence": evidence,
        **diagnostics,
    }


def _check_staff_stamp(img_bgr: np.ndarray, set_id: str, era: int) -> dict:
    """Detect STAFF stamp on prerelease cards.

    Gold "STAFF" text appears near/below the prerelease stamp in the
    upper-right area of the card artwork.  These cards were given to
    tournament staff at prerelease events and are significantly rarer
    and more valuable than standard prerelease cards.

    Era: DP onward (2007-present, with gaps).
    Detection region: x:55-95%, y:20-45% (per stamp_positions.json).

    Strategy:
    1. Crop the staff stamp region and upscale 3x (handled by _ocr_region).
    2. Run RapidOCR on the crop.
    3. Fuzzy-match each OCR token against "STAFF".

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier string.
        era: Era number (1-9, 0 if unknown).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {"detected": False, "confidence": 0.0,
                   "position": "artwork_upper_right"}

    regions = STAMP_REGIONS["staff_stamp"]
    crop = _extract_region(img_bgr, *regions["wide"])
    if crop.size == 0:
        return result_base

    ocr_text = _ocr_region(crop)
    if not ocr_text:
        return result_base

    logger.debug("Staff stamp OCR: %r (set=%s, era=%d)", ocr_text, set_id, era)

    # Check each OCR token for a fuzzy match to "STAFF"
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return result_base

    # Known OCR misreadings of "STAFF"
    staff_variants = {"staff", "staf", "staef", "stalf", "stalff", "siaff",
                      "staiff", "stafe", "staft", "stoff"}

    for token in ocr_text.split():
        clean = token.strip().lower()

        # Skip very short or very long tokens
        if len(clean) < 3 or len(clean) > 8:
            continue

        # Exact match on known variants
        if clean in staff_variants:
            return {
                "detected": True, "confidence": 0.95,
                "position": "artwork_upper_right",
                "evidence": f"ocr_exact_variant_{clean}",
                "ocr_text": ocr_text,
            }

        # Fuzzy match against "staff"
        score = fuzz.ratio(clean, "staff") / 100.0
        if score >= 0.75:
            return {
                "detected": True, "confidence": 0.90 if score >= 0.85 else 0.80,
                "position": "artwork_upper_right",
                "evidence": f"ocr_fuzzy_{score:.2f}",
                "ocr_text": ocr_text,
            }

    return result_base


# ---------------------------------------------------------------------------
# Prerelease stamp detection (multi-preprocessing OCR)
# ---------------------------------------------------------------------------

def _ocr_region_multi(region_bgr: np.ndarray, scale: int = 3) -> list[tuple[str, float, str]]:
    """Run OCR with 6 preprocessing strategies on a region.

    Embossed/foil prerelease stamps are hard to read with a single
    preprocessing approach. Different strategies catch different stamp
    types (gold foil on dark art, silver foil on light art, etc.).

    Returns list of (text, ocr_confidence, strategy_name).
    """
    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        engine = get_rapid_engine()
    except Exception as e:
        logger.debug("Failed to get OCR engine for multi-preprocess: %s", e)
        return []

    h, w = region_bgr.shape[:2]
    if h < 10 or w < 10:
        return []

    upscaled = cv2.resize(region_bgr, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
    # Pad so text is not at the edge
    upscaled = cv2.copyMakeBorder(upscaled, 20, 20, 20, 20,
                                  cv2.BORDER_REPLICATE)

    results: list[tuple[str, float, str]] = []

    def _collect(ocr_result, strategy: str) -> None:
        if not ocr_result:
            return
        for line in ocr_result:
            text = line[1]
            conf = float(line[2])
            if text and conf > 0.2:
                results.append((text.strip(), conf, strategy))

    # Strategy 1: Raw color
    res, _ = engine(upscaled)
    _collect(res, "raw")

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Strategy 2: Unsharp mask
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    res, _ = engine(sharp)
    _collect(res, "sharp")

    # Strategy 3: CLAHE (good for embossed/low-contrast text)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    res, _ = engine(enhanced)
    _collect(res, "clahe")

    # Strategy 4: Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    res, _ = engine(thresh)
    _collect(res, "thresh")

    # Strategy 5: Inverted threshold
    thresh_inv = cv2.bitwise_not(thresh)
    res, _ = engine(thresh_inv)
    _collect(res, "thresh_inv")

    # Strategy 6: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res, _ = engine(otsu)
    _collect(res, "otsu")

    return results


def _is_prerelease_text(text: str) -> tuple[bool, float, str]:
    """Check if OCR text matches "PRERELEASE" using fuzzy matching.

    Uses strict length and character-composition guards to avoid false
    positives from random card text (e.g. "PowereEvolutlenary").

    Returns (is_match, score, method).
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.debug("rapidfuzz not available for prerelease matching")
        return (False, 0.0, "no_rapidfuzz")

    clean = text.strip().upper().replace(" ", "")

    # Too short to be "PRERELEASE" (10 chars)
    if len(clean) < 6:
        return (False, 0.0, "too_short")

    ratio = fuzz.ratio(clean, "PRERELEASE")

    # Character composition check: PRERELEASE letters are P, R, E, L, A, S
    prerelease_chars = set("PRERELASE")
    if len(clean) > 0:
        char_overlap = sum(1 for c in clean if c in prerelease_chars) / len(clean)
    else:
        char_overlap = 0.0

    # Substring checks for garbled OCR (PRENELEMEE, PRERELEAS, etc.)
    has_pre = "PRE" in clean or "PBE" in clean or "PIE" in clean
    has_rele = ("RELE" in clean or "RELF" in clean or "NELE" in clean
                or "RELI" in clean)

    # High confidence: fuzzy ratio >= 75, text is short (just the stamp)
    if ratio >= 75 and 7 <= len(clean) <= 15:
        return (True, ratio, "fuzzy_short")

    # Medium confidence: fuzzy ratio >= 70 with high character overlap
    if ratio >= 70 and char_overlap >= 0.7 and len(clean) >= 7:
        return (True, ratio, "fuzzy_overlap")

    # Substring match: contains both PRE and RELE-like substrings
    if has_pre and has_rele and 8 <= len(clean) <= 16:
        return (True, ratio, "substring")

    # Partial ratio: high partial match with good character overlap
    partial = fuzz.partial_ratio(clean, "PRERELEASE")
    if partial >= 90 and 8 <= len(clean) <= 20 and char_overlap >= 0.6:
        return (True, partial, "partial")

    return (False, max(ratio, partial), "none")


def _check_prerelease_text(img_bgr: np.ndarray) -> dict:
    """Check for WotC/EX/DP era PRERELEASE text stamp on artwork.

    Uses multi-preprocessing OCR across two artwork regions (right half
    and full artwork) with fuzzy matching against "PRERELEASE".

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["prerelease"]
    result_base = {"detected": False, "confidence": 0.0,
                   "position": "artwork_bottom_right"}

    # Check two regions: tight (right side of artwork) and wide (full)
    best_match = False
    best_score = 0.0
    best_text = ""
    best_strategy = ""
    best_method = ""

    for region_name, coords in [("tight", regions["tight"]),
                                ("wide", regions["wide"])]:
        crop = _extract_region(img_bgr, *coords)
        if crop.size == 0:
            continue

        ocr_results = _ocr_region_multi(crop, scale=3)
        for text, conf, strategy in ocr_results:
            is_match, score, method = _is_prerelease_text(text)
            tag = f"{region_name}/{strategy}"

            if is_match and score > best_score:
                best_match = True
                best_score = score
                best_text = text
                best_strategy = tag
                best_method = method
            elif not best_match and score > best_score:
                best_score = score
                best_text = text
                best_strategy = tag
                best_method = method

    if best_match:
        confidence = min(0.95, 0.70 + (best_score - 70) / 100)
        logger.debug("Prerelease text detected: %r score=%.1f via=%s method=%s",
                     best_text, best_score, best_strategy, best_method)
        return {
            "detected": True, "confidence": round(confidence, 2),
            "position": "artwork_bottom_right",
            "evidence": f"ocr_{best_method}",
            "ocr_text": best_text, "ocr_score": best_score,
            "ocr_strategy": best_strategy,
        }

    logger.debug("Prerelease text not found (best score=%.1f, text=%r)",
                 best_score, best_text)
    return result_base


def _check_prerelease_logo(img_bgr: np.ndarray, set_id: str) -> dict:
    """Check for HGSS+ era prerelease set logo stamp on artwork.

    Modern prerelease cards (HGSS onward) have the expansion set logo
    stamped on the artwork instead of "PRERELEASE" text. We look for
    the set name in OCR output from the stamp region.

    Also checks for "PRERELEASE" text since some modern prerelease cards
    still include it alongside the set logo.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["prerelease"]
    result_base = {"detected": False, "confidence": 0.0,
                   "position": "artwork_bottom_right"}

    # Get expected set name words
    expected_words = _PRERELEASE_LOGO_SET_TEXT.get(set_id, [])

    best_result = result_base
    best_conf = 0.0

    for region_name, coords in [("tight", regions["tight"]),
                                ("wide", regions["wide"])]:
        crop = _extract_region(img_bgr, *coords)
        if crop.size == 0:
            continue

        ocr_results = _ocr_region_multi(crop, scale=3)
        all_text_lower = " ".join(t.lower() for t, c, s in ocr_results if c > 0.3)

        # Check 1: "PRERELEASE" text (some modern cards have both)
        for text, conf, strategy in ocr_results:
            is_match, score, method = _is_prerelease_text(text)
            if is_match:
                tag = f"{region_name}/{strategy}"
                confidence = min(0.95, 0.70 + (score - 70) / 100)
                if confidence > best_conf:
                    best_conf = confidence
                    best_result = {
                        "detected": True, "confidence": round(confidence, 2),
                        "position": "artwork_bottom_right",
                        "evidence": f"ocr_prerelease_{method}",
                        "ocr_text": text, "ocr_score": score,
                        "ocr_strategy": tag,
                    }

        # Check 2: Set name words in OCR output
        if expected_words and all_text_lower:
            matches = sum(1 for w in expected_words if w in all_text_lower)
            if matches >= 1:
                total = len(expected_words)
                if total == 1:
                    conf = 0.85 if matches == 1 else 0.70
                else:
                    conf = 0.90 if matches >= 2 else 0.75
                if conf > best_conf:
                    best_conf = conf
                    best_result = {
                        "detected": True, "confidence": conf,
                        "position": "artwork_bottom_right",
                        "evidence": f"set_logo_{matches}_of_{total}_words",
                        "ocr_text": all_text_lower,
                        "matched_words": [w for w in expected_words
                                          if w in all_text_lower],
                    }

    if best_result.get("detected"):
        logger.debug("Prerelease logo detected for %s: evidence=%s",
                     set_id, best_result.get("evidence"))
    else:
        logger.debug("Prerelease logo not found for %s", set_id)

    return best_result




# ---------------------------------------------------------------------------
# Jungle no-symbol error detection
# ---------------------------------------------------------------------------

# The first Unlimited print run of Jungle holos (cards 1-16) is missing the
# Jungle flower set symbol in the bottom-right info bar.  This is a known
# error variant with collectible value.
#
# Detection: crop the set symbol region and measure whether it contains a
# distinct small shape (the flower).  If the region is featureless, it is
# likely the no-symbol error print.
#
# Feasibility caveat: the set symbol is tiny at binder-scan resolution
# (~10-15px), so detection confidence is inherently limited.

_JUNGLE_HOLO_NUMBERS = frozenset(range(1, 17))


def _check_no_symbol_error(img_bgr: np.ndarray, set_id: str,
                           card_rarity: str) -> tuple[bool, float]:
    """Check if a Jungle holo is missing its set symbol (error print).

    The first Unlimited print of Jungle (base2) holos omitted the flower
    set symbol from the bottom-right info bar.  This function crops that
    region and checks whether a distinct shape is present.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (only 'base2' is relevant).
        card_rarity: Rarity string -- must contain 'holo' (case-insensitive)
                     to be eligible.  Examples: "holofoil", "Rare Holo",
                     "1st_edition_holofoil", "unlimited_holofoil".

    Returns:
        (is_no_symbol, confidence) tuple.
        is_no_symbol=True means the set symbol appears to be MISSING (error).
        confidence is 0.0-1.0, reflecting detection certainty.
    """
    # Gate: only Jungle (base2) holo rares
    if set_id != "base2":
        return False, 0.0
    if not card_rarity or "holo" not in card_rarity.lower():
        return False, 0.0

    h, w = img_bgr.shape[:2]

    # Crop the set symbol region: x 80-95%, y 50-65%
    # This is where the Jungle flower should appear on the card.
    x0, x1 = int(0.80 * w), int(0.95 * w)
    y0, y1 = int(0.50 * h), int(0.65 * h)
    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return False, 0.0

    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_h * crop_w

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # --- Signal 1: Edge density (Laplacian variance) ---
    # A region with a symbol has more edges than a blank region.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_var = float(laplacian.var())

    # --- Signal 2: Contour analysis ---
    # Look for a small, compact shape (the flower symbol).
    # Adaptive threshold handles varying card backgrounds.
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter for symbol-sized contours (1-20% of crop area)
    symbol_contours = [
        cnt for cnt in contours
        if 0.01 * crop_area < cv2.contourArea(cnt) < 0.20 * crop_area
    ]

    # --- Signal 3: Canny edge pixel ratio ---
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / crop_area

    # --- Decision logic ---
    has_symbol_contour = len(symbol_contours) > 0
    has_strong_edges = edge_var > 200.0
    has_edge_content = edge_ratio > 0.05

    logger.debug(
        "Jungle no-symbol check: edge_var=%.1f, contours=%d, "
        "symbol_contours=%d, edge_ratio=%.3f",
        edge_var, len(contours), len(symbol_contours), edge_ratio,
    )

    # Symbol present: distinct shape found
    if has_symbol_contour and has_strong_edges:
        logger.debug("Jungle symbol PRESENT (contour + edges)")
        return False, 0.0

    # No symbol: region is featureless
    if not has_symbol_contour and not has_strong_edges and not has_edge_content:
        # High confidence: all three signals agree the region is blank
        logger.debug("Jungle symbol MISSING (no contours, no edges)")
        return True, 0.80

    if not has_symbol_contour and not has_edge_content:
        # Moderate confidence: no shape, low edge content
        logger.debug("Jungle symbol likely MISSING (no contours, low edges)")
        return True, 0.65

    if not has_symbol_contour:
        # Low confidence: no shape but some edge activity (could be noise)
        logger.debug("Jungle symbol possibly MISSING (no contours, some edges)")
        return True, 0.50

    # Ambiguous: shape found but edges are weak (could be noise artifact)
    logger.debug("Jungle symbol check ambiguous")
    return False, 0.0


def _check_no_symbol_error_as_stamp(img_bgr: np.ndarray, set_id: str,
                                    card_rarity: str) -> dict:
    """Wrapper around _check_no_symbol_error returning stamp-checker dict."""
    is_missing, confidence = _check_no_symbol_error(img_bgr, set_id,
                                                    card_rarity)
    if is_missing:
        return {
            "detected": True,
            "confidence": confidence,
            "position": "bottom_right",
            "evidence": "no_jungle_flower_symbol",
        }
    return {"detected": False, "confidence": 0.0, "position": "bottom_right"}


# ---------------------------------------------------------------------------
# Crosshatch holo pattern detection (league/tournament rewards)
# ---------------------------------------------------------------------------
# Play! Pokemon league promos and tournament reward cards use crosshatch
# (grid-like) holographic pattern instead of standard cosmos/galaxy holo.
# Era 3+ (DP onward, ~2007+).  The crosshatch pattern has perpendicular
# line grid visible in the foil.

_LEAGUE_PROMO_SETS = frozenset({
    "dpp",    # DP-era league promos
    "hsp",    # HGSS-era league promos
    "bwp",    # BW-era league/tournament promos
    "xyp",    # XY-era league/tournament promos
    "smp",    # SM-era league/tournament promos
    "swshp",  # SWSH-era league/tournament promos
    "svp",    # SV-era league/tournament promos
})

_CROSSHATCH_REGIONS = {
    "artwork": (0.06, 0.10, 0.94, 0.58),
    "border_top": (0.04, 0.02, 0.96, 0.10),
    "border_bottom": (0.04, 0.90, 0.96, 0.98),
}


def _check_crosshatch_holo(img_bgr: np.ndarray, set_id: str,
                            era: int) -> dict:
    """Detect crosshatch holo pattern (league/tournament exclusive).

    Crosshatch holo has a visible grid of perpendicular lines baked into
    the foil, distinguishing it from standard cosmos/galaxy holo.  The grid
    is regular (consistent spacing) and has strong energy at 0 and 90 degrees.

    Detection approach:
        1. Extract artwork + border regions.
        2. High-pass filter to isolate the foil texture from the printed image.
        3. Hough line detection on the high-pass result.
        4. Classify lines into horizontal (~0/180 deg) and vertical (~90 deg).
        5. Count perpendicular line pairs.  Crosshatch requires strong
           presence of BOTH orientations with roughly regular spacing.
        6. Verify grid regularity by checking line spacing consistency.

    Feasibility note -- binder sleeves:
        Crosshatch pattern is a physical texture in the foil that catches
        light even through sleeves, BUT detection depends heavily on the
        photo angle and lighting.  Under flat/even lighting the grid may
        be invisible.  Under angled light with glare, the grid lines
        become visible as bright streaks.  Expect moderate recall (~40-60%)
        on binder page photos.  False positive rate should be low because
        regular perpendicular grids rarely occur in printed card artwork.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. "bwp").
        era: Era number (1-9, 0 if unknown).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False, "confidence": 0.0,
        "position": "artwork", "evidence": "",
    }

    is_league_set = set_id in _LEAGUE_PROMO_SETS

    if era != 0 and era < 3:
        return result_base

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return result_base

    ax0, ay0, ax1, ay1 = _CROSSHATCH_REGIONS["artwork"]
    artwork = img_bgr[int(ay0 * h):int(ay1 * h), int(ax0 * w):int(ax1 * w)]

    if artwork.size == 0:
        return result_base

    h_lines, v_lines, h_spacings, v_spacings = _detect_grid_lines(artwork)

    border_h, border_v = 0, 0
    for region_name in ("border_top", "border_bottom"):
        bx0, by0, bx1, by1 = _CROSSHATCH_REGIONS[region_name]
        border_crop = img_bgr[int(by0 * h):int(by1 * h),
                              int(bx0 * w):int(bx1 * w)]
        if border_crop.size > 0:
            bh, bv, _, _ = _detect_grid_lines(border_crop)
            border_h += bh
            border_v += bv

    total_h = h_lines + border_h
    total_v = v_lines + border_v

    min_lines_per_dir = 3
    has_both_dirs = (total_h >= min_lines_per_dir
                     and total_v >= min_lines_per_dir)

    if not has_both_dirs:
        logger.debug("Crosshatch: insufficient lines h=%d v=%d (need %d each)",
                     total_h, total_v, min_lines_per_dir)
        return result_base

    h_regular = (_check_spacing_regularity(h_spacings)
                 if len(h_spacings) >= 2 else False)
    v_regular = (_check_spacing_regularity(v_spacings)
                 if len(v_spacings) >= 2 else False)

    confidence = 0.0
    pair_count = min(total_h, total_v)
    if pair_count >= 8:
        confidence = 0.70
    elif pair_count >= 5:
        confidence = 0.55
    elif pair_count >= 3:
        confidence = 0.40

    if h_regular or v_regular:
        confidence += 0.15
    if h_regular and v_regular:
        confidence += 0.10
    if is_league_set:
        confidence += 0.10

    confidence = max(0.0, min(1.0, confidence))
    detected = confidence >= 0.45

    evidence_parts = [f"h={total_h}", f"v={total_v}"]
    if h_regular:
        evidence_parts.append("h_regular")
    if v_regular:
        evidence_parts.append("v_regular")
    if is_league_set:
        evidence_parts.append("league_set")
    evidence_str = ",".join(evidence_parts)

    if detected:
        logger.info("Crosshatch holo detected: conf=%.2f, h_lines=%d, "
                    "v_lines=%d, h_regular=%s, v_regular=%s, "
                    "league_set=%s, set=%s",
                    confidence, total_h, total_v,
                    h_regular, v_regular, is_league_set, set_id)
        return {
            "detected": True, "confidence": round(confidence, 2),
            "position": "artwork", "evidence": evidence_str,
        }

    logger.debug("Crosshatch holo not detected: conf=%.2f, h=%d, v=%d",
                 confidence, total_h, total_v)
    return result_base


def _detect_grid_lines(
    region_bgr: np.ndarray,
) -> tuple[int, int, list[float], list[float]]:
    """Detect horizontal and vertical lines via Hough transform.

    High-pass filters to separate foil texture from artwork, then runs
    probabilistic Hough line detection and classifies by angle.

    Returns:
        (h_count, v_count, h_positions, v_positions)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    if rh < 10 or rw < 10:
        return 0, 0, [], []

    blur_ksize = max(3, (min(rh, rw) // 6) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    highpass = cv2.subtract(gray, blurred)
    highpass = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(highpass, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    min_line_length = max(10, int(min(rh, rw) * 0.15))
    max_line_gap = max(3, int(min(rh, rw) * 0.05))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=max(20, int(min(rh, rw) * 0.08)),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None:
        return 0, 0, [], []

    h_positions: list[float] = []
    v_positions: list[float] = []
    angle_tolerance = 15

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))

        if angle < angle_tolerance or angle > (180 - angle_tolerance):
            h_positions.append((y1 + y2) / 2.0)
        elif abs(angle - 90) < angle_tolerance:
            v_positions.append((x1 + x2) / 2.0)

    h_positions = _deduplicate_positions(sorted(h_positions), min_gap=3)
    v_positions = _deduplicate_positions(sorted(v_positions), min_gap=3)

    return len(h_positions), len(v_positions), h_positions, v_positions


def _deduplicate_positions(positions: list[float],
                           min_gap: float = 3.0) -> list[float]:
    """Merge positions closer than min_gap pixels."""
    if not positions:
        return []

    groups: list[list[float]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos - groups[-1][-1] < min_gap:
            groups[-1].append(pos)
        else:
            groups.append([pos])

    return [sum(g) / len(g) for g in groups]


def _check_spacing_regularity(positions: list[float],
                               max_cv: float = 0.35) -> bool:
    """Check if line spacings are roughly regular (consistent intervals).

    Computes coefficient of variation (std/mean) of inter-line spacings.
    A regular grid has low CV; random lines have high CV.
    """
    if len(positions) < 3:
        return False

    spacings = np.diff(positions)
    if len(spacings) < 2:
        return False

    mean_spacing = float(np.mean(spacings))
    if mean_spacing < 1:
        return False

    cv = float(np.std(spacings)) / mean_spacing
    return cv < max_cv


# ---------------------------------------------------------------------------
# Determine which stamps to check for a given card_id
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Holo finish detection (artwork region analysis)
# ---------------------------------------------------------------------------

# Artwork region: center of card illustration area.
# Same coordinates as holo_detector.py -- excludes name bar and border.
_HOLO_ART_Y0, _HOLO_ART_Y1 = 0.12, 0.56
_HOLO_ART_X0, _HOLO_ART_X1 = 0.10, 0.90

# Minimum saturation/value to consider a pixel "colorful" for hue analysis.
_HOLO_MIN_SAT = 40
_HOLO_MIN_VAL = 40

# Saturation std thresholds for artwork region.
# Through binder sleeves, holo shimmer is suppressed: sat_std ~33+ for holo,
# ~15-30 for clean non-holo artwork.
_HOLO_SAT_STD_THRESHOLD = 33.0

# Hue spatial noise via Laplacian (non-edge, colorful pixels only).
# Holo surfaces produce rapid color speckle even in "flat" areas.
# Non-holo: ~5-30.  Holo through sleeve: ~50-150+.
_HOLO_HUE_LAP_THRESHOLD = 35.0


def _check_holo_finish(
    img_bgr: np.ndarray,
    set_id: str,
    era: int,
) -> tuple[str, float]:
    """Detect if a card has holographic artwork finish.

    Analyzes the artwork region (x:10-90%, y:12-56%) for holographic
    signal using two complementary features:

    1. **Saturation variance** -- Holographic foil creates rainbow shimmer
       with high saturation spread.  Non-holo artwork has moderate sat_std
       (~15-30); holo artwork pushes sat_std >= 33 even through sleeves.

    2. **Hue spatial noise** -- Laplacian of hue channel in non-edge,
       colorful regions.  Holographic surfaces produce rapid, noisy hue
       changes between adjacent pixels (prismatic micro-reflections).
       Non-holo: ~5-30.  Holo: ~50-150+.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR card image (from cv2.imread or segmenter output).
    set_id : str
        Set identifier (e.g., "ex15", "base1").
    era : int
        Card era number (1-9).

    Returns
    -------
    (finish, confidence) where finish is one of:
        "holofoil"  -- artwork area shows holographic signal
        "non_holo"  -- no holographic signal in artwork
        "unknown"   -- insufficient data or ambiguous

    Notes
    -----
    Through binder sleeves, holo shimmer is significantly suppressed.
    A "non_holo" result at high confidence is more reliable than
    "holofoil", because absence of signal is easier to confirm.

    This function analyzes ARTWORK holo only (standard holo rares).
    For reverse holo detection (body/border holo), use
    ``holo_detector.detect_holo_type_from_array``.
    """
    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        logger.debug("Image too small for holo finish check: %dx%d", w, h)
        return "unknown", 0.0

    # Extract artwork region
    art = img_bgr[
        int(_HOLO_ART_Y0 * h):int(_HOLO_ART_Y1 * h),
        int(_HOLO_ART_X0 * w):int(_HOLO_ART_X1 * w),
    ]
    if art.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    # --- Feature 1: Saturation standard deviation ---
    sat_std = float(np.std(s_chan))
    sat_mean = float(np.mean(s_chan))

    # --- Feature 2: Hue spatial noise (Laplacian in flat colorful areas) ---
    # Mask: only colorful pixels (reliable hue) in non-edge areas
    colorful_mask = (s_chan >= _HOLO_MIN_SAT) & (v_chan >= _HOLO_MIN_VAL)

    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    non_edge = edge_dilated == 0

    combined_mask = colorful_mask & non_edge

    hue_lap = cv2.Laplacian(h_chan, cv2.CV_32F, ksize=3)
    abs_hue_lap = np.abs(hue_lap)

    flat_lap = abs_hue_lap[combined_mask]
    if len(flat_lap) < 30:
        hue_noise = 0.0
    else:
        hue_noise = float(np.mean(flat_lap))

    # --- Feature 3: Hue spread in colorful pixels ---
    # Count distinct hue bins with significant presence.
    # Holo foil disperses light across many hues; printed art concentrates.
    colorful_hues = h_chan[colorful_mask]
    if len(colorful_hues) >= 50:
        hist, _ = np.histogram(colorful_hues, bins=36, range=(0, 180))
        threshold = len(colorful_hues) * 0.01
        hue_spread = float(np.sum(hist > threshold))
    else:
        hue_spread = 0.0

    # --- Classification ---
    # Both signals must agree for a confident holo call.
    sat_elevated = sat_std >= _HOLO_SAT_STD_THRESHOLD
    noise_elevated = hue_noise >= _HOLO_HUE_LAP_THRESHOLD

    logger.debug(
        "Holo finish check (set=%s, era=%d): "
        "sat_std=%.1f (>%.0f? %s), hue_noise=%.1f (>%.0f? %s), "
        "hue_spread=%.0f, sat_mean=%.1f",
        set_id, era,
        sat_std, _HOLO_SAT_STD_THRESHOLD, sat_elevated,
        hue_noise, _HOLO_HUE_LAP_THRESHOLD, noise_elevated,
        hue_spread, sat_mean,
    )

    if sat_elevated and noise_elevated:
        # Both signals confirm holo.
        # Confidence scales with how far above thresholds.
        sat_ratio = sat_std / _HOLO_SAT_STD_THRESHOLD
        noise_ratio = hue_noise / _HOLO_HUE_LAP_THRESHOLD
        conf = min(0.95, 0.55 + 0.15 * min(sat_ratio - 1.0, 1.0)
                   + 0.15 * min(noise_ratio - 1.0, 1.0))
        return "holofoil", round(conf, 2)

    if sat_elevated and not noise_elevated:
        # Sat is high but noise is low -- could be colorful non-holo artwork
        # or holo suppressed by sleeve.  Low confidence holo only if sat
        # is well above threshold AND hue spread is high.
        if sat_std >= _HOLO_SAT_STD_THRESHOLD * 1.3 and hue_spread >= 12:
            return "holofoil", 0.45
        # Borderline -- report unknown.
        return "unknown", 0.35

    if noise_elevated and not sat_elevated:
        # Noise is high but sat is low -- unusual, could be colorful artwork
        # with many small details.  Low confidence.
        if hue_noise >= _HOLO_HUE_LAP_THRESHOLD * 1.5:
            return "holofoil", 0.40
        return "unknown", 0.30

    # Neither signal elevated -- confidently non-holo.
    # Confidence is high because absence of signal is reliable.
    max_metric = max(
        sat_std / _HOLO_SAT_STD_THRESHOLD,
        hue_noise / _HOLO_HUE_LAP_THRESHOLD,
    )
    if max_metric < 0.5:
        conf = 0.90
    elif max_metric < 0.75:
        conf = 0.80
    else:
        conf = 0.65  # Close to threshold, less confident

    return "non_holo", round(conf, 2)



# ---------------------------------------------------------------------------
# Cracked ice holo pattern detection
# ---------------------------------------------------------------------------

# Artwork region for holo pattern analysis (fractional coordinates).
_CRACKED_ICE_ART_REGION = (0.10, 0.12, 0.90, 0.56)  # x0, y0, x1, y1

# Eras where cracked ice holo exists (Platinum onward = era 4+).
# Platinum is HGSS-adjacent (era 4), then BW(5), XY(6), SM(7), SWSH(8), SV(9).
_CRACKED_ICE_MIN_ERA = 4


def _check_cracked_ice_holo(
    img_bgr: np.ndarray,
    set_id: str,
    era: int,
) -> tuple[bool, float]:
    """Detect cracked ice holo pattern (theme deck/product exclusive).

    Cracked ice (aka shattered glass) holo has a distinctive geometric
    fractured pattern on the artwork area -- irregular polygonal regions
    separated by sharp straight-line boundaries at many different angles.
    This contrasts with:
      - Cosmos holo: small circular dots / swirl patterns (no straight lines)
      - Standard holo: smooth, broad rainbow shimmer (few edges)
      - Reverse holo: foil on body, not artwork

    Detection approach:
      1. Extract artwork region (x:10-90%, y:12-56%).
      2. Convert to grayscale, apply CLAHE for contrast normalization.
      3. Canny edge detection to find all edges.
      4. Probabilistic Hough line transform to find straight line segments.
      5. Compute angular diversity -- cracked ice has lines at many different
         angles (high entropy in angle histogram), while normal artwork has
         edges concentrated along a few directions (card borders, character
         outlines).
      6. Combine line density and angular diversity into a confidence score.

    Limitations:
      - BINDER SLEEVE EFFECTS: Sleeves add glare and reduce contrast of the
        fractured pattern.  Through-sleeve detection is unreliable.
      - LIGHTING DEPENDENCY: The cracked ice pattern is most visible when
        light catches the facets at an angle.  Even lighting suppresses it.
      - SINGLE-PHOTO: Some angles may not reveal the pattern at all.
      - COLORFUL/BUSY ARTWORK: Cards with many drawn edges (e.g., battle
        scenes) may produce false positives on line density.  Angular
        diversity helps distinguish but does not eliminate this risk.
      - RESOLUTION: At binder-page resolution (~300px per card), the fine
        fracture lines may be below the Nyquist limit.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier string.
        era: Era number (1-9, 0 if unknown).

    Returns:
        (is_cracked_ice, confidence) where confidence is in [0.0, 1.0].
        Returns (False, 0.0) for cards from pre-Platinum eras.
    """
    # Era gate: cracked ice does not exist before Platinum (era 4).
    if era != 0 and era < _CRACKED_ICE_MIN_ERA:
        return False, 0.0

    h, w = img_bgr.shape[:2]
    if h < 80 or w < 60:
        logger.debug("Image too small for cracked ice detection: %dx%d", w, h)
        return False, 0.0

    # --- Step 1: Extract artwork region ---
    x0, y0, x1, y1 = _CRACKED_ICE_ART_REGION
    art = img_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if art.size == 0:
        return False, 0.0

    art_h, art_w = art.shape[:2]
    if art_h < 40 or art_w < 40:
        return False, 0.0

    # --- Step 2: Grayscale + CLAHE contrast normalization ---
    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Light blur to suppress JPEG noise but preserve fracture lines.
    gray_eq = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    # --- Step 3: Canny edge detection ---
    # Use moderate thresholds -- fracture lines are medium-contrast edges.
    edges = cv2.Canny(gray_eq, 50, 150)

    # Edge density: fraction of pixels that are edges.
    edge_density = float(np.count_nonzero(edges)) / (art_h * art_w)

    # --- Step 4: Probabilistic Hough line transform ---
    # minLineLength: fracture segments are short (10-50px at card resolution).
    # maxLineGap: allow small gaps in detected lines.
    min_line_len = max(8, int(art_w * 0.04))
    max_line_gap = max(3, int(art_w * 0.02))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    if lines is None or len(lines) == 0:
        logger.debug("Cracked ice: no Hough lines detected")
        return False, 0.0

    num_lines = len(lines)
    # Normalize line count by region area (lines per 1000 px^2).
    area_k = (art_h * art_w) / 1000.0
    line_density = num_lines / area_k

    # --- Step 5: Angular diversity ---
    # Compute angle of each line segment, quantize into 18 bins (10 deg each).
    angles = []
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l)) % 180
        angles.append(angle)

    angles = np.array(angles)
    num_bins = 18
    hist, _ = np.histogram(angles, bins=num_bins, range=(0, 180))

    # Occupied bins: how many angle bins have at least some lines.
    min_count = max(1, num_lines * 0.02)  # 2% threshold per bin
    occupied_bins = int(np.sum(hist >= min_count))

    # Entropy of angle distribution (higher = more uniform = more cracked-ice-like).
    hist_norm = hist.astype(np.float64)
    hist_norm = hist_norm / hist_norm.sum()
    hist_norm = hist_norm[hist_norm > 0]  # drop zeros for log
    angle_entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
    # Max entropy for 18 bins is log2(18) = ~4.17.
    max_entropy = np.log2(num_bins)
    entropy_ratio = angle_entropy / max_entropy  # 0..1

    logger.debug(
        "Cracked ice metrics: lines=%d, line_density=%.2f/kpx, "
        "edge_density=%.3f, occupied_bins=%d/%d, entropy_ratio=%.2f",
        num_lines, line_density, edge_density, occupied_bins, num_bins,
        entropy_ratio,
    )

    # --- Step 6: Classification ---
    # Cracked ice signature: high line density + high angular diversity.
    # Normal artwork: may have high line density (busy art) but angles
    #   cluster around a few directions (horizontal/vertical character edges).
    # Cosmos holo: low line density (dot pattern, not straight lines).
    # Standard holo: low line density (smooth shimmer).

    # Thresholds calibrated conservatively -- prefer false negatives over
    # false positives.  These are educated guesses since we lack ground truth
    # cracked ice images through binder sleeves.

    # Minimum requirements for a positive detection:
    MIN_LINE_DENSITY = 0.8       # lines per 1000 px^2
    MIN_OCCUPIED_BINS = 10       # out of 18 angle bins
    MIN_ENTROPY_RATIO = 0.70     # angle uniformity (0..1)
    MIN_EDGE_DENSITY = 0.03      # fraction of edge pixels

    # Strong detection thresholds:
    STRONG_LINE_DENSITY = 1.5
    STRONG_OCCUPIED_BINS = 14
    STRONG_ENTROPY_RATIO = 0.85

    if line_density < MIN_LINE_DENSITY:
        logger.debug("Cracked ice: line density %.2f below minimum %.2f",
                      line_density, MIN_LINE_DENSITY)
        return False, 0.0

    if occupied_bins < MIN_OCCUPIED_BINS:
        logger.debug("Cracked ice: occupied bins %d below minimum %d",
                      occupied_bins, MIN_OCCUPIED_BINS)
        return False, 0.0

    if entropy_ratio < MIN_ENTROPY_RATIO:
        logger.debug("Cracked ice: entropy ratio %.2f below minimum %.2f",
                      entropy_ratio, MIN_ENTROPY_RATIO)
        return False, 0.0

    if edge_density < MIN_EDGE_DENSITY:
        logger.debug("Cracked ice: edge density %.3f below minimum %.3f",
                      edge_density, MIN_EDGE_DENSITY)
        return False, 0.0

    # Passed all minimums -- compute confidence.
    # Score each dimension 0..1 based on where it falls between min and strong.
    def _score(val: float, lo: float, hi: float) -> float:
        return min(1.0, max(0.0, (val - lo) / (hi - lo)))

    s_density = _score(line_density, MIN_LINE_DENSITY, STRONG_LINE_DENSITY)
    s_bins = _score(occupied_bins, MIN_OCCUPIED_BINS, STRONG_OCCUPIED_BINS)
    s_entropy = _score(entropy_ratio, MIN_ENTROPY_RATIO, STRONG_ENTROPY_RATIO)

    # Weighted average -- angular diversity matters most.
    raw_conf = 0.30 * s_density + 0.35 * s_bins + 0.35 * s_entropy

    # Map to output confidence range [0.40, 0.85].
    # Capped at 0.85 because single-photo detection through binder sleeves
    # cannot be fully reliable for any holo pattern.
    confidence = 0.40 + raw_conf * 0.45
    confidence = round(min(0.85, confidence), 2)

    logger.info(
        "Cracked ice detected: conf=%.2f (density=%.2f[%.2f], "
        "bins=%d[%.2f], entropy=%.2f[%.2f])",
        confidence, line_density, s_density, occupied_bins, s_bins,
        entropy_ratio, s_entropy,
    )

    return True, confidence



# ---------------------------------------------------------------------------
# McDonald's confetti holo detection
# ---------------------------------------------------------------------------

# All known McDonald's promo set prefixes
_MCDONALDS_SETS = frozenset({
    "mcd11", "mcd12", "mcd14", "mcd15", "mcd16",
    "mcd17", "mcd18", "mcd19", "mcd21", "mcd22",
})

# Confetti holo analysis region: artwork area where the confetti pattern is
# most visible.  Excludes name bar and text box.
_CONFETTI_ART_REGION = (0.08, 0.12, 0.92, 0.56)  # (x0, y0, x1, y1)


def _check_mcdonalds_holo(
    img_bgr: np.ndarray,
    set_id: str,
    era: int,
) -> dict:
    """Detect McDonald's confetti holo pattern.

    McDonald's promo cards have a distinctive pixelated/confetti holographic
    pattern with thick, sparse glitter dots -- visually chunkier than cosmos
    holo (fine uniform shimmer) or standard holo (smooth rainbow gradient).

    Detection strategy:
      1. **Metadata shortcut**: If set_id starts with 'mcd', return high
         confidence immediately.
      2. **Visual detection**: Analyze the artwork region for the confetti
         signature -- sparse, intense bright spots with high local brightness
         variance.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g., 'mcd21', 'swsh1').
        era: Era number (unused, reserved for future gating).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence',
        'is_mcdonalds_set'.
    """
    result_base = {
        "detected": False,
        "confidence": 0.0,
        "position": "artwork",
        "is_mcdonalds_set": False,
    }

    # --- Path 1: Metadata-based detection (authoritative) ---
    is_mcd_set = set_id.startswith("mcd")
    if is_mcd_set:
        return {
            "detected": True,
            "confidence": 0.95,
            "position": "artwork",
            "evidence": "mcdonalds_set_id",
            "is_mcdonalds_set": True,
        }

    # --- Path 2: Visual detection of confetti pattern ---
    art = _extract_region(img_bgr, *_CONFETTI_ART_REGION)
    if art.size == 0 or art.shape[0] < 20 or art.shape[1] < 20:
        return result_base

    try:
        h, w = art.shape[:2]

        # Convert to HSV for brightness and saturation analysis
        hsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)

        # --- Metric 1: Bright spot density and size ---
        p95 = float(np.percentile(v_chan, 95))
        bright_thresh = max(200.0, p95)
        bright_mask = (v_chan >= bright_thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        total_pixels = h * w
        bright_pixel_count = int(np.sum(bright_mask > 0))
        bright_density = bright_pixel_count / total_pixels

        spot_areas: list[float] = []
        min_spot = max(4, int(total_pixels * 0.0002))
        max_spot = int(total_pixels * 0.05)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_spot <= area <= max_spot:
                spot_areas.append(float(area))

        num_spots = len(spot_areas)
        median_spot_area = float(np.median(spot_areas)) if spot_areas else 0.0

        # --- Metric 2: Local brightness variance ---
        block_size = max(8, min(h, w) // 12)
        local_stds: list[float] = []
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = v_chan[by:by + block_size, bx:bx + block_size]
                local_stds.append(float(np.std(block)))

        high_var_blocks = sum(1 for s in local_stds if s > 40.0)
        high_var_frac = (
            high_var_blocks / len(local_stds) if local_stds else 0.0
        )

        # --- Metric 3: Saturation of bright spots ---
        bright_sat = s_chan[v_chan >= bright_thresh]
        mean_bright_sat = (
            float(np.mean(bright_sat)) if len(bright_sat) > 0 else 0.0
        )

        logger.debug(
            "McD confetti: density=%.4f spots=%d median_area=%.1f "
            "high_var_frac=%.2f bright_sat=%.1f",
            bright_density, num_spots, median_spot_area,
            high_var_frac, mean_bright_sat,
        )

        # --- Scoring ---
        score = 0.0

        # Bright density: confetti is 1-8% of artwork pixels
        if 0.01 <= bright_density <= 0.08:
            score += 0.25
        elif 0.005 <= bright_density <= 0.12:
            score += 0.10

        # Spot count: confetti produces many individual dots
        if num_spots >= 25:
            score += 0.25
        elif num_spots >= 15:
            score += 0.15
        elif num_spots >= 8:
            score += 0.05

        # Median spot size: confetti dots are chunky (larger than cosmos)
        spot_frac = median_spot_area / total_pixels if total_pixels > 0 else 0
        if 0.0005 <= spot_frac <= 0.005:
            score += 0.20
        elif 0.0002 <= spot_frac <= 0.01:
            score += 0.10

        # Local brightness variance
        if high_var_frac >= 0.30:
            score += 0.20
        elif high_var_frac >= 0.15:
            score += 0.10

        # Bright spot saturation: prismatic confetti has colored highlights
        if mean_bright_sat >= 30:
            score += 0.10
        elif mean_bright_sat >= 15:
            score += 0.05

        # Threshold: need >= 0.55 out of 1.00 possible
        detected = score >= 0.55
        confidence = min(0.85, score) if detected else score

        if detected:
            return {
                "detected": True,
                "confidence": round(confidence, 2),
                "position": "artwork",
                "evidence": "visual_confetti_pattern",
                "is_mcdonalds_set": False,
                "metrics": {
                    "bright_density": round(bright_density, 4),
                    "num_spots": num_spots,
                    "median_spot_area": round(median_spot_area, 1),
                    "high_var_frac": round(high_var_frac, 2),
                    "mean_bright_sat": round(mean_bright_sat, 1),
                    "score": round(score, 2),
                },
            }

        return result_base

    except Exception as e:
        logger.debug("McDonald\'s confetti holo check failed: %s", e)
        return result_base


def _get_stamps_to_check(card_id: str, set_id: str, era: int) -> list[str]:
    """Return list of stamp types to check based on era and set.

    The key principle: only check stamps that are POSSIBLE for this card's
    era and set.  No point checking for 1st Edition on a Sword & Shield card.
    """
    checks = []

    # WotC era (1): 1st Edition + Black Star Promo + copyright year
    if set_id in _FIRST_EDITION_SETS:
        checks.append("1st_edition")
    if set_id == "base1":
        checks.append("copyright_year")
        checks.append("shadowless")
    if set_id in _BLACK_STAR_PROMO_SETS:
        checks.append("black_star_promo")

    # EX era (2): set logo stamp on reverse holos + Nintendo promo
    if set_id in _EX_STAMPED_SETS:
        checks.append("ex_set_stamp")
    if set_id == "np":
        checks.append("black_star_promo")

    # DP/HGSS/BW/XY/SM (era 3-7): promo sets
    if set_id in _PROMO_SETS:
        checks.append("promo_stamp")

    # SWSH/SV (era 8-9): modern promo pokeball
    if set_id in _MODERN_PROMO_SETS:
        checks.append("modern_promo")

    # Prerelease stamps: era-gated (text-based for WotC/EX/DP, logo for HGSS+)
    if set_id in _PRERELEASE_TEXT_SETS or set_id in _PRERELEASE_LOGO_SETS:
        checks.append("prerelease")

    # Staff stamp: check proactively alongside prerelease (era 3+ = DP onward)
    # Staff stamps only exist on prerelease event cards, but we check
    # proactively rather than conditionally after prerelease detection to
    # ensure they appear in stamps_checked for transparency.
    if era >= 3 or set_id in _PRERELEASE_TEXT_SETS:
        checks.append("staff_stamp")

    # === P2: Moderate-impact ===

    # EX-era set logo stamp on reverse holos (already added above for ex_set_stamp)

    # === P3: Holo variants (best-effort, lighting-dependent) ===

    # Holo finish detection (artwork area holographic signal)
    if era >= 1 or era == 0:
        checks.append("holo_finish")

    # Reverse holo detection (body area holographic signal, era 2+ only)
    if era >= 2 or era == 0:
        checks.append("reverse_holo")

    # === SV-specific stamps ===

    # Build & Battle stamp (SV era SVP promos)
    if set_id == "svp" and (era >= 8 or era == 0):
        checks.append("build_battle")

    # Pokemon Center exclusive stamp (SVP promos, SWSH/SV era)
    if set_id == "svp":
        checks.append("pokemon_center")

    # === Special error variants ===

    # Jungle no-symbol error: base2 holos only
    # Extract variant/rarity from card_id (e.g. "base2-4/holofoil" -> "holofoil")
    if set_id == "base2":
        variant = card_id.split("/", 1)[1] if "/" in card_id else ""
        if "holo" in variant.lower():
            checks.append("no_symbol_error")

    # DP+ eras (3-9): crosshatch holo (league/tournament promos)
    if era == 0 or era >= 3:
        checks.append("crosshatch_holo")

    # McDonald's sets: confetti holo detection
    if set_id in _MCDONALDS_SETS or set_id.startswith("mcd"):
        checks.append("mcdonalds_holo")

    # Deduplicate while preserving priority order
    seen = set()
    deduped = []
    for c in checks:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped




# ---------------------------------------------------------------------------
# Reverse holo detection
# ---------------------------------------------------------------------------

# Reverse holos exist from Legendary Collection (2002, era 2) onward.
# Earlier sets (WotC era 1) never had reverse holos.
_REVERSE_HOLO_MIN_ERA = 2


def _check_reverse_holo(img_bgr: np.ndarray, set_id: str, era: int
                        ) -> tuple[str, float]:
    """Detect reverse holo (holo on borders, matte on artwork).

    Reverse holo cards have holographic foil on the card body (borders, text
    box, name bar) but NOT on the artwork.  Regular holos are the opposite:
    foil on the artwork only.

    Detection approach:
      1. Measure saturation std in border strips (left/right 8%, top/bottom 10%)
      2. Measure saturation std in artwork center (x:20-80%, y:20-50%)
      3. Measure cross-channel color variance (rainbow shimmer detector)
      4. Compare: reverse holo has high border variance, low art variance.

    Era gating: reverse holos only exist from Legendary Collection (2002, era 2)
    onward.  WotC-era cards (era 1) are always 'normal' or 'holo' (never reverse).

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. 'ex15').
        era: Card era number (1-9, 0 if unknown).

    Returns:
        (label, confidence) where label is one of:
          'reverse_holo' - foil on borders, matte artwork
          'holo'         - foil on artwork, matte borders
          'normal'       - no foil anywhere
          'unknown'      - ambiguous signal (both regions shiny or noisy)
    """
    # Era gating: no reverse holos before Legendary Collection
    if era < _REVERSE_HOLO_MIN_ERA and era != 0:
        return ("normal", 0.90)

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return ("unknown", 0.0)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # --- Border strips (avoiding binder sleeve at extreme edges) ---
    # Left 5-13%, right 87-95%, top 3-13%, bottom 87-97%
    border_left = sat[int(h * 0.15):int(h * 0.85),
                      int(w * 0.05):int(w * 0.13)]
    border_right = sat[int(h * 0.15):int(h * 0.85),
                       int(w * 0.87):int(w * 0.95)]
    border_top = sat[int(h * 0.03):int(h * 0.13),
                     int(w * 0.13):int(w * 0.87)]
    border_bottom = sat[int(h * 0.87):int(h * 0.97),
                        int(w * 0.13):int(w * 0.87)]
    border_pixels = np.concatenate([
        border_left.flatten(), border_right.flatten(),
        border_top.flatten(), border_bottom.flatten(),
    ])

    # --- Artwork center (well inside the art box) ---
    art_center = sat[int(h * 0.20):int(h * 0.50),
                     int(w * 0.20):int(w * 0.80)]

    if border_pixels.size == 0 or art_center.size == 0:
        return ("unknown", 0.0)

    border_sat_std = float(border_pixels.std())
    art_sat_std = float(art_center.flatten().std())

    # --- Cross-channel color variance (rainbow shimmer detector) ---
    # Foil creates rapid color shifts: high per-pixel variance across B,G,R.
    def _channel_variance(region_bgr: np.ndarray) -> float:
        """Mean per-pixel std across B, G, R channels."""
        if region_bgr.size == 0:
            return 0.0
        f = region_bgr.astype(np.float32)
        return float(f.std(axis=2).mean())

    border_left_bgr = img_bgr[int(h * 0.15):int(h * 0.85),
                               int(w * 0.05):int(w * 0.13)]
    border_right_bgr = img_bgr[int(h * 0.15):int(h * 0.85),
                                int(w * 0.87):int(w * 0.95)]
    art_center_bgr = img_bgr[int(h * 0.20):int(h * 0.50),
                              int(w * 0.20):int(w * 0.80)]

    border_cvar = (_channel_variance(border_left_bgr)
                   + _channel_variance(border_right_bgr)) / 2.0
    art_cvar = _channel_variance(art_center_bgr)

    logger.debug(
        "Reverse holo check: border_sat_std=%.1f art_sat_std=%.1f "
        "border_cvar=%.1f art_cvar=%.1f (set=%s era=%d)",
        border_sat_std, art_sat_std, border_cvar, art_cvar, set_id, era,
    )

    # --- Decision logic ---
    # Primary signal: border_sat_std vs art_sat_std (spec thresholds)
    # Secondary signal: border_cvar vs art_cvar (cross-channel shimmer)

    # Reverse holo: shiny borders, matte artwork
    if border_sat_std > 25 and art_sat_std < 20:
        conf = min(0.95, 0.70 + (border_sat_std - 25) / 100)
        return ("reverse_holo", conf)

    # Also catch reverse holos via cross-channel variance when sat_std
    # is ambiguous (artwork has some natural color variation)
    if border_cvar > 30 and art_cvar < 15:
        conf = min(0.90, 0.65 + (border_cvar - 30) / 100)
        return ("reverse_holo", conf)

    # Combined: both metrics lean reverse-holo but neither decisive alone
    if border_sat_std > 25 and border_cvar > 25 and art_cvar < 20:
        conf = min(0.85, 0.60 + (border_sat_std - 25) / 150)
        return ("reverse_holo", conf)

    # Regular holo: matte borders, shiny artwork
    if border_sat_std < 15 and art_sat_std > 25:
        conf = min(0.90, 0.65 + (art_sat_std - 25) / 100)
        return ("holo", conf)

    if border_cvar < 15 and art_cvar > 25:
        conf = min(0.85, 0.60 + (art_cvar - 25) / 100)
        return ("holo", conf)

    # Both low: normal (no foil)
    if border_sat_std < 20 and art_sat_std < 20 and border_cvar < 20:
        return ("normal", 0.80)

    # Both high or ambiguous: unknown
    return ("unknown", 0.40)

# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def detect_stamps(image_path: str, card_id: str) -> dict:
    """Detect physical stamps on a card based on its era.

    After card identification, this function checks for stamps that are
    appropriate for the card's era and set.  Each stamp type is checked
    in the CORRECT region of the card (fixed position for each stamp type).

    Args:
        image_path: Path to the card image.
        card_id: Full card identifier (e.g. "base1-4/holofoil").

    Returns:
        {
            "stamps_detected": ["1st_edition"],  # list of detected stamps
            "stamp_details": {
                "1st_edition": {"confidence": 0.92, "position": "left", ...},
            },
            "stamps_checked": ["1st_edition"],  # what was checked
        }
    """
    set_id = _extract_set_id(card_id)
    era = _get_era(card_id)

    stamps_to_check = _get_stamps_to_check(card_id, set_id, era)

    result = {
        "stamps_detected": [],
        "stamp_details": {},
        "stamps_checked": list(stamps_to_check),
    }

    if not stamps_to_check:
        logger.debug("No stamps to check for %s (era=%d, set=%s)",
                     card_id, era, set_id)
        return result

    # Load image once
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image for stamp detection: %s",
                       image_path)
        return result

    logger.debug("Checking %d stamp types for %s (era=%d, set=%s): %s",
                 len(stamps_to_check), card_id, era, set_id, stamps_to_check)

    # Cache holo detector result to avoid running the expensive analysis
    # twice when both holo_finish and reverse_holo are checked.
    _holo_cache: dict = {}

    def _holo_finish_as_dict(img_bgr):
        """Wrap _check_holo_finish (returns tuple) into a dict for dispatch."""
        if "result" not in _holo_cache:
            finish, conf = _check_holo_finish(img_bgr, set_id, era)
            _holo_cache["result"] = (finish, conf)
        finish, conf = _holo_cache["result"]
        return {
            "detected": finish == "holofoil",
            "confidence": conf,
            "position": "artwork",
            "evidence": "holo_detector",
            "holo_type": finish,
        }

    def _reverse_holo_as_dict(img_bgr):
        """Wrap _check_reverse_holo (returns tuple) into a dict for dispatch."""
        if "rev_result" not in _holo_cache:
            label, conf = _check_reverse_holo(img_bgr, set_id, era)
            _holo_cache["rev_result"] = (label, conf)
        label, conf = _holo_cache["rev_result"]
        return {
            "detected": label == "reverse_holo",
            "confidence": conf,
            "position": "body",
            "evidence": "reverse_holo_detector",
            "holo_type": label,
        }

    # Dispatch to appropriate checker
    _STAMP_CHECKERS = {
        "1st_edition": lambda img_bgr: _check_1st_edition(img_bgr),
        "ex_set_stamp": lambda img_bgr: _check_ex_set_stamp(img_bgr, set_id),
        "black_star_promo": lambda img_bgr: _check_black_star_promo(img_bgr),
        "modern_promo": lambda img_bgr: _check_modern_promo(img_bgr),
        "promo_stamp": lambda img_bgr: _check_promo_stamp(img_bgr),
        "copyright_year": lambda img_bgr: _check_copyright_year(img_bgr, set_id),
        "shadowless": lambda img_bgr: _check_shadowless(img_bgr, set_id),
        "pokemon_center": lambda img_bgr: _check_pokemon_center_stamp(img_bgr, set_id, era),
        "retailer_stamp": lambda img_bgr: _check_retailer_stamp(img_bgr, set_id, era),
        "prerelease": lambda img_bgr: _check_prerelease(img_bgr, set_id, era),
        "staff_stamp": lambda img_bgr: _check_staff_stamp(img_bgr, set_id, era),
        "build_battle": lambda img_bgr: _check_build_battle_stamp(img_bgr, set_id, era),
        "holo_finish": _holo_finish_as_dict,
        "reverse_holo": _reverse_holo_as_dict,
        "no_symbol_error": lambda img_bgr: _check_no_symbol_error_as_stamp(
            img_bgr, set_id,
            card_id.split("/", 1)[1] if "/" in card_id else "",
        ),
        "crosshatch_holo": lambda img_bgr: _check_crosshatch_holo(img_bgr, set_id, era),
        "mcdonalds_holo": lambda img_bgr: _check_mcdonalds_holo(img_bgr, set_id, era),
    }

    for stamp_type in stamps_to_check:
        checker = _STAMP_CHECKERS.get(stamp_type)
        if checker is None:
            logger.warning("Unknown stamp type: %s", stamp_type)
            continue

        try:
            detail = checker(img)
            if detail.get("detected"):
                result["stamps_detected"].append(stamp_type)
                stamp_info = {
                    "confidence": detail["confidence"],
                    "position": detail.get("position", "unknown"),
                    "evidence": detail.get("evidence", ""),
                }
                # Include variant for copyright_year detection
                if "variant" in detail:
                    stamp_info["variant"] = detail["variant"]
                # Include retailer name for retailer_stamp detection
                if "retailer" in detail and detail["retailer"]:
                    stamp_info["retailer"] = detail["retailer"]
                # Include thick/thin sub-variant for 1st_edition stamps
                if "stamp_thickness" in detail:
                    stamp_info["stamp_thickness"] = detail["stamp_thickness"]
                    stamp_info["thickness_confidence"] = detail.get(
                        "thickness_confidence", 0.0,
                    )
                # Include holo_type for holo_finish / reverse_holo checks
                if "holo_type" in detail:
                    stamp_info["holo_type"] = detail["holo_type"]
                result["stamp_details"][stamp_type] = stamp_info
                logger.info("Stamp detected: %s on %s (conf=%.2f, evidence=%s%s)",
                            stamp_type, card_id, detail["confidence"],
                            detail.get("evidence", ""),
                            f", variant={detail['variant']}"
                            if "variant" in detail else "")
            else:
                logger.debug("Stamp not detected: %s on %s", stamp_type, card_id)
        except Exception as e:
            logger.warning("Stamp check %s failed for %s: %s",
                           stamp_type, card_id, e)

    # --- Staff stamp conditional check ---
    # Only check for STAFF stamp if prerelease was detected or card is from
    # a promo set.  Staff stamps only exist on prerelease event cards.
    if ("prerelease" in result["stamps_detected"]
            or set_id in _PROMO_SETS | _MODERN_PROMO_SETS):
        if "staff_stamp" not in [s for s in stamps_to_check]:
            try:
                detail = _check_staff_stamp(img, set_id, era)
                if detail.get("detected"):
                    result["stamps_detected"].append("staff_stamp")
                    result["stamp_details"]["staff_stamp"] = {
                        "confidence": detail["confidence"],
                        "position": detail.get("position", "artwork_upper_right"),
                        "evidence": detail.get("evidence", ""),
                    }
                    logger.info("Staff stamp detected on %s (conf=%.2f, evidence=%s)",
                                card_id, detail["confidence"],
                                detail.get("evidence", ""))
            except Exception as e:
                logger.warning("Staff stamp check failed for %s: %s", card_id, e)

    # --- Grey stamp sub-variant detection ---
    # Only runs when 1st_edition was detected.  Crops the tight stamp region
    # and analyzes ink darkness to distinguish black vs grey stamps.
    if "1st_edition" in result["stamps_detected"]:
        try:
            regions = STAMP_REGIONS["1st_edition"]
            stamp_crop = _extract_region(img, *regions["tight"])
            ink_color, ink_conf = _check_grey_stamp(img, stamp_crop)
            ed_detail = result["stamp_details"]["1st_edition"]
            ed_detail["ink_color"] = ink_color
            ed_detail["ink_color_confidence"] = ink_conf
            if ink_color == "grey":
                result["stamps_detected"].append("grey_stamp")
                result["stamp_details"]["grey_stamp"] = {
                    "confidence": ink_conf,
                    "position": ed_detail.get("position", "left"),
                    "evidence": "ink_darkness_analysis",
                    "parent_stamp": "1st_edition",
                }
            logger.info("Grey stamp check for %s: ink=%s (conf=%.2f)",
                        card_id, ink_color, ink_conf)
        except Exception as e:
            logger.warning("Grey stamp sub-variant check failed for %s: %s",
                           card_id, e)

    return result


# ---------------------------------------------------------------------------
# Cosmos holo vs standard holo detection
# ---------------------------------------------------------------------------

# Artwork region for holo texture analysis (x0, y0, x1, y1 as fractions).
_ARTWORK_REGION = (0.10, 0.12, 0.90, 0.56)


def _radial_fft_profile(gray: np.ndarray) -> np.ndarray:
    """Compute radial average of FFT magnitude spectrum.

    Returns array of mean magnitude at each radius from DC component.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    max_r = min(cx, cy)

    Y, X = np.ogrid[:mag.shape[0], :mag.shape[1]]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

    radial = np.zeros(max_r, dtype=np.float64)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial[ri] = np.mean(mag[mask])

    return radial


def _check_cosmos_holo(
    img_bgr: np.ndarray,
    set_id: str,
    era: int,
) -> tuple[str, float, dict]:
    """Distinguish cosmos holo (product exclusive) from standard holo.

    Cosmos holo has a scattered circular-dot overlay pattern visible across
    the entire card surface.  Standard holo patterns vary by era: galaxy
    stars (WotC), horizontal lines, type symbols, etc.

    Detection uses five complementary texture features on the artwork area:

    1. **FFT high-frequency ratio** -- cosmos dots inject energy at mid/high
       spatial frequencies.  Standard holo tends toward smoother gradients.
    2. **Laplacian variance** -- captures overall texture sharpness.  Cosmos
       dots create consistent micro-contrast.
    3. **Blob count** -- SimpleBlobDetector tuned for small circular features.
       Cosmos should yield many small, roughly circular blobs.
    4. **Gabor anisotropy** -- cosmos dots are isotropic (orientation-
       invariant); standard holo patterns (lines, stars) are directional.
    5. **Local contrast uniformity** -- cosmos creates spatially uniform
       micro-contrast; standard holo has localized bright patches.

    Args:
        img_bgr: Full card image in BGR format.
        set_id: Set identifier (e.g. "base1").
        era: Era number (1-9).

    Returns:
        (label, confidence, details) where label is 'cosmos', 'standard',
        or 'unknown' and details is a dict of computed features.
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = _ARTWORK_REGION
    art = img_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    if art.size == 0 or art.shape[0] < 30 or art.shape[1] < 30:
        return ("unknown", 0.0, {"error": "artwork_region_too_small"})

    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)
    art_h, art_w = gray.shape

    # ---------------------------------------------------------------
    # Feature 1: FFT high-frequency energy ratio
    # ---------------------------------------------------------------
    radial = _radial_fft_profile(gray)
    total_energy = radial.sum()
    if total_energy == 0:
        return ("unknown", 0.0, {"error": "zero_fft_energy"})

    # Split into low / mid / high thirds
    third = len(radial) // 3
    high_energy = radial[2 * third:].sum()
    mid_energy = radial[third:2 * third].sum()

    hf_ratio = high_energy / total_energy
    mf_ratio = mid_energy / total_energy

    # Look for spectral peaks in mid-frequency band (cosmos dot spacing).
    # Normalize by subtracting a smooth baseline.
    if len(radial) > 20:
        baseline = np.convolve(radial, np.ones(15) / 15, mode="same")
        residual = radial - baseline
        mid_residual = residual[third:2 * third]
        peak_strength = float(
            np.max(mid_residual) / (np.mean(radial[:third]) + 1e-6)
        )
    else:
        peak_strength = 0.0

    # ---------------------------------------------------------------
    # Feature 2: Laplacian variance (texture energy)
    # ---------------------------------------------------------------
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))
    lap_mean_abs = float(np.mean(np.abs(lap)))

    # ---------------------------------------------------------------
    # Feature 3: Blob detection (small circular features)
    # ---------------------------------------------------------------
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 400
    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    blob_count = len(keypoints)

    # Normalize blob count by area (blobs per 1000 px^2)
    art_area = art_h * art_w
    blob_density = blob_count / (art_area / 1000.0) if art_area > 0 else 0.0

    # Size uniformity of blobs (cosmos dots are roughly same size)
    if len(keypoints) >= 5:
        sizes = np.array([kp.size for kp in keypoints])
        blob_size_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-6))
    else:
        blob_size_cv = 1.0  # high CV = non-uniform = not cosmos

    # ---------------------------------------------------------------
    # Feature 4: Gabor anisotropy
    # ---------------------------------------------------------------
    # Test multiple orientations.  Cosmos is isotropic (low anisotropy),
    # standard patterns with lines/stars are directional (high anisotropy).
    n_orientations = 8
    gabor_responses = np.zeros(n_orientations)
    for i, theta in enumerate(
        np.linspace(0, np.pi, n_orientations, endpoint=False)
    ):
        kern = cv2.getGaborKernel(
            (21, 21), sigma=3.0, theta=theta, lambd=10.0, gamma=0.5,
        )
        filt = cv2.filter2D(gray, cv2.CV_32F, kern)
        gabor_responses[i] = float(np.mean(np.abs(filt)))

    gabor_mean = float(gabor_responses.mean())
    gabor_aniso = float(gabor_responses.std() / (gabor_mean + 1e-6))

    # ---------------------------------------------------------------
    # Feature 5: Local contrast uniformity
    # ---------------------------------------------------------------
    # Cosmos dots create spatially uniform micro-contrast.
    # Standard holo has localized bright patches.
    block = 16
    local_vars: list[float] = []
    for by in range(0, art_h - block, block):
        for bx in range(0, art_w - block, block):
            patch = gray_f[by:by + block, bx:bx + block]
            local_vars.append(float(np.var(patch)))

    if local_vars:
        local_var_arr = np.array(local_vars)
        local_var_mean = float(local_var_arr.mean())
        # CV of local variances: cosmos = uniform texture = low CV
        local_var_cv = float(
            local_var_arr.std() / (local_var_arr.mean() + 1e-6)
        )
    else:
        local_var_mean = 0.0
        local_var_cv = 1.0

    # ---------------------------------------------------------------
    # Scoring: weighted heuristic
    # ---------------------------------------------------------------
    # Each feature contributes a score toward cosmos (positive) or
    # standard (negative).  Thresholds are empirically estimated --
    # these WILL need calibration against real cosmos/standard samples.

    score = 0.0
    evidence_parts: list[str] = []

    # High blob density favors cosmos
    if blob_density > 0.15:
        score += 0.20
        evidence_parts.append(f"high_blob_density({blob_density:.3f})")
    elif blob_density < 0.05:
        score -= 0.15
        evidence_parts.append(f"low_blob_density({blob_density:.3f})")

    # Uniform blob sizes favor cosmos
    if blob_size_cv < 0.5 and blob_count >= 10:
        score += 0.15
        evidence_parts.append(f"uniform_blobs(cv={blob_size_cv:.2f})")
    elif blob_size_cv > 1.0:
        score -= 0.10
        evidence_parts.append(f"varied_blobs(cv={blob_size_cv:.2f})")

    # Low gabor anisotropy favors cosmos (isotropic texture)
    if gabor_aniso < 0.05:
        score += 0.15
        evidence_parts.append(f"isotropic({gabor_aniso:.4f})")
    elif gabor_aniso > 0.15:
        score -= 0.15
        evidence_parts.append(f"directional({gabor_aniso:.4f})")

    # FFT spectral peak in mid-frequency favors cosmos
    if peak_strength > 0.3:
        score += 0.20
        evidence_parts.append(f"spectral_peak({peak_strength:.3f})")

    # Higher HF ratio favors cosmos (dots = more high freq content)
    if hf_ratio > 0.005:
        score += 0.10
        evidence_parts.append(f"high_hf({hf_ratio:.4f})")
    elif hf_ratio < 0.002:
        score -= 0.10
        evidence_parts.append(f"low_hf({hf_ratio:.4f})")

    # Spatially uniform local contrast favors cosmos
    if local_var_cv < 0.8:
        score += 0.10
        evidence_parts.append(f"uniform_contrast(cv={local_var_cv:.2f})")
    elif local_var_cv > 1.5:
        score -= 0.10
        evidence_parts.append(f"patchy_contrast(cv={local_var_cv:.2f})")

    # ---------------------------------------------------------------
    # Decision
    # ---------------------------------------------------------------
    details = {
        "hf_ratio": round(hf_ratio, 5),
        "mf_ratio": round(mf_ratio, 5),
        "peak_strength": round(peak_strength, 4),
        "lap_var": round(lap_var, 2),
        "lap_mean_abs": round(lap_mean_abs, 3),
        "blob_count": blob_count,
        "blob_density": round(blob_density, 4),
        "blob_size_cv": round(blob_size_cv, 3),
        "gabor_aniso": round(gabor_aniso, 4),
        "gabor_mean": round(gabor_mean, 3),
        "local_var_mean": round(local_var_mean, 2),
        "local_var_cv": round(local_var_cv, 3),
        "raw_score": round(score, 3),
        "evidence": evidence_parts,
    }

    # Conservative thresholds -- err toward "unknown" rather than
    # false positives.  Through binder sleeves, cosmos dots may be
    # too blurred to reliably detect.
    if score >= 0.40:
        label = "cosmos"
        confidence = min(0.85, 0.50 + score)
    elif score <= -0.25:
        label = "standard"
        confidence = min(0.80, 0.50 + abs(score))
    else:
        label = "unknown"
        confidence = 0.30 + abs(score)

    details["label"] = label
    details["confidence"] = round(confidence, 3)

    logger.info(
        "Cosmos holo check for set=%s era=%d: label=%s conf=%.2f "
        "score=%.3f [%s]",
        set_id, era, label, confidence, score, ", ".join(evidence_parts),
    )

    return (label, confidence, details)
