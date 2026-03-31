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
    # Wizards "W" gold foil stamp: artwork area, typically bottom-right.
    # Extremely rare WotC-era promo — only 7 cards ever produced (1999-2001).
    # Gold foil Wizards logo "W" stamped directly on the card artwork.
    "w_stamp": {
        "wide": (0.10, 0.14, 0.90, 0.56),
        "tight": (0.40, 0.25, 0.85, 0.55),
        "position": "artwork",
    },
    # Winner tournament stamp: bottom-right of artwork area.
    # Three eras:
    #   WotC (2002-2003): gold foil lowercase "winner" with star dot on "i"
    #   Pokemon USA (2003-2004): Poke Ball + "WINNER" text
    #   Modern (2025+): Poke Ball + "WINNER" in caps
    "winner_stamp": {
        "wide": (0.50, 0.25, 0.98, 0.58),
        "tight": (0.55, 0.30, 0.95, 0.55),
        "position": "artwork_bottom_right",
    },
    # Peelable Ditto face icon: bottom-left corner of card.
    # Pokemon GO set (2022) has Bidoof/Numel/Spinarak cards where some copies
    # have a tiny purple Ditto face icon.  Peeling the sticker front reveals
    # a Ditto card underneath.  Icon is ~3-4mm, purple/pink smiley.
    "peelable_ditto": {
        "wide": (0.03, 0.83, 0.18, 0.97),
        "tight": (0.05, 0.85, 0.15, 0.95),
        "position": "bottom_left",
    },
    # Pokemon Day event stamp: appears on the card artwork, typically
    # bottom-right area.  Annual event promo with "POKEMON DAY" logo text.
    "pokemon_day": {
        "wide": (0.40, 0.14, 0.95, 0.58),
        "tight": (0.50, 0.25, 0.95, 0.55),
        "position": "artwork_bottom_right",
    },
}

# ---------------------------------------------------------------------------
# Special Delivery promo cards — Pokemon Center online exclusives
# ---------------------------------------------------------------------------
# These are specific card_ids, so detection is a simple set membership check.
# No visual detection needed — the card_id uniquely identifies them.
_SPECIAL_DELIVERY_CARD_IDS = frozenset({
    "swshp-SWSH074",   # Special Delivery Pikachu
    "swshp-SWSH075",   # Special Delivery Charizard
    "swshp-SWSH177",   # Special Delivery Bidoof
})

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

# ---------------------------------------------------------------------------
# Winner stamp eligible sets
# ---------------------------------------------------------------------------
# Winner stamps appear on tournament prize cards across three eras:
#   WotC (2002-2003): Jungle, Fossil, Base Set 2, Team Rocket, Gym Heroes/Challenge
#   Pokemon USA (2003-2004): EX-era sets (Ruby & Sapphire through Hidden Legends)
#   Modern (2025+): SV-era sets
# These overlap heavily with prerelease-eligible sets since both are event promos.
_WINNER_STAMP_SETS = (
    _PRERELEASE_TEXT_SETS | _PRERELEASE_LOGO_SETS | _FIRST_EDITION_SETS
)


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


def _has_promo_star_shape(region_bgr: np.ndarray,
                          upscale: int = 3,
                          min_area: int = 60,
                          max_area: int = 600,
                          max_solidity: float = 0.45) -> tuple[bool, float, dict]:
    """Check if a region contains a dark star shape (promo stamp indicator).

    Promo stars have low solidity (~0.25-0.40) due to concavities between
    star points.  Normal set symbols and text are more solid (>0.55).

    Uses multi-threshold binarization to handle varying card backgrounds.

    Args:
        region_bgr: BGR image of the region to check.
        upscale: Upscale factor for better contour detection on small crops.
        min_area: Minimum contour area (at upscaled resolution).
        max_area: Maximum contour area (at upscaled resolution).
        max_solidity: Maximum solidity to qualify as a star shape.

    Returns:
        (found, confidence, details) where:
          - found: True if a promo-star-shaped dark blob was detected.
          - confidence: 0.0-0.90 based on how star-like the shape is.
          - details: dict with 'solidity', 'circularity', 'area' of best match.
    """
    if region_bgr.size == 0:
        return False, 0.0, {}

    try:
        region_up = cv2.resize(region_bgr, None, fx=upscale, fy=upscale,
                               interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(region_up, cv2.COLOR_BGR2GRAY)

        best_match: tuple[float, float, float, float] | None = None

        for dark_thresh in [80, 100, 120]:
            _, dark_mask = cv2.threshold(gray, dark_thresh, 255,
                                         cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Tighter solidity at higher thresholds to reduce false positives
            thresh_max_sol = max_solidity if dark_thresh <= 80 else min(max_solidity, 0.42)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue

                solidity = area / hull_area
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if solidity >= thresh_max_sol:
                    continue  # Too solid — not a star shape

                # Confidence based on solidity range
                if solidity < 0.30:
                    conf = 0.90
                elif solidity < 0.38:
                    conf = 0.80
                else:
                    conf = 0.65

                if best_match is None or conf > best_match[3]:
                    best_match = (solidity, circularity, area, conf)

            # Good match at low threshold → skip higher thresholds
            if best_match is not None and best_match[3] >= 0.80:
                break

        if best_match is not None:
            sol, circ, area, conf = best_match
            logger.debug("Promo star shape: solidity=%.3f, circularity=%.3f, "
                         "area=%.0f, confidence=%.2f", sol, circ, area, conf)
            return True, conf, {
                "solidity": round(sol, 3),
                "circularity": round(circ, 3),
                "area": round(area, 0),
            }
    except Exception as e:
        logger.debug("Promo star shape check failed: %s", e)

    return False, 0.0, {}


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


# ---------------------------------------------------------------------------
# DINOv2 differential 1st Edition detection
# ---------------------------------------------------------------------------
# Threshold for DINOv2 differential 1st Edition stamp detection.
# The score is (control_sim - stamp_sim): how much MORE the stamp region
# differs from reference compared to the control region.
# 1st Edition cards have high differential (stamp changes one region
# but not the other).  Unlimited cards have near-zero differential.
#
# Calibrated on synthesized 1st Edition stamps across 50 WotC cards
# (base1-neo4, 5 per set):
#   1st Edition range:  0.1609 - 0.5704 (mean 0.3330)
#   Unlimited range:    0.0047 - 0.0954 (mean 0.0213)
#   Gap: 0.0655 (clean separation between 0.0954 and 0.1609)
#   Threshold at 0.10 gives 100% accuracy (midpoint ~0.128).
_1ST_ED_DINO_DIFF_THRESHOLD = 0.10

# 1st Edition stamp region: left side, between artwork and text box
_1ST_ED_STAMP_REGION = (0.03, 0.53, 0.15, 0.67)
# Control region: artwork center (never has a stamp on any edition)
_1ST_ED_CONTROL_REGION = (0.15, 0.15, 0.85, 0.45)


def _check_1st_edition_dino(img_bgr: np.ndarray, card_id: str,
                             set_id: str) -> dict | None:
    """Detect 1st Edition stamp via DINOv2 differential region comparison.

    Compares TWO regions between the scan and its unlimited reference image:
      1. Stamp region (left side, below artwork -- where the stamp appears)
      2. Control region (artwork center -- stamp-free on all editions)

    The differential (control_sim - stamp_sim) isolates the stamp effect
    from overall scan-vs-reference domain gap (lighting, angle, quality).
    1st Edition cards show a large differential; unlimited cards show
    near-zero.

    This complements the OCR-based _check_1st_edition: it works even when
    the stamp text is too small or blurry for OCR, and provides a
    confidence score based on the magnitude of visual difference.

    Parameters
    ----------
    img_bgr : np.ndarray
        Scanned card image in BGR format.
    card_id : str
        Full card identifier (e.g. "base1-4/normal").
    set_id : str
        Set identifier (e.g. "base1").

    Returns
    -------
    dict or None
        Detection result dict if the check ran successfully, or None
        if it could not run (no reference image, model failure, etc.)
        so that the caller can fall back to OCR.
    """
    from cardprice.ml.ref_matcher import get_reference_image_path

    # Find reference image (always the unlimited/normal variant)
    ref_path = get_reference_image_path(card_id)
    if ref_path is None:
        logger.debug("1st ed DINO: no reference image for %s", card_id)
        return None  # fall back to OCR

    try:
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            logger.debug("1st ed DINO: could not read ref image %s", ref_path)
            return None

        # Crop stamp region from both scan and reference
        scan_stamp = _extract_region(img_bgr, *_1ST_ED_STAMP_REGION)
        ref_stamp = _extract_region(ref_img, *_1ST_ED_STAMP_REGION)
        if scan_stamp.size == 0 or ref_stamp.size == 0:
            return None

        # Crop control region from both
        scan_ctrl = _extract_region(img_bgr, *_1ST_ED_CONTROL_REGION)
        ref_ctrl = _extract_region(ref_img, *_1ST_ED_CONTROL_REGION)
        if scan_ctrl.size == 0 or ref_ctrl.size == 0:
            return None

        # Compute similarities for both regions in one batch
        stamp_sim, ctrl_sim = _dino_crop_similarity_batch(
            [scan_stamp, scan_ctrl],
            [ref_stamp, ref_ctrl],
        )

        # Differential: how much stamp region differs more than control
        diff = ctrl_sim - stamp_sim

        logger.debug(
            "1st ed DINO: %s stamp_sim=%.4f ctrl_sim=%.4f "
            "diff=%.4f (threshold=%.2f)",
            card_id, stamp_sim, ctrl_sim, diff,
            _1ST_ED_DINO_DIFF_THRESHOLD,
        )

        is_1st_ed = diff > _1ST_ED_DINO_DIFF_THRESHOLD

        if is_1st_ed:
            margin = diff - _1ST_ED_DINO_DIFF_THRESHOLD
            # Scale confidence: threshold+0.00 => 0.70, threshold+0.10 => 1.0
            conf = min(0.70 + margin * 3.0, 0.99)
            return {
                "detected": True,
                "confidence": conf,
                "position": "left",
                "evidence": "dino_differential_comparison",
                "stamp_similarity": stamp_sim,
                "control_similarity": ctrl_sim,
                "differential": diff,
                "threshold": _1ST_ED_DINO_DIFF_THRESHOLD,
            }
        else:
            return {
                "detected": False,
                "confidence": 0.0,
                "position": "left",
                "evidence": "dino_differential_comparison",
                "stamp_similarity": stamp_sim,
                "control_similarity": ctrl_sim,
                "differential": diff,
                "threshold": _1ST_ED_DINO_DIFF_THRESHOLD,
            }

    except Exception as e:
        logger.warning("1st ed DINO failed for %s: %s", card_id, e)
        return None  # fall back to OCR


def _check_ghost_stamp(img_bgr: np.ndarray, set_id: str) -> dict:
    """Detect partially printed 1st Edition 'ghost' stamp.

    A ghost stamp is a manufacturing error where only part of the 1st Edition
    stamp was printed -- common on Base Set Pikachu from Zap! theme decks.
    The stamp appears as a faint or partial dark mark in the same region as
    the normal 1st Edition stamp.

    Detection logic:
      1. Extract the 1st Edition stamp region.
      2. Count dark pixels at two thresholds: strict (<=80, normal stamp)
         and relaxed (<=120, faint ink).
      3. If the strict count is too low for a real 1st Edition stamp but
         the relaxed count shows some dark signal, flag as ghost.
      4. Require the dark pixels to be spatially clustered (not just noise)
         by checking that the largest dark contour is >=30% of total dark area.

    Only applicable to base1 cards.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (only 'base1' is valid).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    if set_id != "base1":
        return {"detected": False, "confidence": 0.0, "position": "left"}

    regions = STAMP_REGIONS["1st_edition"]
    tight = _extract_region(img_bgr, *regions["tight"])
    if tight.size == 0:
        return {"detected": False, "confidence": 0.0, "position": "left"}

    gray = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_pixels = h * w
    if total_pixels == 0:
        return {"detected": False, "confidence": 0.0, "position": "left"}

    # Strict threshold: what a normal 1st edition stamp would show
    _, strict_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    strict_dark = np.count_nonzero(strict_mask)
    strict_frac = strict_dark / total_pixels

    # If enough dark pixels for a normal stamp, this isn't a ghost --
    # let _check_1st_edition handle it
    if strict_frac > 0.05:
        return {"detected": False, "confidence": 0.0, "position": "left",
                "evidence": "too_dark_for_ghost"}

    # Relaxed threshold: captures faint/partial ink
    _, relaxed_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    relaxed_dark = np.count_nonzero(relaxed_mask)
    relaxed_frac = relaxed_dark / total_pixels

    # Need some faint signal but not too much (noise)
    if relaxed_frac < 0.008 or relaxed_frac > 0.25:
        return {"detected": False, "confidence": 0.0, "position": "left",
                "evidence": "no_faint_signal"}

    # Check spatial clustering: the dark pixels should form a coherent mark,
    # not random noise scattered across the region
    contours, _ = cv2.findContours(relaxed_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"detected": False, "confidence": 0.0, "position": "left"}

    largest_area = max(cv2.contourArea(c) for c in contours)
    cluster_ratio = largest_area / relaxed_dark if relaxed_dark > 0 else 0.0

    # Largest contour should be a meaningful fraction of the dark pixels --
    # random noise would be many tiny scattered blobs
    if cluster_ratio < 0.30:
        return {"detected": False, "confidence": 0.0, "position": "left",
                "evidence": "scattered_noise"}

    # Confidence scales with how much faint signal is present and how
    # well-clustered it is
    signal_score = min(relaxed_frac / 0.04, 1.0)  # saturates at 4%
    cluster_score = min(cluster_ratio / 0.60, 1.0)  # saturates at 60%
    confidence = round(0.50 + 0.40 * signal_score * cluster_score, 2)
    confidence = min(confidence, 0.90)  # cap -- ghost stamps are inherently
    #                                     uncertain without manual verification

    logger.info(
        "Ghost stamp candidate: relaxed_frac=%.4f, strict_frac=%.4f, "
        "cluster_ratio=%.2f, confidence=%.2f",
        relaxed_frac, strict_frac, cluster_ratio, confidence,
    )

    return {
        "detected": True,
        "confidence": confidence,
        "position": "left",
        "evidence": "faint_partial_ink",
        "relaxed_dark_frac": round(relaxed_frac, 4),
        "strict_dark_frac": round(strict_frac, 4),
        "cluster_ratio": round(cluster_ratio, 2),
    }


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


def _check_ex_set_stamp(img_bgr: np.ndarray, set_id: str,
                        card_id: str | None = None) -> dict:
    """Check for EX-era set logo stamp via DINOv2 region comparison.

    Crops the stamp region (bottom-right of artwork) from both the scan
    and the clean reference image.  Computes DINOv2 CLS embedding
    similarity.  Stamped cards show lower similarity because the foil
    stamp overlay changes the region visually.

    Falls back to the old OCR-based approach if no reference image is
    available or if DINOv2 extraction fails.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {"detected": False, "confidence": 0.0,
                   "position": "artwork_bottom_right"}

    # --- DINOv2-based detection (primary) ---
    if card_id:
        dino_result = _check_ex_stamp_dino(img_bgr, card_id, set_id)
        if dino_result is not None:
            return dino_result

    # --- OCR fallback (legacy) ---
    regions = STAMP_REGIONS["ex_set_stamp"]

    stamp_region = _extract_region(img_bgr, *regions["wide"])
    if stamp_region.size == 0:
        return result_base

    ocr_text = _ocr_region(stamp_region)
    if not ocr_text:
        return result_base

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

    return result_base


# Threshold for DINOv2 differential stamp detection.
# The score is (control_sim - stamp_sim): how much MORE the stamp region
# differs from reference compared to the control region.
# Stamped cards have high differential (stamp overlay changes one region
# but not the other).  Normal cards have near-zero differential.
#
# Calibrated on ex15 Dragon Frontiers (9 cards, 4 stamped, 5 normal):
#   Stamped range:  0.1178 - 0.3584 (mean 0.1979)
#   Normal range:  -0.0237 - 0.0699 (mean 0.0362)
#   Gap: 0.0479 (clean separation between 0.0699 and 0.1178)
#   Threshold at 0.09 = midpoint of gap.
_EX_STAMP_DINO_DIFF_THRESHOLD = 0.09


def _check_ex_stamp_dino(img_bgr: np.ndarray, card_id: str,
                         set_id: str) -> dict | None:
    """Detect EX-era set stamp via DINOv2 differential region comparison.

    Compares TWO regions between the scan and reference:
      1. Stamp region (bottom-right artwork, where the stamp appears)
      2. Control region (top-left artwork, stamp-free on all cards)

    The differential (control_sim - stamp_sim) isolates the stamp effect
    from overall scan-vs-reference domain gap (lighting, angle, quality).
    Stamped cards show a large differential; normal cards show near-zero.

    Parameters
    ----------
    img_bgr : np.ndarray
        Scanned card image in BGR format.
    card_id : str
        Full card identifier (e.g. "ex15-26/normal").
    set_id : str
        Set identifier (e.g. "ex15").

    Returns
    -------
    dict or None
        Detection result dict if the check ran successfully, or None
        if it could not run (no reference image, model failure, etc.)
        so that the caller can fall back to OCR.
    """
    from cardprice.ml.ref_matcher import get_reference_image_path

    # Find reference image
    ref_path = get_reference_image_path(card_id)
    if ref_path is None:
        logger.debug("EX stamp DINO: no reference image for %s", card_id)
        return None  # fall back to OCR

    try:
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            logger.debug("EX stamp DINO: could not read ref image %s",
                         ref_path)
            return None

        # Stamp region: bottom-right of artwork area
        stamp_region = (0.55, 0.35, 0.90, 0.55)
        # Control region: top-center of artwork (never has a stamp)
        control_region = (0.10, 0.10, 0.55, 0.35)

        # Crop stamp region from both
        scan_stamp = _extract_region(img_bgr, *stamp_region)
        ref_stamp = _extract_region(ref_img, *stamp_region)
        if scan_stamp.size == 0 or ref_stamp.size == 0:
            return None

        # Crop control region from both
        scan_ctrl = _extract_region(img_bgr, *control_region)
        ref_ctrl = _extract_region(ref_img, *control_region)
        if scan_ctrl.size == 0 or ref_ctrl.size == 0:
            return None

        # Compute similarities for both regions in one batch
        stamp_sim, ctrl_sim = _dino_crop_similarity_batch(
            [scan_stamp, scan_ctrl],
            [ref_stamp, ref_ctrl],
        )

        # Differential: how much stamp region differs more than control
        diff = ctrl_sim - stamp_sim

        logger.debug(
            "EX stamp DINO: %s stamp_sim=%.4f ctrl_sim=%.4f "
            "diff=%.4f (threshold=%.2f)",
            card_id, stamp_sim, ctrl_sim, diff,
            _EX_STAMP_DINO_DIFF_THRESHOLD,
        )

        is_stamped = diff > _EX_STAMP_DINO_DIFF_THRESHOLD

        if is_stamped:
            margin = diff - _EX_STAMP_DINO_DIFF_THRESHOLD
            conf = min(0.70 + margin * 3.0, 0.99)
            return {
                "detected": True,
                "confidence": conf,
                "position": "artwork_bottom_right",
                "evidence": "dino_differential_comparison",
                "stamp_similarity": stamp_sim,
                "control_similarity": ctrl_sim,
                "differential": diff,
                "threshold": _EX_STAMP_DINO_DIFF_THRESHOLD,
            }
        else:
            return {
                "detected": False,
                "confidence": 0.0,
                "position": "artwork_bottom_right",
                "evidence": "dino_differential_comparison",
                "stamp_similarity": stamp_sim,
                "control_similarity": ctrl_sim,
                "differential": diff,
                "threshold": _EX_STAMP_DINO_DIFF_THRESHOLD,
            }

    except Exception as e:
        logger.warning("EX stamp DINO failed for %s: %s", card_id, e)
        return None  # fall back to OCR


def _dino_crop_similarity(crop1_bgr: np.ndarray,
                          crop2_bgr: np.ndarray) -> float:
    """Compute DINOv2 CLS embedding similarity between two BGR crops.

    Reuses the globally cached DINOv2 model from dino_matcher.
    Both crops are resized to 224x224, normalized, and passed through
    the model.  Returns the dot product of L2-normalized CLS tokens.

    Parameters
    ----------
    crop1_bgr, crop2_bgr : np.ndarray
        BGR image crops (any size, will be resized to 224x224).

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  Higher means more similar.
    """
    sims = _dino_crop_similarity_batch([crop1_bgr], [crop2_bgr])
    return sims[0]


def _dino_crop_similarity_batch(
    crops_a: list[np.ndarray],
    crops_b: list[np.ndarray],
) -> list[float]:
    """Compute DINOv2 CLS similarity for multiple crop pairs in one pass.

    Parameters
    ----------
    crops_a, crops_b : list[np.ndarray]
        Lists of BGR image crops.  crops_a[i] is compared with crops_b[i].
        Must be the same length.

    Returns
    -------
    list[float]
        Cosine similarities, one per pair.
    """
    import torch
    from PIL import Image
    from cardprice.ml.dino_matcher import _load_model, _get_transform

    assert len(crops_a) == len(crops_b), "Crop lists must be same length"
    n = len(crops_a)
    if n == 0:
        return []

    model, device = _load_model()
    transform = _get_transform()

    # Convert all crops to tensors: interleave a[0], b[0], a[1], b[1], ...
    tensors = []
    for i in range(n):
        pil_a = Image.fromarray(cv2.cvtColor(crops_a[i], cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(crops_b[i], cv2.COLOR_BGR2RGB))
        tensors.append(transform(pil_a))
        tensors.append(transform(pil_b))

    batch = torch.stack(tensors).to(device)  # (2*n, 3, 224, 224)

    with torch.no_grad():
        embeddings = model(batch)  # (2*n, 768) CLS tokens

    embs = embeddings.cpu().numpy().astype(np.float32)

    # L2-normalize all
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    # Compute pairwise similarities
    similarities = []
    for i in range(n):
        sim = float(np.dot(embs[2 * i], embs[2 * i + 1]))
        similarities.append(sim)

    return similarities


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

    # Look for a dark star shape using solidity-based detection.
    # Stars have low solidity (~0.25-0.40) due to concavities between points.
    found, conf, details = _has_promo_star_shape(wide)
    if found:
        return {
            "detected": True, "confidence": conf,
            "position": "bottom_right",
            "evidence": "star_shape",
            **details,
        }

    # Also try the WotC-specific region near the set symbol area (right side
    # of artwork, where WotC black star promos display a prominent star).
    wotc_star_region = _extract_region(img_bgr, 0.76, 0.44, 0.98, 0.60)
    if wotc_star_region.size > 0:
        found2, conf2, details2 = _has_promo_star_shape(wotc_star_region)
        if found2:
            return {
                "detected": True, "confidence": conf2,
                "position": "artwork_right",
                "evidence": "wotc_star_shape",
                **details2,
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



def _check_world_championship(img_bgr: np.ndarray, set_id: str) -> dict:
    """Detect World Championship deck cards (reproductions, not real).

    Key indicator: grey/silver border instead of normal yellow/colored.
    Region: outer border strips (x:0-3% and x:97-100%).

    Normal cards have colored borders with moderate-to-high saturation
    (yellow borders sat>40, colored borders even higher).  WC deck cards
    have distinctive grey/silver borders with very low saturation (<25).

    This is a cheap pixel-based check (no OCR), runs in <1ms.

    Returns dict with 'detected', 'confidence', etc.
    """
    h, w = img_bgr.shape[:2]

    # Sample outer 3% border strips (left and right)
    border_w = max(int(w * 0.03), 2)
    left_strip = img_bgr[:, :border_w]
    right_strip = img_bgr[:, w - border_w:]

    # Combine both strips
    border_pixels = np.concatenate([left_strip, right_strip], axis=1)

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(border_pixels, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))

    # Grey/silver: low saturation, moderate-to-high value (not black)
    # Normal borders: higher saturation (yellow ~100+, blue/red ~80+)
    is_grey = mean_sat < 25 and mean_val > 80

    # Confidence based on how definitively grey the border is
    if is_grey:
        # Lower saturation = higher confidence (0 sat = perfect grey)
        sat_score = max(0.0, 1.0 - mean_sat / 25.0)
        # Value should be moderate (silver ~120-200), not too dark or bright
        val_score = 1.0 if 100 < mean_val < 220 else 0.7
        confidence = round(0.6 * sat_score + 0.4 * val_score, 3)
    else:
        confidence = 0.0

    logger.debug(
        "WC deck check: mean_sat=%.1f, mean_val=%.1f, is_grey=%s, conf=%.3f",
        mean_sat, mean_val, is_grey, confidence,
    )

    return {
        "detected": is_grey,
        "confidence": confidence,
        "position": "border",
        "evidence": f"border_sat={mean_sat:.1f},border_val={mean_val:.1f}",
        "warning": "World Championship deck card — reproduction, not tournament legal",
    }


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

    # Sanity check: negative deltas mean the outer strip is darker than
    # the inner strip, which indicates background contamination (binder
    # page, dark sleeve, etc.) bleeding into the crop -- not a shadowless
    # signal.  Reject if either individual delta is significantly negative
    # or if either side shows a clear shadow (large positive delta).
    if right_delta < -5 or bottom_delta < -5:
        logger.debug(
            "Shadowless rejected: negative delta (background contamination) "
            "avg_delta=%.1f (right=%.1f, bottom=%.1f)",
            avg_delta, right_delta, bottom_delta,
        )
        return {
            "detected": False, "confidence": 0.0,
            "position": "right_edge",
            "evidence": "background_contamination",
            **diagnostics,
        }

    # If either edge individually shows a moderate-to-clear shadow (>8),
    # the card is unlimited.  A truly shadowless card has *both* edges
    # uniformly bright -- one shadowed edge is enough to reject.
    if right_delta > 8 or bottom_delta > 8:
        logger.debug(
            "Unlimited (one edge has clear shadow): "
            "avg_delta=%.1f (right=%.1f, bottom=%.1f)",
            avg_delta, right_delta, bottom_delta,
        )
        return {
            "detected": False, "confidence": 0.0,
            "position": "right_edge",
            "evidence": "shadow_present_one_edge",
            **diagnostics,
        }

    # Decision thresholds (both deltas are non-negative and <=15):
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

    # Check for dark star shape in bottom-left (SM/SWSH/SV promos)
    found, conf, details = _has_promo_star_shape(wide)
    if found:
        return {
            "detected": True, "confidence": conf,
            "position": "bottom_left",
            "evidence": "star_shape",
            **details,
        }

    # Check bottom-right for star shape (DP/HGSS/BW/XY promos have the
    # star next to the card number in the bottom-right area).
    br_star_region = _extract_region(img_bgr, 0.70, 0.86, 0.99, 0.98)
    if br_star_region.size > 0:
        found2, conf2, details2 = _has_promo_star_shape(br_star_region)
        if found2:
            return {
                "detected": True, "confidence": conf2,
                "position": "bottom_right",
                "evidence": "star_shape_br",
                **details2,
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


# ---------------------------------------------------------------------------
# Toys R Us stamp: Generations through Ultra Prism (2016-2018).
# Stamped promo cards distributed at Toys R Us store events.
# Highly collectible since Toys R Us closed.
# ---------------------------------------------------------------------------
_TOYS_R_US_ERAS = frozenset({6, 7})  # XY=6, SM=7

# OCR confusion substitutions specific to "Toys R Us" text.
_TRU_OCR_SUBS: dict[str, str] = {
    "ioys": "toys",
    "t0ys": "toys",
    "tays": "toys",
    "iovs": "toys",
    "tqys": "toys",
}


def _check_toys_r_us_stamp(img_bgr: np.ndarray, set_id: str,
                            era: int) -> tuple[bool, float]:
    """Detect 'Toys R Us' text stamp on card.

    Toys R Us distributed stamped promo cards at store events (2016-2018,
    Generations through Ultra Prism).  The stamp is a text overlay reading
    "Toys 'R' Us" on the card artwork.  Position varies by card, so we
    scan the full artwork region.

    False positive rate matters more than recall -- these are rare cards
    and a false positive would misidentify a common card as a collectible
    variant.

    Detection strategy:
      1. Era-gate to XY/SM eras only (eras 6-7).
      2. Crop the artwork region (wide and tight).
      3. Run OCR and apply TRU-specific confusion substitutions.
      4. Require BOTH "toys" AND a secondary keyword ("r us", "rus", " us")
         to match.  Single-keyword matches are rejected to avoid false
         positives from card text containing "toys" or "us" independently.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. "smp", "g1").
        era: Era number (6 for XY, 7 for SM).

    Returns:
        (is_tru, confidence) tuple.  is_tru is True if a Toys R Us stamp
        is detected; confidence is 0.0-1.0.
    """
    # Era gate: TRU stamps only existed in XY/SM eras (2016-2018)
    if era not in _TOYS_R_US_ERAS and era != 0:
        return (False, 0.0)

    regions = STAMP_REGIONS["retailer_stamp"]

    # --- Tight region first (less noise, higher confidence) ---
    tight = _extract_region(img_bgr, *regions["tight"])
    ocr_tight = ""
    if tight.size > 0:
        ocr_tight = _ocr_region(tight)
        for wrong, right in _TRU_OCR_SUBS.items():
            ocr_tight = ocr_tight.replace(wrong, right)

    # --- Wide region (full artwork, more noise) ---
    wide = _extract_region(img_bgr, *regions["wide"])
    ocr_wide = ""
    if wide.size > 0:
        ocr_wide = _ocr_region(wide)
        for wrong, right in _TRU_OCR_SUBS.items():
            ocr_wide = ocr_wide.replace(wrong, right)

    logger.debug("TRU stamp OCR tight=%r wide=%r", ocr_tight, ocr_wide)

    if not ocr_tight and not ocr_wide:
        return (False, 0.0)

    # --- Match logic: require "toys" + secondary keyword ---
    # Two-keyword requirement prevents false positives from card text
    # that might contain "toys" or "us" independently.
    _SECONDARY = ("r us", "rus", " us")

    for source_name, ocr_text in [("tight", ocr_tight), ("wide", ocr_wide)]:
        if not ocr_text:
            continue

        has_toys = "toys" in ocr_text or "toysrus" in ocr_text
        has_secondary = any(kw in ocr_text for kw in _SECONDARY)
        has_combined = "toysrus" in ocr_text

        if has_toys and (has_secondary or has_combined):
            if source_name == "tight":
                conf = 0.92 if (has_combined or "r us" in ocr_text) else 0.85
            else:
                conf = 0.80 if (has_combined or "r us" in ocr_text) else 0.72

            logger.info(
                "Toys R Us stamp detected (conf=%.2f, source=%s, ocr=%r)",
                conf, source_name, ocr_text,
            )
            return (True, conf)

    return (False, 0.0)


def _toys_r_us_as_dict(img_bgr: np.ndarray, set_id: str,
                        era: int) -> dict:
    """Wrap _check_toys_r_us_stamp tuple into a dict for the dispatch table."""
    is_tru, conf = _check_toys_r_us_stamp(img_bgr, set_id, era)
    return {
        "detected": is_tru,
        "confidence": conf,
        "position": "artwork",
        "evidence": "ocr_toys_r_us" if is_tru else "",
    }


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


# ---------------------------------------------------------------------------
# Build-A-Bear Workshop stamp detection
# ---------------------------------------------------------------------------

_BUILD_A_BEAR_KEYWORDS: list[list[str]] = [
    ["build", "bear", "workshop"],  # full text
    ["build", "bear"],              # partial: "BUILD-A-BEAR"
    ["build", "workshop"],          # partial: "BUILD ... WORKSHOP"
    ["bear", "workshop"],           # partial: "BEAR WORKSHOP"
]

_BUILD_A_BEAR_OCR_SUBS: dict[str, str] = {
    "bui1d": "build",
    "bu1ld": "build",
    "bulld": "build",
    "w0rkshop": "workshop",
    "worksho9": "workshop",
    "vorkshop": "workshop",
    "w0rksh0p": "workshop",
}


def _check_build_a_bear_stamp(img_bgr: np.ndarray, set_id: str,
                               era: int) -> tuple[bool, float]:
    """Detect Build-A-Bear Workshop stamp.

    Build-A-Bear cards were distributed with Pokemon stuffed animals.
    Approximately 8 unique stamped cards exist, all from the XY/SM era
    (2014-2019).  The stamp reads "BUILD-A-BEAR WORKSHOP" and is overlaid
    on the card artwork, similar in position to prerelease stamps.

    Detection strategy:
      1. Era-gate to XY/SM (eras 6-7).  Build-A-Bear stamps did not
         exist outside this window.
      2. Crop the artwork region (wide then tight).
      3. Run OCR on each crop with Build-A-Bear-specific confusion subs.
      4. Require at least TWO keywords to match (e.g. "build" + "bear").
         Single-keyword matches are rejected to avoid false positives
         on cards with attack names containing "build" or "bear".
      5. Three-keyword match ("build" + "bear" + "workshop") gets higher
         confidence than two-keyword match.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. "smp", "xyp").
        era: Era number (6-7 for XY/SM).

    Returns:
        (is_bab, confidence) tuple.  is_bab is True when the stamp is
        detected.  confidence is 0.0-1.0.
    """
    # Era gate: Build-A-Bear stamps only existed in XY/SM eras
    if era not in (0, 6, 7):
        return (False, 0.0)

    regions = STAMP_REGIONS["retailer_stamp"]

    # --- Wide region scan ---
    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return (False, 0.0)

    ocr_wide = _ocr_region(wide)
    for wrong, right in _BUILD_A_BEAR_OCR_SUBS.items():
        ocr_wide = ocr_wide.replace(wrong, right)

    # --- Tight region scan ---
    tight = _extract_region(img_bgr, *regions["tight"])
    ocr_tight = ""
    if tight.size > 0:
        ocr_tight = _ocr_region(tight)
        for wrong, right in _BUILD_A_BEAR_OCR_SUBS.items():
            ocr_tight = ocr_tight.replace(wrong, right)

    logger.debug("Build-A-Bear OCR wide=%r tight=%r", ocr_wide, ocr_tight)

    if not ocr_wide.strip() and not ocr_tight.strip():
        return (False, 0.0)

    # --- Keyword matching ---
    best_confidence = 0.0

    for group in _BUILD_A_BEAR_KEYWORDS:
        n_keywords = len(group)

        # Tight region match (less noise, higher confidence)
        if ocr_tight and all(kw in ocr_tight for kw in group):
            conf = 0.95 if n_keywords >= 3 else 0.85
            best_confidence = max(best_confidence, conf)

        # Wide region match (more noise, lower confidence)
        elif all(kw in ocr_wide for kw in group):
            conf = 0.85 if n_keywords >= 3 else 0.75
            best_confidence = max(best_confidence, conf)

    if best_confidence > 0.0:
        logger.info(
            "Build-A-Bear stamp detected (conf=%.2f, set=%s)",
            best_confidence, set_id,
        )
        return (True, best_confidence)

    return (False, 0.0)


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



# ---------------------------------------------------------------------------
# Special Delivery promo detection (ID-based, no visual analysis)
# ---------------------------------------------------------------------------

def _check_special_delivery(img_bgr: np.ndarray, card_id: str) -> dict:
    """Detect Special Delivery Pokemon Center promo.

    These are specific SVP promo cards -- check card_id against known list.
    Known cards: SWSH074 (Pikachu), SWSH075 (Charizard), SWSH177 (Bidoof).

    No visual detection needed -- the card_id uniquely identifies these promos.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    # Strip variant suffix: "swshp-SWSH074/normal" -> "swshp-SWSH074"
    bare_id = card_id.split("/")[0] if "/" in card_id else card_id

    is_special = bare_id in _SPECIAL_DELIVERY_CARD_IDS

    return {
        "detected": is_special,
        "confidence": 1.0 if is_special else 0.0,
        "position": "card_id",
        "evidence": f"card_id_match:{bare_id}" if is_special else "no_match",
    }


# ---------------------------------------------------------------------------
# Pokemon Day event stamp detection (OCR-based)
# ---------------------------------------------------------------------------

def _check_pokemon_day_stamp(img_bgr: np.ndarray, set_id: str,
                             era: int) -> dict:
    """Detect Pokemon Day event stamp on card artwork.

    Pokemon Day promos have a "POKEMON DAY" logo stamped on the card artwork,
    typically in the bottom-right area.  Uses OCR to scan for the text.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False,
        "confidence": 0.0,
        "position": "artwork_bottom_right",
    }

    regions = STAMP_REGIONS["pokemon_day"]

    best_score = 0.0
    best_evidence = ""

    for region_name, coords in [("tight", regions["tight"]),
                                ("wide", regions["wide"])]:
        crop = _extract_region(img_bgr, *coords)
        if crop.size == 0:
            continue

        ocr_text = _ocr_region(crop)
        if not ocr_text:
            continue

        is_match, score, method = _is_pokemon_day_text(ocr_text)
        if is_match and score > best_score:
            best_score = score
            best_evidence = f"ocr_{region_name}:{method}:{ocr_text[:50]}"

    if best_score > 0:
        return {
            "detected": True,
            "confidence": min(0.95, best_score / 100.0),
            "position": "artwork_bottom_right",
            "evidence": best_evidence,
        }

    return result_base


def _is_pokemon_day_text(text: str) -> tuple[bool, float, str]:
    """Check if OCR text matches "POKEMON DAY" using fuzzy matching.

    Returns (is_match, score, method).
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.debug("rapidfuzz not available for Pokemon Day matching")
        return (False, 0.0, "no_rapidfuzz")

    clean = text.strip().upper()

    # Direct substring check -- most reliable
    if "POKEMON DAY" in clean or "POKÉMON DAY" in clean:
        return (True, 95.0, "exact_substring")

    # Partial ratio against "POKEMON DAY" to handle OCR noise
    partial = fuzz.partial_ratio(clean, "POKEMON DAY")
    if partial >= 80:
        return (True, partial, "partial_ratio")

    # Token-level check: both "POKEMON" and "DAY" present (possibly garbled)
    tokens = clean.replace(".", " ").replace(",", " ").split()
    has_pokemon = any(
        fuzz.ratio(t, "POKEMON") >= 70 or fuzz.ratio(t, "POKÉMON") >= 70
        for t in tokens
    )
    has_day = any(fuzz.ratio(t, "DAY") >= 70 for t in tokens)

    if has_pokemon and has_day:
        return (True, 80.0, "token_match")

    return (False, max(partial, 0.0), "none")



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
    # The B&B stamp is distinguished from red artwork by the CO-OCCURRENCE
    # of red, white, AND dark pixels.  Red cards (Arcanine, Orthworm) have
    # red_frac > 0.8 but white_frac = 0 and dark_frac = 0.  The stamp has
    # red ~0.41, white ~0.025, dark ~0.06.
    #
    # Strategy: require at least 2 of {white_present, dark_present,
    # pokeball_blob, context_spike} alongside red, rather than summing
    # independent weak signals that fire on any warm-toned card.
    score = 0.0
    evidence_parts = []

    # Gate: red must be present (but not too high -- pure red = artwork)
    has_moderate_red = 0.15 < tight_red_frac < 0.70
    has_white = tight_white_frac > 0.005
    has_dark = tight_dark_frac > 0.02
    has_context_spike = red_ratio_vs_context > 1.3

    # Count co-occurrence signals (stamp-specific, not just "has red")
    cooccurrence_count = sum([has_white, has_dark, has_pokeball_blob,
                              has_context_spike])

    if has_moderate_red:
        evidence_parts.append("moderate_red")
        # Base score from red
        score += 0.15

        # Co-occurrence bonus: each additional signal adds confidence
        if has_white:
            score += 0.15
            evidence_parts.append("white_pixels")
        if has_dark:
            score += 0.15
            evidence_parts.append("dark_pixels")
        if has_pokeball_blob:
            score += 0.20
            evidence_parts.append("pokeball_blob")
        if has_context_spike:
            score += 0.15
            evidence_parts.append("red_spike_vs_context")
        if left_redder and has_pokeball_blob:
            score += 0.05
            evidence_parts.append("lr_structure")

        # Penalty: pure red artwork (no co-occurrence signals)
        if cooccurrence_count == 0:
            score = max(score - 0.10, 0.0)
            evidence_parts.append("no_cooccurrence_penalty")
    else:
        # Not moderate red -- could still detect with very strong signals
        if tight_red_frac >= 0.70:
            evidence_parts.append("red_too_high")
        elif tight_red_frac <= 0.15:
            evidence_parts.append("red_too_low")

        # Only pokeball blob + context can trigger without moderate red
        if has_pokeball_blob and has_context_spike and has_white:
            score += 0.45
            evidence_parts.extend(["pokeball_blob", "red_spike_vs_context",
                                   "white_pixels"])

    score = min(score, 1.0)
    detected = score >= 0.40 and cooccurrence_count >= 2
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


def _is_winner_text(text: str) -> tuple[bool, float, str]:
    """Check if OCR text matches "WINNER" using fuzzy matching.

    Winner stamps read "WINNER" (6 chars) or lowercase "winner" (WotC era).
    Uses length and character guards similar to _is_prerelease_text.

    Returns (is_match, score, method).
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return (False, 0.0, "no_rapidfuzz")

    clean = text.strip().upper().replace(" ", "")

    # Exact match
    if clean == "WINNER":
        return (True, 100.0, "exact")

    # Too short or too long to be "WINNER" (6 chars)
    if len(clean) < 4 or len(clean) > 12:
        return (False, 0.0, "length_guard")

    ratio = fuzz.ratio(clean, "WINNER")

    # Character composition: WINNER letters are W, I, N, E, R
    winner_chars = set("WINER")
    char_overlap = (sum(1 for c in clean if c in winner_chars) / len(clean)
                    if clean else 0.0)

    # High confidence: fuzzy ratio >= 75, text is short (just the stamp)
    if ratio >= 75 and 4 <= len(clean) <= 10:
        return (True, ratio, "fuzzy_short")

    # Medium confidence: ratio >= 70 with high character overlap
    if ratio >= 70 and char_overlap >= 0.7 and len(clean) >= 4:
        return (True, ratio, "fuzzy_overlap")

    # Partial match for longer garbled reads containing "WINNER"
    partial = fuzz.partial_ratio(clean, "WINNER")
    if partial >= 90 and 5 <= len(clean) <= 14 and char_overlap >= 0.6:
        return (True, partial, "partial")

    return (False, max(ratio, partial), "none")


def _check_winner_stamp(img_bgr: np.ndarray, set_id: str = "",
                        era: int = 0) -> dict:
    """Detect WINNER tournament stamp on artwork.

    Winner stamps appear on tournament prize cards in three eras:
      - WotC (2002-2003): gold foil lowercase "winner" with star for "i" dot
      - Pokemon USA (2003-2004): Poke Ball + "WINNER" text
      - Modern (2025+): Poke Ball + "WINNER" in caps

    All three appear in the bottom-right of the artwork area (same general
    region as prerelease stamps). Detection uses multi-preprocessing OCR
    with fuzzy matching against "WINNER".

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set identifier (e.g. "base2", "sv1").
        era: Era number (1-9, 0 if unknown).

    Returns:
        dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["winner_stamp"]
    result_base = {"detected": False, "confidence": 0.0,
                   "position": "artwork_bottom_right"}

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
            is_match, score, method = _is_winner_text(text)
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
        logger.debug("Winner stamp detected: %r score=%.1f via=%s method=%s",
                     best_text, best_score, best_strategy, best_method)
        return {
            "detected": True, "confidence": round(confidence, 2),
            "position": "artwork_bottom_right",
            "evidence": f"ocr_{best_method}",
            "ocr_text": best_text, "ocr_score": best_score,
            "ocr_strategy": best_strategy,
        }

    logger.debug("Winner stamp not found (best score=%.1f, text=%r)",
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
# diagonal lines (~45 and ~135 degrees) baked into the foil.

_LEAGUE_PROMO_SETS = frozenset({
    "dpp",    # DP-era league promos
    "hsp",    # HGSS-era league promos
    "bwp",    # BW-era league/tournament promos
    "xyp",    # XY-era league/tournament promos
    "smp",    # SM-era league/tournament promos
    "swshp",  # SWSH-era league/tournament promos
    "svp",    # SV-era league/tournament promos
})

# Artwork region for crosshatch analysis (x0, y0, x1, y1 as fractions).
_CROSSHATCH_REGION = (0.10, 0.12, 0.90, 0.56)

# Angular wedge half-width (degrees) for collecting energy around each
# diagonal axis in the FFT magnitude spectrum.
_CROSSHATCH_WEDGE_DEG = 12

# Minimum ratio of diagonal-wedge energy to median angular energy for a
# single axis to count as a "peak".  Crosshatch produces concentrated
# diagonal energy; random artwork textures distribute energy uniformly.
_CROSSHATCH_MIN_PEAK_RATIO = 1.8

# Minimum product of the two diagonal peak ratios.  Crosshatch needs BOTH
# diagonals elevated.  A single bright edge (card border, sleeve glare)
# scores high on one axis but near 1.0 on the other, keeping the product
# below this threshold.
_CROSSHATCH_MIN_PRODUCT = 4.0


def _check_crosshatch_holo(img_bgr: np.ndarray, set_id: str,
                            era: int) -> dict:
    """Detect crosshatch holo pattern (league/tournament exclusive).

    Crosshatch = regular grid of holo lines at ~45 degree angles.
    Region: artwork area (x:10-90%, y:12-56%).

    Detection approach -- FFT angular energy analysis:
        1. Extract the artwork region and convert to grayscale.
        2. High-pass filter to remove the printed image, isolating the
           foil texture (crosshatch lines are a physical overlay).
        3. Apply a Hanning window to reduce spectral leakage from the
           rectangular crop boundary.
        4. Compute 2D FFT magnitude spectrum (shifted so DC is center).
        5. Build an angular energy profile: for each angle 0-179 degrees,
           compute the mean FFT magnitude in a narrow wedge at that angle.
        6. Crosshatch has strong peaks at two perpendicular diagonal
           orientations (~45 and ~135 degrees).  Compute the ratio of
           energy in each diagonal wedge vs the median angular energy.
        7. Require BOTH diagonals to show elevated energy (peak_ratio
           above threshold) to distinguish from single-direction artifacts
           like card edges or sleeve reflections.

    Feasibility note -- binder sleeves:
        Crosshatch is a physical foil texture that catches light even
        through sleeves, but detection depends on photo angle and lighting.
        Under flat/even lighting the grid lines may be invisible to the
        camera, making FFT analysis indistinguishable from non-holo.
        Under angled light with glare, the diagonal grid becomes visible
        as bright streaks that produce clear FFT peaks.  Expect moderate
        recall (~30-50%) on binder page photos.  False positive rate
        should be low because perpendicular diagonal grids at consistent
        spacing rarely occur in printed card artwork.

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

    # Crosshatch holos exist from DP era onward (era 3+).
    if era != 0 and era < 3:
        return result_base

    h, w = img_bgr.shape[:2]
    if h < 80 or w < 60:
        return result_base

    # --- Extract artwork region ---
    ax0, ay0, ax1, ay1 = _CROSSHATCH_REGION
    artwork = img_bgr[int(ay0 * h):int(ay1 * h), int(ax0 * w):int(ax1 * w)]
    if artwork.size == 0:
        return result_base

    ah, aw = artwork.shape[:2]
    if ah < 32 or aw < 32:
        return result_base

    # --- Compute angular energy profile via FFT ---
    profile = _crosshatch_angular_profile(artwork)
    if profile is None:
        return result_base

    # --- Measure diagonal peak strength ---
    # Profile covers 0-179 degrees.  Crosshatch diagonals appear at ~45
    # and ~135 degrees in the image, which map to the same angles in the
    # FFT spectrum (frequency-domain angles match spatial angles).
    wedge = _CROSSHATCH_WEDGE_DEG

    # Median energy across all angles (baseline).
    median_energy = float(np.median(profile))
    if median_energy < 1e-6:
        return result_base

    # Energy in each diagonal wedge (mean over wedge width).
    diag1_energy = float(np.mean(profile[45 - wedge:45 + wedge + 1]))
    diag2_energy = float(np.mean(profile[135 - wedge:135 + wedge + 1]))

    # Ratios relative to median.
    diag1_ratio = diag1_energy / median_energy
    diag2_ratio = diag2_energy / median_energy
    product = diag1_ratio * diag2_ratio

    is_league_set = set_id in _LEAGUE_PROMO_SETS

    # --- Score confidence ---
    # Both diagonals must be elevated.  Single-diagonal artifacts (card
    # border, sleeve edge) score high on one axis but near 1.0 on the
    # other, producing a low product.
    confidence = 0.0

    if (diag1_ratio >= _CROSSHATCH_MIN_PEAK_RATIO
            and diag2_ratio >= _CROSSHATCH_MIN_PEAK_RATIO
            and product >= _CROSSHATCH_MIN_PRODUCT):
        # Base confidence from product strength.
        if product >= 12.0:
            confidence = 0.80
        elif product >= 8.0:
            confidence = 0.65
        elif product >= _CROSSHATCH_MIN_PRODUCT:
            confidence = 0.50

        # Bonus for very strong individual peaks.
        if min(diag1_ratio, diag2_ratio) >= 3.0:
            confidence += 0.10

        # Bonus for balanced diagonals (ratio of ratios near 1.0).
        balance = (min(diag1_ratio, diag2_ratio)
                   / max(diag1_ratio, diag2_ratio))
        if balance >= 0.6:
            confidence += 0.05

    # League set prior: slight boost.
    if is_league_set:
        confidence += 0.10

    confidence = max(0.0, min(1.0, confidence))
    detected = confidence >= 0.45

    evidence_parts = [
        f"d1={diag1_ratio:.2f}",
        f"d2={diag2_ratio:.2f}",
        f"prod={product:.2f}",
    ]
    if is_league_set:
        evidence_parts.append("league_set")
    evidence_str = ",".join(evidence_parts)

    if detected:
        logger.info(
            "Crosshatch holo detected: conf=%.2f, diag1=%.2f, "
            "diag2=%.2f, product=%.2f, league_set=%s, set=%s",
            confidence, diag1_ratio, diag2_ratio, product,
            is_league_set, set_id,
        )
        return {
            "detected": True, "confidence": round(confidence, 2),
            "position": "artwork", "evidence": evidence_str,
        }

    logger.debug(
        "Crosshatch holo not detected: conf=%.2f, d1=%.2f, d2=%.2f, "
        "prod=%.2f, set=%s",
        confidence, diag1_ratio, diag2_ratio, product, set_id,
    )
    return result_base


def _crosshatch_angular_profile(region_bgr: np.ndarray) -> np.ndarray | None:
    """Compute angular energy distribution of the high-pass FFT spectrum.

    Isolates foil texture by subtracting a blurred version (high-pass),
    then computes the 2D FFT and bins magnitude by angle (0-179 degrees).

    The high-pass step is critical: without it, the printed card artwork
    dominates the FFT and masks the subtle crosshatch frequency peaks.

    Returns:
        1D array of length 180 (mean energy per degree bin), or None if
        the region is too small.
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape
    if rh < 32 or rw < 32:
        return None

    # High-pass: subtract a heavily blurred version to remove artwork.
    # Kernel size ~1/4 of the smaller dimension, must be odd.
    blur_k = max(3, (min(rh, rw) // 4) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    highpass = cv2.subtract(gray, blurred).astype(np.float32)

    # Hanning window to suppress spectral leakage from rectangular edges.
    win_y = np.hanning(rh).astype(np.float32)
    win_x = np.hanning(rw).astype(np.float32)
    highpass *= np.outer(win_y, win_x)

    # 2D FFT, shift DC to center.
    fft_shift = np.fft.fftshift(np.fft.fft2(highpass))
    mag = np.abs(fft_shift)

    # Zero out DC neighborhood (residual low-frequency energy).
    cy, cx = rh // 2, rw // 2
    dc_radius = max(2, min(rh, rw) // 20)
    Y, X = np.ogrid[:rh, :rw]
    mag[(X - cx) ** 2 + (Y - cy) ** 2 <= dc_radius ** 2] = 0.0

    # Precompute angle of each pixel relative to center (0-179 degrees).
    yy = np.arange(rh, dtype=np.float32) - cy
    xx = np.arange(rw, dtype=np.float32) - cx
    XX, YY = np.meshgrid(xx, yy)
    # arctan2(-YY, XX) gives angle from positive-x axis, counter-clockwise.
    # Mod 180 folds the spectrum (symmetric for real input).
    angle_map = np.degrees(np.arctan2(-YY, XX)) % 180.0
    angle_bins = np.clip(angle_map.astype(int), 0, 179)

    # Mean magnitude in each 1-degree angular bin.
    profile = np.zeros(180, dtype=np.float64)
    for deg in range(180):
        mask = angle_bins == deg
        if mask.any():
            profile[deg] = float(np.mean(mag[mask]))

    return profile

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



# ---------------------------------------------------------------------------
# League / Championship / Professor stamp detection
# ---------------------------------------------------------------------------

# Keyword patterns for each league stamp type.  Keys are stamp sub-types,
# values are lists of (keyword_phrase, min_fuzzy_score) pairs.  We match
# against OCR text from the full card using both exact substring and fuzzy
# matching so embossed / foil text with OCR errors still triggers.

_LEAGUE_STAMP_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "league": [
        ("league", 80),
        ("play pokemon", 75),
        ("play! pokemon", 75),
    ],
    "league_challenge_1st": [
        ("1st place", 80),
        ("first place", 80),
    ],
    "league_challenge_2nd": [
        ("2nd place", 80),
        ("second place", 80),
    ],
    "league_challenge_3rd": [
        ("3rd place", 80),
        ("third place", 80),
    ],
    "league_challenge_4th": [
        ("4th place", 80),
        ("fourth place", 80),
    ],
    "championship": [
        ("champion", 80),
        ("championship", 80),
        ("regional", 80),
        ("international", 80),
    ],
    "professor": [
        ("professor", 80),
        ("professor program", 75),
    ],
}

# All league stamp sub-types as a flat set for quick lookup.
_LEAGUE_STAMP_TYPES = frozenset(_LEAGUE_STAMP_KEYWORDS.keys())


def _check_league_stamps(img_bgr: np.ndarray, set_id: str,
                         era: int) -> dict:
    """Detect league/championship/professor stamps on a card.

    These stamps can appear anywhere on the card (artwork, text box, borders),
    so we OCR the full card with multiple preprocessing strategies and scan
    all extracted text for keyword matches.

    Returns dict compatible with the stamp dispatcher::

        {
            "detected": True/False,
            "confidence": float,
            "position": "full_card",
            "evidence": str,       # best matching keyword
            "stamp_type": str,     # sub-type (e.g. "league_challenge_1st")
            "all_stamps": dict,    # {sub_type: (detected, confidence)}
        }
    """
    from rapidfuzz import fuzz

    result_base = {
        "detected": False,
        "confidence": 0.0,
        "position": "full_card",
        "evidence": "",
        "stamp_type": "",
        "all_stamps": {k: (False, 0.0) for k in _LEAGUE_STAMP_KEYWORDS},
    }

    try:
        # OCR the full card with multiple preprocessing strategies.
        # League stamps are often embossed/foil and hard to read.
        ocr_hits = _ocr_region_multi(img_bgr, scale=2)
        if not ocr_hits:
            return result_base

        # Build a single lowercase text blob for substring matching,
        # plus keep individual lines for fuzzy matching.
        all_text = " ".join(t for t, _c, _s in ocr_hits).lower()
        ocr_lines = [(t.lower(), c) for t, c, _s in ocr_hits]

        best_type = ""
        best_conf = 0.0
        best_evidence = ""
        stamps_found: dict[str, tuple[bool, float]] = {}

        for stamp_type, keywords in _LEAGUE_STAMP_KEYWORDS.items():
            type_detected = False
            type_conf = 0.0
            type_evidence = ""

            for keyword, min_score in keywords:
                # --- Exact substring match (fast path) ---
                if keyword in all_text:
                    # Find the OCR confidence of the line containing match
                    line_conf = 0.0
                    for line_text, conf in ocr_lines:
                        if keyword in line_text:
                            line_conf = max(line_conf, conf)
                    match_conf = max(0.85, line_conf)
                    if match_conf > type_conf:
                        type_detected = True
                        type_conf = match_conf
                        type_evidence = f"exact:{keyword}"
                    continue

                # --- Fuzzy match against each OCR line ---
                for line_text, conf in ocr_lines:
                    if len(line_text) < 3:
                        continue
                    score = fuzz.partial_ratio(keyword, line_text)
                    if score >= min_score:
                        # Scale confidence by fuzzy score and OCR confidence
                        match_conf = (score / 100.0) * max(0.5, conf)
                        if match_conf > type_conf:
                            type_detected = True
                            type_conf = match_conf
                            type_evidence = (f"fuzzy:{keyword}"
                                             f"(score={score})")

            stamps_found[stamp_type] = (type_detected, round(type_conf, 3))

            if type_detected and type_conf > best_conf:
                best_conf = type_conf
                best_type = stamp_type
                best_evidence = type_evidence

        result_base["all_stamps"] = stamps_found

        if best_type:
            result_base["detected"] = True
            result_base["confidence"] = round(best_conf, 2)
            result_base["evidence"] = best_evidence
            result_base["stamp_type"] = best_type
            logger.info(
                "League stamp detected: %s (conf=%.2f, evidence=%s)",
                best_type, best_conf, best_evidence,
            )

        return result_base

    except Exception as e:
        logger.debug("League stamp check failed: %s", e)
        return result_base

def _check_peelable_ditto(img_bgr: np.ndarray, set_id: str) -> dict:
    """Detect Ditto face icon on Pokemon GO peelable cards.

    Pokemon GO (2022, set_id='pgo') has Bidoof, Numel, and Spinarak cards
    where some copies have a tiny purple Ditto face icon in the bottom-left
    corner.  Peeling off the sticker front reveals a Ditto card underneath.

    Icon position: bottom-left corner, x:5-15%, y:85-95%.
    The Ditto face is very small (~3-4mm on a real card) and purple/pink
    with a simple smiley shape.

    Detection strategy:
      1. Crop the bottom-left region.
      2. Upscale for sub-pixel analysis (the icon is tiny).
      3. Build an HSV mask for purple/pink hues (Ditto's signature color).
      4. Find contours in the mask and look for a small, roughly circular
         cluster of purple pixels -- the Ditto face.

    Only applies to the pgo set.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False, "confidence": 0.0, "position": "bottom_left",
    }

    if set_id != "pgo":
        return result_base

    regions = STAMP_REGIONS["peelable_ditto"]
    crop = _extract_region(img_bgr, *regions["wide"])
    if crop.size == 0:
        return result_base

    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return result_base

    # Upscale small crops so color detection is reliable.
    # Target at least 100px on the short side.
    scale = max(1, 100 // min(h, w))
    if scale > 1:
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Ditto purple/pink sits in two HSV hue ranges:
    #   - Main purple:  hue 120-160 (OpenCV 0-180 scale), S 30-255, V 60-255
    #   - Pink wrap:    hue 160-175, S 30-255, V 60-255
    mask_purple = cv2.inRange(hsv, (120, 30, 60), (175, 255, 255))

    # Also catch lighter lavender tones that may appear under flash/glare.
    mask_lavender = cv2.inRange(hsv, (110, 20, 100), (145, 180, 255))

    mask = cv2.bitwise_or(mask_purple, mask_lavender)

    # Morphological close to merge nearby purple pixels into a blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the purple mask.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return result_base

    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_h * crop_w

    best_score = 0.0
    best_evidence = ""

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / crop_area

        # The Ditto face icon should be a meaningful fraction of the crop
        # but not huge.  Expected range: 1-20% of the wide crop area.
        if area_frac < 0.005 or area_frac > 0.30:
            continue

        # Check circularity -- the face is roughly round.
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.3:
            continue

        # Compute bounding box aspect ratio (should be roughly square).
        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.5:
            continue

        # Score: weighted combination of circularity and size adequacy.
        # Ideal size ~5-15% of crop area.
        size_score = min(area_frac / 0.05, 1.0)
        circ_score = min(circularity / 0.7, 1.0)
        score = 0.5 * circ_score + 0.3 * size_score + 0.2 * aspect

        if score > best_score:
            best_score = score
            best_evidence = (
                f"purple_blob area_frac={area_frac:.3f} "
                f"circ={circularity:.2f} aspect={aspect:.2f}"
            )

    if best_score < 0.40:
        return result_base

    # Map raw score to confidence.
    confidence = min(0.95, 0.50 + best_score * 0.50)

    return {
        "detected": True,
        "confidence": round(confidence, 2),
        "position": "bottom_left",
        "evidence": best_evidence,
    }



# ---------------------------------------------------------------------------
# Wizards "W" gold foil stamp detection
# ---------------------------------------------------------------------------
# Only 7 cards were ever produced with a gold foil Wizards "W" logo stamp
# on the artwork (1999-2001 WotC promo era).  These are extremely rare
# promotional variants given out at events.
#
# Eligible cards (all from basep -- WotC Black Star Promos):
#   basep-1   Pikachu
#   basep-2   Electabuzz
#   basep-3   Mewtwo
#   basep-4   Pikachu (movie promo)
#   basep-6   Arcanine
#   basep-8   Mew
#   basep-9   Mew (movie promo)

_W_STAMP_ELIGIBLE_CARDS = frozenset({
    "basep-1", "basep-2", "basep-3", "basep-4",
    "basep-6", "basep-8", "basep-9",
})

# Gold foil HSV range: warm yellow-gold hue, moderate-to-high saturation,
# high value.  Tuned for the foil stamp under typical scan/photo lighting.
_W_STAMP_HUE_LO = 15
_W_STAMP_HUE_HI = 45
_W_STAMP_SAT_LO = 60
_W_STAMP_VAL_LO = 150


def _check_w_stamp(img_bgr: np.ndarray, set_id: str) -> dict:
    """Detect Wizards 'W' gold foil stamp on artwork.

    Only 7 cards ever received this stamp (all basep promos).  The stamp
    is a gold-colored Wizards logo "W" shape on the artwork area.

    Detection approach:
      1. Crop the artwork region (wide stamp region).
      2. Threshold for gold-colored pixels in HSV space.
      3. Find contours in the gold mask and filter by area and aspect ratio.
      4. Check for the distinctive "W" shape: a contour wider than it is
         tall with multiple downward-pointing vertices (3 or 5 peaks).
      5. Require minimum gold pixel density within the bounding box.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    result_base = {
        "detected": False,
        "confidence": 0.0,
        "position": "artwork",
        "evidence": "",
    }

    # Only basep cards can have this stamp
    if set_id != "basep":
        return result_base

    try:
        # Crop artwork region
        region = STAMP_REGIONS["w_stamp"]
        artwork = _extract_region(img_bgr, *region["wide"])
        h, w = artwork.shape[:2]
        if h < 20 or w < 20:
            return result_base

        art_area = h * w

        # Convert to HSV and threshold for gold-colored pixels
        hsv = cv2.cvtColor(artwork, cv2.COLOR_BGR2HSV)
        gold_mask = cv2.inRange(
            hsv,
            np.array([_W_STAMP_HUE_LO, _W_STAMP_SAT_LO, _W_STAMP_VAL_LO]),
            np.array([_W_STAMP_HUE_HI, 255, 255]),
        )

        # Morphological cleanup: close small gaps, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the gold mask
        contours, _ = cv2.findContours(
            gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return result_base

        # Filter contours by area: stamp should be 0.5%-8% of artwork area
        min_area = art_area * 0.005
        max_area = art_area * 0.08
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                candidates.append(cnt)

        if not candidates:
            return result_base

        # Evaluate each candidate for "W" shape characteristics
        best_score = 0.0
        best_evidence = ""

        for cnt in candidates:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0 or cw == 0:
                continue

            aspect = cw / ch
            # "W" is wider than tall (aspect ratio ~1.2-3.0)
            if aspect < 1.0 or aspect > 4.0:
                continue

            # Gold pixel density within the bounding box
            roi_mask = gold_mask[y:y + ch, x:x + cw]
            density = np.count_nonzero(roi_mask) / (cw * ch)

            # A solid "W" stamp has moderate density (not a filled rectangle,
            # not just scattered pixels).  Expect 25%-75%.
            if density < 0.15 or density > 0.85:
                continue

            # Approximate the contour and look for the "W" zigzag pattern.
            # A "W" has 4-6 dominant vertices forming peaks and valleys.
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            n_vertices = len(approx)

            # Score components
            score = 0.0

            # Aspect ratio score: ideal ~1.5-2.5
            if 1.2 <= aspect <= 3.0:
                score += 0.25
            elif 1.0 <= aspect <= 4.0:
                score += 0.10

            # Density score
            if 0.25 <= density <= 0.65:
                score += 0.25
            elif 0.15 <= density <= 0.75:
                score += 0.15

            # Vertex count score: "W" typically approximates to 6-12 points
            if 5 <= n_vertices <= 14:
                score += 0.25
            elif 4 <= n_vertices <= 18:
                score += 0.10

            # Check for multiple valleys (downward peaks) in the contour.
            # A "W" has 2 valleys along the bottom edge.
            pts = approx.reshape(-1, 2)
            # Normalize y-coords relative to bounding box
            y_norm = (pts[:, 1] - y) / ch
            # Bottom-half points (y_norm > 0.4) that dip down
            bottom_pts = pts[y_norm > 0.4]
            if len(bottom_pts) >= 2:
                # Count local minima in x-sorted bottom points
                x_order = bottom_pts[bottom_pts[:, 0].argsort()]
                if len(x_order) >= 3:
                    valleys = 0
                    for i in range(1, len(x_order) - 1):
                        if (x_order[i, 1] > x_order[i - 1, 1]
                                and x_order[i, 1] > x_order[i + 1, 1]):
                            valleys += 1
                    if valleys >= 1:
                        score += 0.25
                    elif valleys == 0 and len(bottom_pts) >= 3:
                        score += 0.05

            evidence = (
                f"aspect={aspect:.2f}, density={density:.2f}, "
                f"vertices={n_vertices}, "
                f"area_frac={cv2.contourArea(cnt) / art_area:.4f}"
            )

            if score > best_score:
                best_score = score
                best_evidence = evidence

        if best_score < 0.50:
            return result_base

        # Map score to confidence (0.50-1.00 -> 0.55-0.95)
        confidence = min(0.95, 0.55 + (best_score - 0.50) * 0.80)

        return {
            "detected": True,
            "confidence": round(confidence, 2),
            "position": "artwork",
            "evidence": best_evidence,
        }

    except Exception as e:
        logger.debug("W stamp check failed: %s", e)
        return result_base


def _get_stamps_to_check(card_id: str, set_id: str, era: int,
                         fast: bool = False) -> list[str]:
    """Return list of stamp types to check based on era and set.

    The key principle: only check stamps that are POSSIBLE for this card's
    era and set.  No point checking for 1st Edition on a Sword & Shield card.

    When ``fast=True``, skip OCR-heavy checks (1st_edition, prerelease,
    staff_stamp, ex_set_stamp) and holo-pattern analysis (holo_finish,
    reverse_holo, crosshatch_holo).  1st edition alone takes ~25s (3 OCR
    passes).  Fast mode runs only cheap pixel-based checks (shadowless).
    All OCR-based checks are skipped in fast mode.
    """
    checks = []

    # --- Cheap pixel-based checks (always run) ---

    # World Championship deck detection — border saturation analysis, <1ms
    # Runs for ALL cards: WC decks exist across many eras/sets
    checks.append("world_championship")

    # Shadowless (base1 only) — pure pixel gradient analysis, <5ms
    if set_id == "base1":
        checks.append("shadowless")

    # Special Delivery promo (ID-based, no visual analysis, <1ms)
    if set_id == "swshp":
        checks.append("special_delivery")

    # --- OCR-based checks: skip ALL in fast mode ---
    if not fast:
        # WotC era (1): 1st Edition + copyright year + Black Star Promo
        if set_id in _FIRST_EDITION_SETS:
            checks.append("1st_edition")
        if set_id == "base1":
            checks.append("copyright_year")
            checks.append("ghost_stamp")
        if set_id in _BLACK_STAR_PROMO_SETS:
            checks.append("black_star_promo")

        # Wizards "W" gold foil stamp (basep promos only, 7 eligible cards)
        if set_id == "basep":
            bare_id = card_id.split("/")[0]
            if bare_id in _W_STAMP_ELIGIBLE_CARDS:
                checks.append("w_stamp")

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

        # Toys R Us stamp: XY/SM eras (2016-2018, Generations-Ultra Prism)
        if era in _TOYS_R_US_ERAS or era == 0:
            checks.append("toys_r_us")

        # Prerelease stamps: era-gated (text-based for WotC/EX/DP, logo for HGSS+)
        if set_id in _PRERELEASE_TEXT_SETS or set_id in _PRERELEASE_LOGO_SETS:
            checks.append("prerelease")

        # Staff stamp: check proactively alongside prerelease (era 3+ = DP onward)
        if era >= 3 or set_id in _PRERELEASE_TEXT_SETS:
            checks.append("staff_stamp")

        # Winner tournament stamp: same sets as prerelease (event promos)
        if set_id in _WINNER_STAMP_SETS:
            checks.append("winner_stamp")

        # === Holo pattern checks ===

        # Holo finish detection (artwork area holographic signal)
        if era >= 1 or era == 0:
            checks.append("holo_finish")

        # Reverse holo detection (body area holographic signal, era 2+ only)
        if era >= 2 or era == 0:
            checks.append("reverse_holo")

        # DP+ eras (3-9): crosshatch holo (league/tournament promos)
        if era == 0 or era >= 3:
            checks.append("crosshatch_holo")

        # === SV-specific stamps ===

        # Build & Battle stamp (SV era SVP promos)
        if set_id == "svp" and (era >= 8 or era == 0):
            checks.append("build_battle")

        # Pokemon Center exclusive stamp (SVP promos, SWSH/SV era)
        if set_id == "svp":
            checks.append("pokemon_center")

        # === Special error variants ===

        # Jungle no-symbol error: base2 holos only
        if set_id == "base2":
            variant = card_id.split("/", 1)[1] if "/" in card_id else ""
            if "holo" in variant.lower():
                checks.append("no_symbol_error")

        # Build-A-Bear Workshop stamp (XY/SM era, eras 6-7)
        if era in (6, 7) or era == 0:
            checks.append("build_a_bear")

        # McDonald's sets: confetti holo detection
        if set_id in _MCDONALDS_SETS or set_id.startswith("mcd"):
            checks.append("mcdonalds_holo")

        # Sequin holo: General Mills cereal promo exclusive (SM/SWSH era)
        if set_id in _GENERAL_MILLS_SETS:
            checks.append("sequin_holo")

        # League / Championship / Professor stamps: any era from DP onward
        # (era 3+).  League promos exist in older eras but are extremely rare
        # and the crosshatch holo check already flags them.
        if era >= 3 or era == 0:
            checks.append("league_stamps")

        # Pokemon GO peelable Ditto face icon (cheap color check, no OCR)
        if set_id == "pgo":
            checks.append("peelable_ditto")

        # Pokemon Day event stamp (OCR-based, any promo set SWSH+)
        if set_id in _MODERN_PROMO_SETS or set_id in _PROMO_SETS:
            checks.append("pokemon_day")

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

    # --- Border hue variance (rainbow shimmer detector) ---
    # Real reverse holos produce rapid hue shifts across the border from
    # holographic foil.  Colored theme borders (e.g. Team Aqua blue,
    # Team Magma red) have uniformly high saturation but *low* hue variance.
    hue = hsv[:, :, 0].astype(np.float32)
    border_left_hue = hue[int(h * 0.15):int(h * 0.85),
                          int(w * 0.05):int(w * 0.13)]
    border_right_hue = hue[int(h * 0.15):int(h * 0.85),
                           int(w * 0.87):int(w * 0.95)]
    border_hue_std = float(np.concatenate([
        border_left_hue.flatten(), border_right_hue.flatten(),
    ]).std())

    logger.debug(
        "Reverse holo check: border_sat_std=%.1f art_sat_std=%.1f "
        "border_cvar=%.1f art_cvar=%.1f border_hue_std=%.1f (set=%s era=%d)",
        border_sat_std, art_sat_std, border_cvar, art_cvar, border_hue_std,
        set_id, era,
    )

    # --- Decision logic ---
    # Primary signal: border_sat_std vs art_sat_std (spec thresholds)
    # Secondary signal: border_cvar vs art_cvar (cross-channel shimmer)
    # Guard: border_hue_std must be high enough to indicate rainbow shimmer
    #   (not just a uniformly colored border like Team Aqua/Magma)
    #   Normal colored borders: hue_std ~1-4; real reverse holos: hue_std 10+

    # Reverse holo: shiny borders, matte artwork
    if border_sat_std > 25 and art_sat_std < 20 and border_hue_std > 8:
        conf = min(0.95, 0.70 + (border_sat_std - 25) / 100)
        return ("reverse_holo", conf)

    # Also catch reverse holos via cross-channel variance when sat_std
    # is ambiguous (artwork has some natural color variation)
    if border_cvar > 30 and art_cvar < 15 and border_hue_std > 8:
        conf = min(0.90, 0.65 + (border_cvar - 30) / 100)
        return ("reverse_holo", conf)

    # Combined: both metrics lean reverse-holo but neither decisive alone
    if (border_sat_std > 25 and border_cvar > 25 and art_cvar < 20
            and border_hue_std > 8):
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

def _build_a_bear_as_dict(img_bgr: np.ndarray, set_id: str,
                         era: int) -> dict:
    """Wrap _check_build_a_bear_stamp tuple return into a dict for dispatch."""
    is_bab, conf = _check_build_a_bear_stamp(img_bgr, set_id, era)
    return {
        "detected": is_bab,
        "confidence": conf,
        "position": "artwork",
        "evidence": "build_a_bear_ocr" if is_bab else "",
        "retailer": "Build-A-Bear" if is_bab else None,
    }


def detect_stamps(image_path: str, card_id: str,
                  fast: bool = True) -> dict:
    """Detect physical stamps on a card based on its era.

    After card identification, this function checks for stamps that are
    appropriate for the card's era and set.  Each stamp type is checked
    in the CORRECT region of the card (fixed position for each stamp type).

    Args:
        image_path: Path to the card image.
        card_id: Full card identifier (e.g. "base1-4/holofoil").
        fast: If True (default), skip ALL OCR-based and pattern-analysis
            checks.  Only runs cheap pixel-based checks (shadowless).
            OCR checks (1st_edition, promo, copyright_year, prerelease,
            etc.) take 2-25s each.  Set to False for full analysis.

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

    stamps_to_check = _get_stamps_to_check(card_id, set_id, era, fast=fast)

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

    def _sequin_holo_as_dict(img_bgr, sid, e):
        """Wrap _check_sequin_holo (returns tuple) into a dict for dispatch."""
        label, conf, det = _check_sequin_holo(img_bgr, sid, e)
        return {
            "detected": label == "sequin",
            "confidence": conf,
            "position": "artwork",
            "evidence": "sequin_holo_detector",
            "holo_type": label,
            "metrics": det,
        }

    # Dispatch to appropriate checker
    _STAMP_CHECKERS = {
        "world_championship": lambda img_bgr: _check_world_championship(img_bgr, set_id),
        "1st_edition": lambda img_bgr: _check_1st_edition(img_bgr),
        "1st_edition_dino": lambda img_bgr: _check_1st_edition_dino(img_bgr, card_id, set_id),
        "ghost_stamp": lambda img_bgr: _check_ghost_stamp(img_bgr, set_id),
        "ex_set_stamp": lambda img_bgr: _check_ex_set_stamp(img_bgr, set_id, card_id),
        "black_star_promo": lambda img_bgr: _check_black_star_promo(img_bgr),
        "modern_promo": lambda img_bgr: _check_modern_promo(img_bgr),
        "promo_stamp": lambda img_bgr: _check_promo_stamp(img_bgr),
        "copyright_year": lambda img_bgr: _check_copyright_year(img_bgr, set_id),
        "shadowless": lambda img_bgr: _check_shadowless(img_bgr, set_id),
        "pokemon_center": lambda img_bgr: _check_pokemon_center_stamp(img_bgr, set_id, era),
        "retailer_stamp": lambda img_bgr: _check_retailer_stamp(img_bgr, set_id, era),
        "toys_r_us": lambda img_bgr: _toys_r_us_as_dict(img_bgr, set_id, era),
        "prerelease": lambda img_bgr: _check_prerelease(img_bgr, set_id, era),
        "staff_stamp": lambda img_bgr: _check_staff_stamp(img_bgr, set_id, era),
        "winner_stamp": lambda img_bgr: _check_winner_stamp(img_bgr, set_id, era),
        "build_battle": lambda img_bgr: _check_build_battle_stamp(img_bgr, set_id, era),
        "build_a_bear": lambda img_bgr: _build_a_bear_as_dict(img_bgr, set_id, era),
        "holo_finish": _holo_finish_as_dict,
        "reverse_holo": _reverse_holo_as_dict,
        "no_symbol_error": lambda img_bgr: _check_no_symbol_error_as_stamp(
            img_bgr, set_id,
            card_id.split("/", 1)[1] if "/" in card_id else "",
        ),
        "crosshatch_holo": lambda img_bgr: _check_crosshatch_holo(img_bgr, set_id, era),
        "mcdonalds_holo": lambda img_bgr: _check_mcdonalds_holo(img_bgr, set_id, era),
        "sequin_holo": lambda img_bgr: _sequin_holo_as_dict(img_bgr, set_id, era),
        "league_stamps": lambda img_bgr: _check_league_stamps(img_bgr, set_id, era),
        "peelable_ditto": lambda img_bgr: _check_peelable_ditto(img_bgr, set_id),
        "w_stamp": lambda img_bgr: _check_w_stamp(img_bgr, set_id),
        "special_delivery": lambda img_bgr: _check_special_delivery(img_bgr, card_id),
        "pokemon_day": lambda img_bgr: _check_pokemon_day_stamp(img_bgr, set_id, era),
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
                # Include warning for world_championship detection
                if "warning" in detail:
                    stamp_info["warning"] = detail["warning"]
                # Include league stamp sub-type and per-type breakdown
                if "stamp_type" in detail and detail["stamp_type"]:
                    stamp_info["stamp_type"] = detail["stamp_type"]
                if "all_stamps" in detail:
                    stamp_info["all_stamps"] = detail["all_stamps"]
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


# ---------------------------------------------------------------------------
# Sequin holo detection (General Mills cereal promo exclusive, 2017+)
# ---------------------------------------------------------------------------

# General Mills cereal promo set IDs.  These are the only sets that can
# have sequin holo -- a sparkly pattern of large, distinct bright dots
# across the entire card surface, visually chunkier than cosmos holo.
_GENERAL_MILLS_SETS = frozenset({
    "sm01",   # Sun & Moon cereal promos (2017)
    "sm02",   # Guardians Rising cereal promos
    "sm03",   # Burning Shadows cereal promos
    "sm04",   # Crimson Invasion cereal promos
    "sm05",   # Ultra Prism cereal promos
    "sm06",   # Forbidden Light cereal promos
    "sm07",   # Celestial Storm cereal promos
    "sm08",   # Lost Thunder cereal promos
    "sm09",   # Team Up cereal promos
    "sm10",   # Unbroken Bonds cereal promos
    "sm11",   # Unified Minds cereal promos
    "sm12",   # Cosmic Eclipse cereal promos
    "swsh01", # Sword & Shield cereal promos
    "swsh02", # Rebel Clash cereal promos
    "swsh03", # Darkness Ablaze cereal promos
    "swsh04", # Vivid Voltage cereal promos
    "swsh05", # Battle Styles cereal promos
    "swsh06", # Chilling Reign cereal promos
    "swsh07", # Evolving Skies cereal promos
})


def _check_sequin_holo(
    img_bgr: np.ndarray,
    set_id: str,
    era: int,
) -> tuple[str, float, dict]:
    """Detect sequin holo pattern (cereal promo exclusive).

    Sequin holo is EXCLUSIVE to General Mills cereal promo packs (2017+).
    The pattern has larger, more distinct bright spots than cosmos holo --
    sparkly sequin-like dots scattered across the full card surface.

    Detection: blob analysis on the artwork region in HSV space.  Sequin
    dots are specular highlights (high V, moderate S) that are larger and
    fewer than cosmos dots.

    IMPORTANT: Sequin holo is probably not detectable through binder
    sleeves.  The specular highlights are diffused by the plastic, making
    the dots indistinguishable from general glare.  This detector will
    return 'unknown' conservatively rather than false-positive.

    Args:
        img_bgr: Full card image in BGR format.
        set_id: Set identifier (e.g. "sm03").
        era: Era number (1-9).

    Returns:
        (label, confidence, details) where label is 'sequin', 'not_sequin',
        or 'unknown'.
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = _ARTWORK_REGION
    art = img_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    if art.size == 0 or art.shape[0] < 30 or art.shape[1] < 30:
        return ("unknown", 0.0, {"error": "artwork_region_too_small"})

    art_h, art_w = art.shape[:2]
    art_area = art_h * art_w

    # Convert to HSV for specular highlight isolation
    hsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)

    # -------------------------------------------------------------------
    # Step 1: Isolate specular highlights (high brightness, moderate sat)
    # -------------------------------------------------------------------
    # Sequin dots are bright reflective spots: V > 220 and S in 20-180
    # (not pure white glare which has S~0, not deeply saturated art).
    p90_v = float(np.percentile(v_chan, 90))
    bright_thresh = max(220.0, p90_v + 15.0)

    specular_mask = (
        (v_chan >= bright_thresh) & (s_chan >= 20) & (s_chan <= 180)
    ).astype(np.uint8) * 255

    # -------------------------------------------------------------------
    # Step 2: Blob analysis on specular mask
    # -------------------------------------------------------------------
    # Find connected components of specular highlights.
    contours, _ = cv2.findContours(
        specular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Sequin dots are larger than cosmos dots but smaller than glare blobs.
    # Filter by area: sequin dot ~ 0.02-0.5% of artwork area.
    min_blob_area = max(8, int(art_area * 0.0002))
    max_blob_area = int(art_area * 0.005)

    blob_areas: list[float] = []
    blob_circularities: list[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_blob_area <= area <= max_blob_area:
            blob_areas.append(float(area))
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circ = 4.0 * np.pi * area / (perimeter * perimeter)
                blob_circularities.append(float(circ))

    blob_count = len(blob_areas)
    blob_density = blob_count / (art_area / 1000.0) if art_area > 0 else 0.0

    if blob_areas:
        mean_blob_area = float(np.mean(blob_areas))
        blob_area_cv = float(np.std(blob_areas) / (mean_blob_area + 1e-6))
    else:
        mean_blob_area = 0.0
        blob_area_cv = 1.0

    if blob_circularities:
        mean_circularity = float(np.mean(blob_circularities))
    else:
        mean_circularity = 0.0

    # -------------------------------------------------------------------
    # Step 3: Spatial distribution -- sequin dots should be spread
    # across the artwork, not clustered in one region
    # -------------------------------------------------------------------
    # Divide artwork into a 3x3 grid; count blobs per cell.
    grid_counts = np.zeros((3, 3), dtype=int)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_blob_area <= area <= max_blob_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                gi = min(2, cy * 3 // art_h)
                gj = min(2, cx * 3 // art_w)
                grid_counts[gi, gj] += 1

    cells_with_blobs = int(np.sum(grid_counts > 0))
    spatial_coverage = cells_with_blobs / 9.0

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------
    score = 0.0
    evidence_parts: list[str] = []

    # Blob count: sequin has moderate count (fewer than cosmos, more than
    # random glare).  Expect 15-80 blobs in artwork region.
    if 15 <= blob_count <= 80:
        score += 0.20
        evidence_parts.append(f"sequin_blob_count({blob_count})")
    elif blob_count > 80:
        # Too many small blobs -- more likely cosmos
        score -= 0.15
        evidence_parts.append(f"too_many_blobs({blob_count})")
    elif blob_count < 5:
        score -= 0.20
        evidence_parts.append(f"too_few_blobs({blob_count})")

    # Mean blob area: sequin dots are larger than cosmos dots
    mean_area_frac = mean_blob_area / art_area if art_area > 0 else 0
    if 0.0003 <= mean_area_frac <= 0.003:
        score += 0.20
        evidence_parts.append(f"sequin_size(frac={mean_area_frac:.5f})")
    elif mean_area_frac > 0.003:
        score -= 0.10
        evidence_parts.append(f"blobs_too_large(frac={mean_area_frac:.5f})")

    # Blob size uniformity: sequin dots are moderately uniform
    if blob_area_cv < 0.8 and blob_count >= 10:
        score += 0.15
        evidence_parts.append(f"uniform_size(cv={blob_area_cv:.2f})")

    # Circularity: sequin dots are roughly circular
    if mean_circularity > 0.5 and blob_count >= 10:
        score += 0.15
        evidence_parts.append(f"circular(mean={mean_circularity:.2f})")

    # Spatial coverage: dots should span most of the artwork
    if spatial_coverage >= 0.67:
        score += 0.15
        evidence_parts.append(f"good_coverage({spatial_coverage:.2f})")
    elif spatial_coverage < 0.33:
        score -= 0.15
        evidence_parts.append(f"poor_coverage({spatial_coverage:.2f})")

    # -------------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------------
    details = {
        "blob_count": blob_count,
        "blob_density": round(blob_density, 4),
        "mean_blob_area": round(mean_blob_area, 1),
        "mean_area_frac": round(mean_area_frac, 6),
        "blob_area_cv": round(blob_area_cv, 3),
        "mean_circularity": round(mean_circularity, 3),
        "spatial_coverage": round(spatial_coverage, 2),
        "cells_with_blobs": cells_with_blobs,
        "bright_thresh": round(bright_thresh, 1),
        "raw_score": round(score, 3),
        "evidence": evidence_parts,
    }

    # Very conservative thresholds.  Through binder sleeves the specular
    # highlights are diffused, so we almost always return 'unknown'.
    # Only confident detection on bare/unsleeved cards.
    if score >= 0.50:
        label = "sequin"
        confidence = min(0.80, 0.45 + score)
    elif score <= -0.20:
        label = "not_sequin"
        confidence = min(0.75, 0.45 + abs(score))
    else:
        label = "unknown"
        confidence = 0.25 + abs(score)

    details["label"] = label
    details["confidence"] = round(confidence, 3)

    logger.info(
        "Sequin holo check for set=%s era=%d: label=%s conf=%.2f "
        "score=%.3f [%s]",
        set_id, era, label, confidence, score, ", ".join(evidence_parts),
    )

    return (label, confidence, details)


# ---------------------------------------------------------------------------
# detect_all_variants: Complete conditional detection tree
# ---------------------------------------------------------------------------

def detect_all_variants(image_path: str, card_id: str,
                        fast: bool = False) -> dict:
    """Run ALL applicable variant detections based on card era and set.

    This is the single entry point for the complete variant detection tree.
    Each check is gated on the card's era, set, and properties so we never
    waste time on impossible variants (e.g. 1st Edition on SV cards).

    The conditional tree::

        ALL CARDS:
        |-- World Championship deck (grey borders) -> flag reproduction, STOP
        |
        |-- WotC Base Set (base1):
        |   |-- 1st Edition stamp
        |   |   |-- Grey stamp sub-variant [if 1st ed found]
        |   |   +-- Thick/thin stamp sub-variant [if 1st ed found]
        |   |-- Ghost stamp (partial 1st ed impression)
        |   |-- Shadowless vs Unlimited (border shadow gradient)
        |   +-- Copyright year (1999 vs 1999-2000)
        |
        |-- WotC Jungle (base2):
        |   |-- 1st Edition stamp
        |   +-- No-symbol error (holos only)
        |
        |-- WotC Fossil/Rocket (base3, base5):
        |   +-- 1st Edition stamp
        |
        |-- WotC Gym (gym1, gym2):
        |   +-- 1st Edition stamp
        |
        |-- WotC Neo (neo1-neo4):
        |   +-- 1st Edition stamp
        |
        |-- EX era ex7-ex16:
        |   +-- EX set logo stamp (reverse holo confirmation)
        |
        |-- Prerelease/Staff (era-gated):
        |   |-- WotC/EX/DP text-based prerelease
        |   |-- HGSS+ logo-based prerelease
        |   +-- Staff stamp [era 3+ or prerelease-eligible]
        |
        |-- Winner tournament stamp (prerelease-eligible sets)
        |
        |-- WotC promos (basep):
        |   |-- W gold stamp (7 specific cards)
        |   +-- Black star promo
        |
        |-- Nintendo promos (np):
        |   +-- Black star promo
        |
        |-- DP-SM promos (dpp, hsp, bwp, xyp, smp):
        |   +-- Promo stamp
        |
        |-- SWSH/SV Promos (swshp/svp):
        |   |-- Modern promo pokeball
        |   |-- Build & Battle stamp
        |   +-- Pokemon Center stamp (svp only)
        |
        |-- Retailer exclusives:
        |   |-- Toys R Us stamp (XY/SM, eras 6-7)
        |   +-- Build-A-Bear stamp (XY/SM, eras 6-7)
        |
        |-- McDonald's sets (mcd*):
        |   +-- Confetti holo pattern
        |
        |-- Pokemon GO (pgo):
        |   +-- Peelable Ditto face icon
        |
        |-- League/tournament (era 3+):
        |   +-- League/Championship/Professor stamps + crosshatch holo
        |
        |-- Holo pattern analysis:
        |   |-- Holo finish (artwork shimmer, all eras)
        |   |-- Reverse holo (body shimmer, era 2+)
        |   +-- Cracked ice holo (theme deck, era 4+)

    Args:
        image_path: Path to the card image file.
        card_id: Full card identifier (e.g. "base1-4/holofoil").
        fast: If True, only run cheap pixel-based checks (world championship,
            shadowless).  Skip all OCR and pattern analysis.

    Returns:
        dict with keys:
            stamps_detected: list[str] -- detected stamp/variant types
            stamp_details: dict[str, dict] -- per-stamp detection details
            stamps_checked: list[str] -- all checks that were run
            variant_flags: dict -- high-level variant flags for pipeline use
    """
    set_id = _extract_set_id(card_id)
    era = _get_era(card_id)
    variant_suffix = card_id.split("/", 1)[1] if "/" in card_id else ""

    result: dict = {
        "stamps_detected": [],
        "stamp_details": {},
        "stamps_checked": [],
        "variant_flags": {},
    }

    # Load image once
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("detect_all_variants: could not read image: %s",
                       image_path)
        return result

    # ----- helper: run a single check and record results -----
    def _run(name: str, checker_fn, *args, **kwargs) -> dict | None:
        """Execute one detection check.  Records in result dict.

        Returns the raw detail dict if detected, else None.
        """
        result["stamps_checked"].append(name)
        try:
            detail = checker_fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Check %s failed for %s: %s", name, card_id, e)
            return None

        if not detail.get("detected"):
            logger.debug("Not detected: %s on %s", name, card_id)
            return None

        result["stamps_detected"].append(name)
        info: dict = {
            "confidence": detail["confidence"],
            "position": detail.get("position", "unknown"),
            "evidence": detail.get("evidence", ""),
        }
        # Preserve extra fields from specific detectors
        for key in ("variant", "retailer", "stamp_thickness",
                    "thickness_confidence", "holo_type", "warning",
                    "ink_color", "ink_color_confidence", "ocr_text",
                    "stamp_type", "all_stamps"):
            if key in detail:
                info[key] = detail[key]
        result["stamp_details"][name] = info
        logger.info("Detected: %s on %s (conf=%.2f, evidence=%s)",
                    name, card_id, detail["confidence"],
                    detail.get("evidence", ""))
        return detail

    # ===========================================================
    # TIER 0: Cheap pixel checks -- always run, even in fast mode
    # ===========================================================

    # World Championship deck (grey/silver borders) -- <1ms
    wc = _run("world_championship", _check_world_championship, img, set_id)
    if wc:
        result["variant_flags"]["is_reproduction"] = True
        # WC decks are reproductions with no collectible variants; stop here
        return result

    # Shadowless (base1 only) -- border gradient, <5ms
    if set_id == "base1":
        if _run("shadowless", _check_shadowless, img, set_id):
            result["variant_flags"]["shadowless"] = True

    # In fast mode, return after cheap pixel checks only
    if fast:
        return result

    # ===========================================================
    # TIER 1: WotC era stamp checks
    # ===========================================================

    # --- 1st Edition stamp (base1-base5, gym1-2, neo1-4) ---
    if set_id in _FIRST_EDITION_SETS:
        ed = _run("1st_edition", _check_1st_edition, img)

        # DINOv2 fallback: if OCR missed the stamp, try visual comparison
        if not ed and card_id:
            dino_result = _check_1st_edition_dino(img, card_id, set_id)
            if dino_result is not None:
                if dino_result["detected"]:
                    # DINOv2 detected a stamp that OCR missed
                    ed = dino_result
                    result["stamps_detected"].append("1st_edition")
                    result["stamp_details"]["1st_edition"] = {
                        "confidence": dino_result["confidence"],
                        "position": dino_result["position"],
                        "evidence": dino_result["evidence"],
                        "stamp_similarity": dino_result["stamp_similarity"],
                        "control_similarity": dino_result["control_similarity"],
                        "differential": dino_result["differential"],
                    }
                    logger.info(
                        "1st ed DINO fallback detected stamp on %s "
                        "(diff=%.4f, threshold=%.2f)",
                        card_id, dino_result["differential"],
                        dino_result["threshold"],
                    )
                else:
                    # Store DINO result for debugging even when not detected
                    result["stamp_details"].setdefault("1st_edition_dino", {
                        "differential": dino_result["differential"],
                        "stamp_similarity": dino_result["stamp_similarity"],
                        "control_similarity": dino_result["control_similarity"],
                    })

        if ed:
            result["variant_flags"]["1st_edition"] = True

            # Conditional sub-variants (only when 1st ed stamp found):

            # Grey stamp (faded ink, mostly Team Rocket era)
            regions = STAMP_REGIONS["1st_edition"]
            stamp_crop = _extract_region(img, *regions["tight"])
            ink_color, ink_conf = _check_grey_stamp(img, stamp_crop)
            ed_info = result["stamp_details"]["1st_edition"]
            ed_info["ink_color"] = ink_color
            ed_info["ink_color_confidence"] = ink_conf
            if ink_color == "grey":
                result["stamps_detected"].append("grey_stamp")
                result["stamp_details"]["grey_stamp"] = {
                    "confidence": ink_conf,
                    "position": ed_info.get("position", "left"),
                    "evidence": "ink_darkness_analysis",
                    "parent_stamp": "1st_edition",
                }
                result["variant_flags"]["grey_stamp"] = True

    # Ghost stamp: partial 1st ed impression (base1 only)
    if set_id == "base1":
        _run("ghost_stamp", _check_ghost_stamp, img, set_id)

    # Copyright year: 1999 vs 1999-2000 (base1 only)
    if set_id == "base1":
        cr = _run("copyright_year", _check_copyright_year, img, set_id)
        if cr and cr.get("variant") == "4th_print":
            result["variant_flags"]["4th_print"] = True

    # Jungle no-symbol error (base2 holos only)
    if set_id == "base2" and "holo" in variant_suffix.lower():
        if _run("no_symbol_error", _check_no_symbol_error_as_stamp,
                img, set_id, variant_suffix):
            result["variant_flags"]["no_symbol_error"] = True

    # ===========================================================
    # TIER 2: EX era stamp checks
    # ===========================================================

    # EX set logo stamp on reverse holos (ex7-ex16 only)
    if set_id in _EX_STAMPED_SETS:
        if _run("ex_set_stamp", _check_ex_set_stamp, img, set_id, card_id):
            result["variant_flags"]["ex_stamped_reverse"] = True

    # ===========================================================
    # TIER 3: Prerelease, Staff, and Winner stamps (era-gated)
    # ===========================================================

    prerelease_found = False

    # Prerelease text stamp: WotC/EX/DP sets
    if set_id in _PRERELEASE_TEXT_SETS:
        if _run("prerelease", _check_prerelease, img, set_id, era):
            prerelease_found = True
            result["variant_flags"]["prerelease"] = True

    # Prerelease logo stamp: HGSS+ sets
    elif set_id in _PRERELEASE_LOGO_SETS:
        if _run("prerelease", _check_prerelease, img, set_id, era):
            prerelease_found = True
            result["variant_flags"]["prerelease"] = True

    # Staff stamp: DP onward (era 3+) or prerelease-eligible WotC/EX sets
    if (era >= 3 or era == 0 or set_id in _PRERELEASE_TEXT_SETS
            or prerelease_found):
        if _run("staff_stamp", _check_staff_stamp, img, set_id, era):
            result["variant_flags"]["staff"] = True

    # Winner tournament stamp: prerelease-eligible + 1st ed sets
    if set_id in _WINNER_STAMP_SETS:
        if _run("winner_stamp", _check_winner_stamp, img, set_id, era):
            result["variant_flags"]["winner"] = True

    # ===========================================================
    # TIER 4: Promo set checks
    # ===========================================================

    # WotC Black Star Promos (basep)
    if set_id == "basep":
        _run("black_star_promo", _check_black_star_promo, img)

        # W gold stamp: extremely rare, only 7 eligible cards
        bare_id = card_id.split("/")[0]
        if bare_id in _W_STAMP_ELIGIBLE_CARDS:
            if _run("w_stamp", _check_w_stamp, img, set_id):
                result["variant_flags"]["w_stamp"] = True

    # Nintendo Black Star Promos (np, EX era)
    if set_id == "np":
        _run("black_star_promo", _check_black_star_promo, img)

    # DP-SM era promo sets (dpp, hsp, bwp, xyp, smp)
    if set_id in _PROMO_SETS:
        _run("promo_stamp", _check_promo_stamp, img)

    # SWSH/SV era modern promo sets (swshp, svp)
    if set_id in _MODERN_PROMO_SETS:
        _run("modern_promo", _check_modern_promo, img)

        # Build & Battle stamp (SWSH/SV promo cards from B&B boxes)
        _run("build_battle", _check_build_battle_stamp, img, set_id, era)

        # Pokemon Center exclusive stamp (svp only)
        if set_id == "svp":
            _run("pokemon_center", _check_pokemon_center_stamp,
                 img, set_id, era)

    # ===========================================================
    # TIER 5: Retailer exclusives
    # ===========================================================

    # Toys R Us stamp (XY/SM eras, 2016-2018)
    if era in _TOYS_R_US_ERAS or era == 0:
        if _run("toys_r_us", _toys_r_us_as_dict, img, set_id, era):
            result["variant_flags"]["toys_r_us"] = True

    # Build-A-Bear Workshop stamp (XY/SM eras, eras 6-7)
    if era in (6, 7) or era == 0:
        if _run("build_a_bear", _build_a_bear_as_dict, img, set_id, era):
            result["variant_flags"]["build_a_bear"] = True

    # ===========================================================
    # TIER 6: Special product variants
    # ===========================================================

    # McDonald's confetti holo
    if set_id in _MCDONALDS_SETS or set_id.startswith("mcd"):
        if _run("mcdonalds_holo", _check_mcdonalds_holo, img, set_id, era):
            result["variant_flags"]["mcdonalds_holo"] = True

    # Pokemon GO peelable Ditto icon
    if set_id == "pgo":
        if _run("peelable_ditto", _check_peelable_ditto, img, set_id):
            result["variant_flags"]["peelable_ditto"] = True

    # ===========================================================
    # TIER 7: League / tournament stamps (era 3+)
    # ===========================================================

    if era >= 3 or era == 0:
        if _run("league_stamps", _check_league_stamps, img, set_id, era):
            result["variant_flags"]["league_stamp"] = True

        # Crosshatch holo pattern (league/tournament promo exclusive)
        if _run("crosshatch_holo", _check_crosshatch_holo,
                img, set_id, era):
            result["variant_flags"]["crosshatch_holo"] = True

    # ===========================================================
    # TIER 8: Holo pattern analysis (most expensive, run last)
    # ===========================================================

    # Shared cache to avoid duplicate expensive analysis
    _holo_cache: dict = {}

    # --- Holo finish (artwork shimmer) -- all eras ---
    def _hf_check(img_bgr):
        if "hf" not in _holo_cache:
            finish, conf = _check_holo_finish(img_bgr, set_id, era)
            _holo_cache["hf"] = (finish, conf)
        finish, conf = _holo_cache["hf"]
        return {
            "detected": finish == "holofoil",
            "confidence": conf,
            "position": "artwork",
            "evidence": "holo_detector",
            "holo_type": finish,
        }

    if _run("holo_finish", _hf_check, img):
        result["variant_flags"]["holofoil"] = True

    # --- Reverse holo (body shimmer) -- era 2+ only ---
    if era >= _REVERSE_HOLO_MIN_ERA or era == 0:
        def _rh_check(img_bgr):
            if "rh" not in _holo_cache:
                label, conf = _check_reverse_holo(img_bgr, set_id, era)
                _holo_cache["rh"] = (label, conf)
            label, conf = _holo_cache["rh"]
            return {
                "detected": label == "reverse_holo",
                "confidence": conf,
                "position": "body",
                "evidence": "reverse_holo_detector",
                "holo_type": label,
            }

        if _run("reverse_holo", _rh_check, img):
            result["variant_flags"]["reverse_holofoil"] = True

    # --- Cracked ice holo (theme deck, era 4+) ---
    if era >= _CRACKED_ICE_MIN_ERA or era == 0:
        def _ci_check(img_bgr):
            detected, conf = _check_cracked_ice_holo(img_bgr, set_id, era)
            return {
                "detected": detected,
                "confidence": conf,
                "position": "artwork",
                "evidence": "cracked_ice_pattern",
            }

        if _run("cracked_ice_holo", _ci_check, img):
            result["variant_flags"]["cracked_ice_holo"] = True

    # ===========================================================
    # Summary
    # ===========================================================

    n_checked = len(result["stamps_checked"])
    n_found = len(result["stamps_detected"])
    logger.info(
        "detect_all_variants for %s (era=%d, set=%s): %d/%d detected [%s]",
        card_id, era, set_id, n_found, n_checked,
        ", ".join(result["stamps_detected"]) if n_found else "none",
    )

    return result
