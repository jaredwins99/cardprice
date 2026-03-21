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
# Individual stamp detection functions
# ---------------------------------------------------------------------------

def _check_1st_edition(img_bgr: np.ndarray) -> dict:
    """Check for 1st Edition stamp.

    Returns dict with 'detected', 'confidence', 'position', 'evidence'.
    """
    regions = STAMP_REGIONS["1st_edition"]

    # Wide region OCR
    wide = _extract_region(img_bgr, *regions["wide"])
    if wide.size == 0:
        return {"detected": False, "confidence": 0.0, "position": "left"}

    ocr_text = _ocr_region(wide)
    has_1st = "1st" in ocr_text
    has_edition = "edition" in ocr_text

    if has_1st and has_edition:
        return {
            "detected": True, "confidence": 0.95,
            "position": "left", "evidence": "ocr_both_tokens",
            "ocr_text": ocr_text,
        }
    if has_1st or has_edition:
        return {
            "detected": True, "confidence": 0.85,
            "position": "left", "evidence": "ocr_one_token",
            "ocr_text": ocr_text,
        }

    # Tight region OCR
    tight = _extract_region(img_bgr, *regions["tight"])
    if tight.size > 0:
        ocr_tight = _ocr_region(tight)
        has_1st_t = "1st" in ocr_tight
        has_edition_t = "edition" in ocr_tight

        if has_1st_t and has_edition_t:
            return {
                "detected": True, "confidence": 0.95,
                "position": "left", "evidence": "tight_ocr_both",
                "ocr_text": ocr_tight,
            }
        if has_1st_t or has_edition_t:
            return {
                "detected": True, "confidence": 0.85,
                "position": "left", "evidence": "tight_ocr_one",
                "ocr_text": ocr_tight,
            }
        combined_ocr = ocr_text + " " + ocr_tight
    else:
        combined_ocr = ocr_text

    # Circle detection + partial OCR evidence
    has_blob = _has_dark_circular_blob(wide)
    has_hough = _has_dark_circle_hough(tight if tight.size > 0 else wide)
    has_circle = has_blob or has_hough

    if has_circle and "1" in combined_ocr:
        method = "blob" if has_blob else "hough"
        return {
            "detected": True, "confidence": 0.70,
            "position": "left", "evidence": f"circle_{method}_plus_digit",
            "ocr_text": combined_ocr,
        }

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


# ---------------------------------------------------------------------------
# Determine which stamps to check for a given card_id
# ---------------------------------------------------------------------------

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

    return checks


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

    # Dispatch to appropriate checker
    _STAMP_CHECKERS = {
        "1st_edition": lambda img_bgr: _check_1st_edition(img_bgr),
        "ex_set_stamp": lambda img_bgr: _check_ex_set_stamp(img_bgr, set_id),
        "black_star_promo": lambda img_bgr: _check_black_star_promo(img_bgr),
        "modern_promo": lambda img_bgr: _check_modern_promo(img_bgr),
        "promo_stamp": lambda img_bgr: _check_promo_stamp(img_bgr),
        "copyright_year": lambda img_bgr: _check_copyright_year(img_bgr, set_id),
        "shadowless": lambda img_bgr: _check_shadowless(img_bgr, set_id),
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
