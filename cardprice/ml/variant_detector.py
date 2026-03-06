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

3. **1st Edition stamp** -- We look for the stamp using PaddleOCR on
   the expected stamp region (left side, just below the artwork frame).
   A contour-based circular blob check is used as supporting evidence
   but never triggers alone (requires OCR confirmation to avoid false
   positives from card artwork shadows).

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
# Era-to-variant constraint mapping
# ---------------------------------------------------------------------------
# Maps era numbers (from era_detector.py) to the set of valid variant strings
# that can appear in that era.  Variant strings match the card_id format used
# in dim_cards (e.g. "base1-4/holofoil").
#
# Era numbers:
#   1 = WotC Classic (1999-2003)   -- Base Set through Skyridge
#   2 = EX era (2003-2007)
#   3 = Diamond & Pearl (2007-2010)
#   4 = HeartGold SoulSilver (2010-2011)
#   5 = Black & White (2011-2013)
#   6 = XY (2014-2016)
#   7 = Sun & Moon (2017-2019)
#   8 = Sword & Shield (2020-2022)
#   9 = Scarlet & Violet (2023+)
#
# Variant key reference (7 TCGCSV subtypes mapped to card_id suffixes):
#   "normal"              -- flat print, no holo
#   "holofoil"            -- holo artwork only
#   "reverse_holofoil"    -- holo on border/text, not artwork
#   "1st_edition"         -- 1st Edition stamp, non-holo (Unlimited = "normal")
#   "1st_edition_holofoil"-- 1st Edition stamp + holo artwork
#   "unlimited"           -- explicitly Unlimited print (WotC only, = normal)
#   "unlimited_holofoil"  -- Unlimited holo (WotC only, = holofoil)
# ---------------------------------------------------------------------------

ERA_VALID_VARIANTS: dict[int, set[str]] = {
    # Era 1: WotC Classic (1999-2003)
    # Base Set through Neo Destiny had 1st Edition / Unlimited print runs.
    # Legendary Collection (base6) and e-Card series (ecard1-3) introduced
    # reverse holofoil but dropped 1st Edition.  We use a broad union here;
    # set-specific overrides in SET_SPECIAL_VARIANTS handle the details.
    1: {
        "normal",
        "holofoil",
        "reverse_holofoil",   # Legendary Collection + e-Card sets only
        "1st_edition",
        "1st_edition_holofoil",
        "unlimited",
        "unlimited_holofoil",
    },

    # Era 2: EX era (2003-2007)
    # No more 1st Edition.  Reverse holofoil standard from Ruby & Sapphire on.
    2: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 3: Diamond & Pearl (2007-2010)
    3: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 4: HeartGold SoulSilver (2010-2011)
    4: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 5: Black & White (2011-2013)
    5: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 6: XY (2014-2016)
    6: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 7: Sun & Moon (2017-2019)
    7: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 8: Sword & Shield (2020-2022)
    8: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },

    # Era 9: Scarlet & Violet (2023+)
    # Reverse holofoil rebranded as "cosmos holo" in some sets but same
    # pricing category.
    9: {
        "normal",
        "holofoil",
        "reverse_holofoil",
    },
}

# ---------------------------------------------------------------------------
# Set-specific variant overrides
# ---------------------------------------------------------------------------
# Some individual sets have unique variant patterns that differ from their
# era's defaults.  These override or extend ERA_VALID_VARIANTS for that set.
#
# Format: set_prefix -> dict with optional keys:
#   "valid":   set of valid variants (replaces the era default entirely)
#   "add":     set of extra variants to add to the era default
#   "remove":  set of variants to remove from the era default
#   "notes":   human-readable note about what's special
# ---------------------------------------------------------------------------

SET_SPECIAL_VARIANTS: dict[str, dict] = {
    # --- Era 1 WotC sets with 1st Edition ---
    # Base Set (base1): 1st Edition, Unlimited, AND Shadowless (unique).
    # Shadowless = Unlimited print run without drop shadow on card frame,
    # printed between 1st Edition and standard Unlimited runs.
    "base1": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
            "shadowless", "shadowless_holofoil",
        },
        "notes": "Only set with Shadowless variant (no drop shadow on frame).",
    },

    # Base Set 2 (base4): reprint set, no 1st Edition, no reverse holo.
    "base4": {
        "valid": {"normal", "holofoil"},
        "notes": "Reprint of Base/Jungle. Unlimited only, no 1st Edition.",
    },

    # Jungle (base2), Fossil (base3): 1st Edition + Unlimited, no reverse holo.
    "base2": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },
    "base3": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },

    # Team Rocket (base5): 1st Edition + Unlimited, no reverse holo.
    "base5": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },

    # Gym Heroes (gym1) and Gym Challenge (gym2): 1st Edition + Unlimited.
    "gym1": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },
    "gym2": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },

    # Neo Genesis (neo1) through Neo Destiny (neo4): last 1st Edition sets.
    "neo1": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },
    "neo2": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },
    "neo3": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
    },
    "neo4": {
        "valid": {
            "normal", "holofoil",
            "1st_edition", "1st_edition_holofoil",
            "unlimited", "unlimited_holofoil",
        },
        "notes": "Last set to have 1st Edition print run.",
    },

    # Legendary Collection (base6): first set with reverse holofoil.
    # Unique "fireworks" holographic pattern on reverse holo cards.
    # No 1st Edition (all Unlimited).
    "base6": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
        "notes": ("First reverse holofoil set. Unique 'fireworks' pattern "
                  "reverse holo (not the standard linear reverse holo)."),
    },

    # e-Card sets: Expedition (ecard1), Aquapolis (ecard2), Skyridge (ecard3).
    # Have reverse holofoil (introduced in LC), no 1st Edition.
    "ecard1": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
        "notes": "Expedition Base Set. Reverse holo has unique 'cosmic' pattern.",
    },
    "ecard2": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
    },
    "ecard3": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
    },

    # WotC Black Star Promos (basep): promos, no reverse holo or 1st ed.
    "basep": {
        "valid": {"normal", "holofoil"},
        "notes": "Wizard's Black Star Promos. Some are holo, most are normal.",
    },

    # Best of Game (bp): promo cards, holo only.
    "bp": {
        "valid": {"holofoil"},
        "notes": "Best of Game promo set. All cards are holofoil.",
    },

    # Southern Islands (si1): normal + some confetti holo.
    "si1": {
        "valid": {"normal", "holofoil"},
        "notes": "Southern Islands collection. Some cards have confetti holo.",
    },

    # --- Era 2 EX-era special sets ---
    "np": {
        "valid": {"normal", "holofoil"},
        "notes": "Nintendo Black Star Promos.",
    },

    # POP Series: normal/holo only, no reverse.
    "pop1": {"valid": {"normal", "holofoil"}},
    "pop2": {"valid": {"normal", "holofoil"}},
    "pop3": {"valid": {"normal", "holofoil"}},
    "pop4": {"valid": {"normal", "holofoil"}},
    "pop5": {"valid": {"normal", "holofoil"}},
    "pop6": {"valid": {"normal", "holofoil"}},
    "pop7": {"valid": {"normal", "holofoil"}},
    "pop8": {"valid": {"normal", "holofoil"}},
    "pop9": {"valid": {"normal", "holofoil"}},

    # Trainer Kits: normal only.
    "tk1a": {"valid": {"normal"}},
    "tk1b": {"valid": {"normal"}},
    "tk2a": {"valid": {"normal"}},
    "tk2b": {"valid": {"normal"}},

    # --- Era 3-4 promo/special sets ---
    "dpp": {"valid": {"normal", "holofoil"}},
    "hsp": {"valid": {"normal", "holofoil"}},
    "ru1": {
        "valid": {"normal"},
        "notes": "Pokemon Rumble promos. All normal with Rumble stamp.",
    },
    "col1": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
    },

    # --- Era 5 BW special sets ---
    "bwp": {"valid": {"normal", "holofoil"}},
    "dv1": {
        "valid": {"normal", "holofoil"},
        "notes": "Dragon Vault. All cards are holofoil.",
    },
    "dc1": {
        "valid": {"normal", "holofoil"},
        "notes": "Double Crisis. No reverse holofoil.",
    },

    # --- Era 6 XY promo/special sets ---
    "xyp": {"valid": {"normal", "holofoil"}},
    "xy0": {"valid": {"normal"}},
    "g1": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
        "notes": "Generations. Has radiant collection subset.",
    },

    # --- Era 7 SM special sets ---
    "smp": {"valid": {"normal", "holofoil"}},
    "sma": {
        "valid": {"normal", "holofoil"},
        "notes": "Hidden Fates Shiny Vault. All shiny/holo.",
    },
    "det1": {
        "valid": {"normal", "holofoil"},
        "notes": "Detective Pikachu. No reverse holofoil.",
    },
    "mcd18": {"valid": {"normal", "holofoil"}},
    "mcd19": {"valid": {"normal", "holofoil"}},

    # --- Era 8 SWSH special sets ---
    "swshp": {"valid": {"normal", "holofoil"}},
    "swsh35": {
        "valid": {"normal", "holofoil"},
        "notes": "Champion's Path. No reverse holofoil.",
    },
    "swsh45": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
    },
    "swsh45sv": {
        "valid": {"normal", "holofoil"},
        "notes": "Shining Fates Shiny Vault. All shiny/holo.",
    },
    "cel25": {
        "valid": {"normal", "holofoil"},
        "notes": "Celebrations. All cards are holofoil (cosmos holo pattern).",
    },
    "cel25c": {
        "valid": {"holofoil"},
        "notes": "Celebrations Classic Collection. All holofoil reprints.",
    },
    "pgo": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
        "notes": "Pokemon GO. Has peelable ditto cards.",
    },
    "fut20": {"valid": {"normal", "holofoil"}},
    "mcd21": {"valid": {"normal", "holofoil"}},
    "mcd22": {"valid": {"normal", "holofoil"}},

    # Trainer Gallery subsets: special art, no reverse holo.
    "swsh9tg":     {"valid": {"normal", "holofoil"}},
    "swsh10tg":    {"valid": {"normal", "holofoil"}},
    "swsh11tg":    {"valid": {"normal", "holofoil"}},
    "swsh12tg":    {"valid": {"normal", "holofoil"}},
    "swsh12pt5":   {"valid": {"normal", "holofoil", "reverse_holofoil"}},
    "swsh12pt5gg": {"valid": {"normal", "holofoil"}},

    # --- Era 9 SV special sets ---
    "svp": {"valid": {"normal", "holofoil"}},
    "sve": {
        "valid": {"normal"},
        "notes": "SV basic Energy cards.",
    },

    # McDonald's promos (various eras): normal + holo only.
    "mcd11": {"valid": {"normal", "holofoil"}},
    "mcd12": {"valid": {"normal", "holofoil"}},
    "mcd14": {"valid": {"normal", "holofoil"}},
    "mcd15": {"valid": {"normal", "holofoil"}},
    "mcd16": {"valid": {"normal", "holofoil"}},
    "mcd17": {"valid": {"normal", "holofoil"}},
}


def get_valid_variants(set_id: str, era: int = 0) -> set[str]:
    """Return the set of valid variant strings for a given set/era.

    Checks SET_SPECIAL_VARIANTS first for set-specific overrides, then
    falls back to ERA_VALID_VARIANTS.

    Args:
        set_id: Set prefix (e.g. "base1", "ex5", "sv3").
        era: Era number (1-9).  If 0, falls back to all common variants.

    Returns:
        Set of valid variant strings (e.g. {"normal", "holofoil"}).
    """
    if set_id in SET_SPECIAL_VARIANTS:
        spec = SET_SPECIAL_VARIANTS[set_id]
        if "valid" in spec:
            return set(spec["valid"])
        base = set(ERA_VALID_VARIANTS.get(era, {"normal", "holofoil"}))
        if "add" in spec:
            base |= spec["add"]
        if "remove" in spec:
            base -= spec["remove"]
        return base

    return set(ERA_VALID_VARIANTS.get(era, {"normal", "holofoil"}))


def is_valid_variant(set_id: str, era: int, variant: str) -> bool:
    """Check whether a variant is valid for the given set/era.

    Args:
        set_id: Set prefix (e.g. "base1").
        era: Era number (1-9).
        variant: Variant string (e.g. "reverse_holofoil").

    Returns:
        True if the variant is valid for this set/era.
    """
    return variant in get_valid_variants(set_id, era)


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


def _ocr_stamp_region(stamp_bgr: np.ndarray) -> str:
    """Run PaddleOCR on the stamp region and return concatenated lowercase text.

    Reuses the PaddleOCR TextDetection/TextRecognition singletons from
    ocr_matcher to avoid loading separate models.  Upscales small regions
    for better OCR accuracy.  Returns empty string on any failure.
    """
    try:
        import os
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        from cardprice.ml.ocr_matcher import _paddle_det, _paddle_rec
        import cardprice.ml.ocr_matcher as _ocr_mod

        # Initialize singletons if needed
        det = _paddle_det
        rec = _paddle_rec
        if det is None or rec is None:
            from paddleocr import TextDetection, TextRecognition
            if _ocr_mod._paddle_det is None:
                _ocr_mod._paddle_det = TextDetection(
                    model_name='PP-OCRv5_server_det')
            if _ocr_mod._paddle_rec is None:
                _ocr_mod._paddle_rec = TextRecognition(
                    model_name='en_PP-OCRv5_mobile_rec')
            det = _ocr_mod._paddle_det
            rec = _ocr_mod._paddle_rec

        # Upscale small regions -- PaddleOCR struggles below ~150px
        h, w = stamp_bgr.shape[:2]
        scale = max(1, 150 // max(h, 1))
        if scale > 1:
            stamp_up = cv2.resize(stamp_bgr, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
        else:
            stamp_up = stamp_bgr

        # Add padding so text isn't at the edge
        stamp_up = cv2.copyMakeBorder(stamp_up, 20, 20, 20, 20,
                                      cv2.BORDER_REPLICATE)

        # Upscale 3x for reliable detection on small stamp text
        stamp_up = cv2.resize(stamp_up, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)

        # Run detection
        det_results = list(det.predict(stamp_up))
        if not det_results or not det_results[0]:
            return ""

        det_out = det_results[0]
        polys = det_out.get('dt_polys', [])
        scores = det_out.get('dt_scores', [])

        texts = []
        for poly, det_score in zip(polys, scores):
            if det_score < 0.3:
                continue
            pts = np.array(poly, dtype=np.float32)
            x, y, bw, bh = cv2.boundingRect(pts)
            text_crop = stamp_up[max(0, y):y + bh, max(0, x):x + bw]
            if text_crop.size == 0:
                continue
            rec_results = list(rec.predict(text_crop))
            if rec_results and rec_results[0]:
                text = rec_results[0].get('rec_text', '').strip()
                conf = float(rec_results[0].get('rec_score', 0.0))
                if text and conf > 0.3:
                    texts.append(text)

        return " ".join(texts).lower()
    except Exception as e:
        logger.debug("PaddleOCR stamp check failed: %s", e)
        return ""


def _has_dark_circular_blob(stamp_bgr: np.ndarray) -> bool:
    """Check if the stamp region contains a dark circular blob consistent
    with the 1st Edition stamp shape.

    Uses stricter thresholds than a generic contour search:
    - Circularity >= 0.65 (real stamp is quite round)
    - Area between 3% and 30% of the region
    """
    try:
        gray = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        h_stamp, w_stamp = stamp_bgr.shape[:2]
        min_area = h_stamp * w_stamp * 0.03
        max_area = h_stamp * w_stamp * 0.30
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.65:
                    logger.debug("Dark circular blob found "
                                 "(area=%.0f, circ=%.2f)", area, circularity)
                    return True
    except Exception as e:
        logger.debug("Contour-based blob check failed: %s", e)
    return False


def _check_1st_edition(img_bgr: np.ndarray) -> bool:
    """Check for 1st Edition stamp using OCR and contour analysis.

    The 1st Edition stamp appears as a small black "1" inside a circle with
    "EDITION" text, located on the left side just below the artwork.

    Detection strategy:
    1. OCR the stamp region with PaddleOCR -- if "1st" or "edition" is found,
       return True immediately (high confidence).
    2. Look for a dark circular blob AND require at least partial OCR evidence
       (a "1" digit anywhere in the text) to confirm.  A blob alone is not
       sufficient -- too many false positives from card artwork and shadows.

    Returns True if a 1st Edition indicator is found.
    """
    stamp_region = _extract_region(img_bgr, STAMP_X0, STAMP_Y0,
                                   STAMP_X1, STAMP_Y1)
    if stamp_region.size == 0:
        return False

    # Strategy 1: OCR the stamp region
    ocr_text = _ocr_stamp_region(stamp_region)
    if "1st" in ocr_text or "edition" in ocr_text:
        logger.debug("1st Edition detected via OCR: %r", ocr_text)
        return True

    # Strategy 2: Dark circular blob + partial OCR evidence ("1" in text)
    if _has_dark_circular_blob(stamp_region) and "1" in ocr_text:
        logger.debug("1st Edition detected via blob + '1' in OCR: %r", ocr_text)
        return True

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
