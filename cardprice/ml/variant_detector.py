"""Detect Pokemon card variant from a phone photo.

Variant types:
  - normal:          Flat print, no holographic effects
  - holofoil:        Holographic pattern on the artwork area only
  - reverse_holofoil: Holographic pattern on everything EXCEPT the artwork
  - 1st_edition:     Has a "1st Edition" stamp (left side, below artwork)
  - full_art:        Artwork extends to card edges, no visible border
  - gold:            Gold/secret rare -- entire card dominated by gold hue
  - rainbow_rare:    Rainbow rare -- high saturation across multiple hue peaks

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

4. **Gold / secret rare detection** -- Gold cards have a distinctive gold
   (HSV hue ~15-45) color scheme dominating >40% of the card surface including
   the borders.  Rainbow rares have high saturation spread across 4+ of 6 hue
   segments.  Both are era-gated to era >= 7 (Sun & Moon, 2017+).

5. **Full art detection** -- Full art / alt art cards have artwork extending
   to the card edges with NO visible border.  Normal cards have a uniform
   yellow/silver/white border (low saturation, low hue variance) in the
   outer ~5% strip.  Full art cards show complex, colorful artwork in that
   same outer strip (high saturation, high hue variance).  We measure both
   metrics on the four edge strips and compare against thresholds.  Era-gated
   to eras 5+ (Black & White onward, 2011+) since full arts did not exist
   before then.

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
        "shadowless",          # Base Set (base1) only
        "shadowless_holofoil", # Base Set (base1) only
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
    # First era with full art cards (EX full arts).
    5: {
        "normal",
        "holofoil",
        "reverse_holofoil",
        "full_art",
    },

    # Era 6: XY (2014-2016)
    6: {
        "normal",
        "holofoil",
        "reverse_holofoil",
        "full_art",
    },

    # Era 7: Sun & Moon (2017-2019)
    # Gold/secret rares and rainbow rares introduced in this era.
    7: {
        "normal",
        "holofoil",
        "reverse_holofoil",
        "gold",
        "rainbow_rare",
        "full_art",
    },

    # Era 8: Sword & Shield (2020-2022)
    # Alt art / full art variants common (V, VMAX, VSTAR full arts).
    8: {
        "normal",
        "holofoil",
        "reverse_holofoil",
        "gold",
        "rainbow_rare",
        "full_art",
    },

    # Era 9: Scarlet & Violet (2023+)
    # Reverse holofoil rebranded as "cosmos holo" in some sets but same
    # pricing category.  Alt arts / illustration rares are full art.
    9: {
        "normal",
        "holofoil",
        "reverse_holofoil",
        "gold",
        "rainbow_rare",
        "full_art",
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
        "valid": {"normal", "holofoil", "promo"},
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
        "valid": {"normal", "holofoil", "promo"},
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
    "dpp": {"valid": {"normal", "holofoil", "promo"}},
    "hsp": {"valid": {"normal", "holofoil", "promo"}},
    "ru1": {
        "valid": {"normal"},
        "notes": "Pokemon Rumble promos. All normal with Rumble stamp.",
    },
    "col1": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
    },

    # --- Era 5 BW special sets ---
    "bwp": {"valid": {"normal", "holofoil", "promo"}},
    "dv1": {
        "valid": {"normal", "holofoil"},
        "notes": "Dragon Vault. All cards are holofoil.",
    },
    "dc1": {
        "valid": {"normal", "holofoil"},
        "notes": "Double Crisis. No reverse holofoil.",
    },

    # --- Era 6 XY promo/special sets ---
    "xyp": {"valid": {"normal", "holofoil", "promo"}},
    "xy0": {"valid": {"normal"}},
    "g1": {
        "valid": {"normal", "holofoil", "reverse_holofoil"},
        "notes": "Generations. Has radiant collection subset.",
    },

    # --- Era 7 SM special sets ---
    "smp": {"valid": {"normal", "holofoil", "promo"}},
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
    "swshp": {"valid": {"normal", "holofoil", "promo"}},
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
    "svp": {"valid": {"normal", "holofoil", "promo"}},
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

# ---------------------------------------------------------------------------
# TCGCSV subtype -> our variant mapping
# ---------------------------------------------------------------------------
# TCGCSV price data uses "subTypeName" to distinguish print variants within a
# single product.  Pokemon TCG (category 3) uses exactly 7 subtypes.  The
# remaining 21 subtypes belong to other TCGs (Flesh and Blood, MetaZoo,
# Digimon, etc.) and are mapped to None.
#
# Our variant system also includes visual-only variants that TCGCSV does NOT
# track as separate subtypes (they share the same product/subtype):
#   shadowless, shadowless_holofoil  -- Base Set only, detected visually
#   full_art                         -- BW+ era, detected visually
#   gold                             -- SM+ era, detected visually
#   rainbow_rare                     -- SM+ era, detected visually
#
# These visual variants require the detect_variant() CV pipeline below and
# cannot be derived from TCGCSV data alone.
# ---------------------------------------------------------------------------

TCGCSV_SUBTYPE_TO_VARIANT: dict[str, str | None] = {
    # --- Pokemon TCG subtypes (7) ---
    "Normal":               "normal",
    "Holofoil":             "holofoil",
    "Reverse Holofoil":     "reverse_holofoil",
    "1st Edition":          "1st_edition",
    "1st Edition Holofoil": "1st_edition_holofoil",
    "Unlimited":            "unlimited",
    "Unlimited Holofoil":   "unlimited_holofoil",

    # --- Non-Pokemon subtypes (mapped to None -- not applicable) ---
    "Foil":                          None,  # Flesh and Blood, Digimon, etc.
    "Cold Foil":                     None,  # Flesh and Blood
    "Rainbow Foil":                  None,  # Flesh and Blood
    "1st Edition Foil":              None,  # Flesh and Blood
    "1st Edition Cold Foil":         None,  # Flesh and Blood
    "1st Edition Normal":            None,  # Flesh and Blood
    "1st Edition Rainbow Foil":      None,  # Flesh and Blood
    "1st Wave Foil":                 None,  # Flesh and Blood
    "Unlimited Edition Foil":        None,  # Flesh and Blood
    "Unlimited Edition Normal":      None,  # Flesh and Blood
    "Unlimited Edition Rainbow Foil": None,  # Flesh and Blood
    "Holo":                          None,  # Digimon, other TCGs
    "Reverse Holo":                  None,  # Digimon, other TCGs
    "Holohex":                       None,  # MetaZoo
    "Parallel Foil":                 None,  # Digimon
    "Limited":                       None,  # Various non-Pokemon
    "Card and Die":                  None,  # Dice Masters, etc.
    "Card Only":                     None,  # Dice Masters, etc.
    "Die Only":                      None,  # Dice Masters, etc.
    "Metal":                         None,  # Metal card variants
    "Plastic":                       None,  # Plastic/oversized cards
}

# All valid variant suffixes used in our card_id system
ALL_VARIANTS = {
    "normal",
    "holofoil",
    "reverse_holofoil",
    "1st_edition",
    "1st_edition_holofoil",
    "unlimited",
    "unlimited_holofoil",
    "shadowless",
    "shadowless_holofoil",
    "full_art",
    "gold",
    "rainbow_rare",
    "promo",
}


def tcgcsv_subtype_to_variant(subtype_name: str) -> str | None:
    """Convert a TCGCSV subTypeName to our variant string.

    Returns None if the subtype is not a Pokemon TCG subtype.
    Raises KeyError if the subtype is completely unknown.
    """
    if subtype_name in TCGCSV_SUBTYPE_TO_VARIANT:
        return TCGCSV_SUBTYPE_TO_VARIANT[subtype_name]
    raise KeyError(f"Unknown TCGCSV subtype: {subtype_name!r}")


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

# 1st Edition stamp region (left side, just below artwork frame).
# On a 1008x1530 segment the stamp sits at approximately x=30-200, y=700-950.
# We use slightly wider margins to handle alignment/rotation variance.
STAMP_X0, STAMP_Y0, STAMP_X1, STAMP_Y1 = 0.02, 0.44, 0.24, 0.65

# Tighter stamp region focused on the expected stamp location.
# The stamp is a small ~30-40px circle at x: 5-12%, y: 55-65% of card.
STAMP_TIGHT_X0, STAMP_TIGHT_Y0 = 0.03, 0.53
STAMP_TIGHT_X1, STAMP_TIGHT_Y1 = 0.15, 0.67

# Sets that had 1st Edition print runs (Base Set through Neo Destiny).
# Used for era-gating: only check for 1st Edition stamp on these sets.
FIRST_EDITION_SETS = frozenset({
    "base1", "base2", "base3", "base5",  # Base, Jungle, Fossil, Team Rocket
    "gym1", "gym2",                        # Gym Heroes, Gym Challenge
    "neo1", "neo2", "neo3", "neo4",        # Neo Genesis through Neo Destiny
})

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

# ---------------------------------------------------------------------------
# Full art detection thresholds
# ---------------------------------------------------------------------------
# Width of the edge strip as a fraction of image dimensions.
# 5% captures the border area without reaching into the artwork on normal cards.
FULL_ART_EDGE_FRAC = 0.05

# Mean saturation of edge pixels.  Normal card borders (yellow/silver/white)
# have low saturation (typically 20-60).  Full art edges with colorful artwork
# extending to the border have higher saturation (typically 80+).
FULL_ART_MEAN_SAT_THRESHOLD = 65.0

# Standard deviation of hue across edge pixels.  Normal card borders have
# uniform hue (std < 15).  Full art edges with varied artwork have diverse
# hues (std > 20).
FULL_ART_HUE_STD_THRESHOLD = 18.0

# Fraction of edge pixels that must exceed MIN_SATURATION to be considered
# "colorful".  Normal borders: < 30% colorful.  Full art: > 40% colorful.
FULL_ART_COLORFUL_FRAC_THRESHOLD = 0.35

# Minimum number of the 4 edge strips that must pass the full art test.
# Requiring 3/4 prevents false positives from cards with one colorful edge
# (e.g., cards photographed on a colored surface with bleed).
FULL_ART_MIN_EDGES_PASSING = 3

# Eras where full art cards can exist (Black & White onward).
FULL_ART_MIN_ERA = 5

# ---------------------------------------------------------------------------
# Reverse holofoil detection thresholds
# ---------------------------------------------------------------------------
# Reverse holo cards have holographic foil on the BORDER and TEXT BOX, but
# NOT on the artwork.  In phone photos through binder sleeves this manifests
# as higher color variance (saturation std dev, hue std dev) in the border
# and text box regions compared to normal cards.
#
# We measure HSV saturation std dev and hue std dev in:
#   - Border strips (outer 6% on each side)
#   - Text box region (bottom 35-45% of card, excluding outer border)
#   - Artwork region (for comparison -- should be lower on reverse holo)
#
# Reverse holo exists from era 2 (EX era) onward, plus Legendary Collection
# and e-Card sets in era 1.  Era-gated accordingly.
# ---------------------------------------------------------------------------

# Minimum era for reverse holo (EX era onward).
# Era 1 sets base6/ecard1-3 also have reverse holo (handled via set override).
REVERSE_HOLO_MIN_ERA = 2

# Border strip width as fraction of card dimensions for reverse holo analysis.
REVERSE_HOLO_BORDER_FRAC = 0.06

# Text box region (below artwork, above bottom border)
REVERSE_HOLO_TEXT_Y0 = 0.58
REVERSE_HOLO_TEXT_Y1 = 0.92
REVERSE_HOLO_TEXT_X0 = 0.08
REVERSE_HOLO_TEXT_X1 = 0.92

# Saturation std dev threshold for border+text regions.
# Normal cards: border sat_std ~15-30 (uniform yellow/grey border).
# Reverse holo: border sat_std ~35-60+ (foil creates color patches).
REVERSE_HOLO_SAT_STD_THRESHOLD = 33.0

# Hue std dev threshold for border+text regions.
# Normal cards: border hue_std ~10-20 (uniform border color).
# Reverse holo: border hue_std ~25-45+ (foil refracts into rainbow patches).
REVERSE_HOLO_HUE_STD_THRESHOLD = 22.0

# Ratio: border/text variance must exceed artwork variance by this factor.
# This distinguishes reverse holo (foil on border, flat art) from regular holo
# (foil on art) or colorful normal cards.
REVERSE_HOLO_BORDER_ART_RATIO = 1.25

# Era 1 sets that have reverse holofoil (before EX era)
_ERA1_REVERSE_HOLO_SETS = {"base6", "ecard1", "ecard2", "ecard3"}


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


def _normalize_stamp_ocr(text: str) -> str:
    """Apply OCR confusion substitutions for 1st Edition stamp text.

    The stamp contains "1st" and "EDITION" in small, often low-contrast text.
    Common OCR misreads:
      - "1" read as "l" or "i" or "|"
      - "E" read as "F" or "C"
      - "D" read as "O" or "0"
      - "I" read as "l" or "1" or "|"
      - "N" read as "H" or "M"

    Returns the text with substitutions applied for matching.
    """
    # Work on lowercase
    t = text.lower()

    # Apply substitutions that help match "1st" and "edition"
    # For "1st": l->1, i->1, |->1
    # For "edition": common garbles
    subs = {
        "l": "1",  # lowercase L -> digit 1
        "|": "1",
    }
    normalized = ""
    for ch in t:
        normalized += subs.get(ch, ch)
    return normalized


def _ocr_stamp_region(stamp_bgr: np.ndarray) -> str:
    """Run PaddleOCR on the stamp region and return concatenated lowercase text.

    Reuses the PaddleOCR TextDetection/TextRecognition singletons from
    ocr_matcher to avoid loading separate models.  Upscales small regions
    for better OCR accuracy.  Returns empty string on any failure.
    """
    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        engine = get_rapid_engine()

        # Upscale small regions -- OCR struggles below ~150px
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

        result, _ = engine(stamp_up)
        if not result:
            return ""

        texts = []
        for box, text, conf in result:
            if text and float(conf) > 0.3:
                texts.append(text.strip())

        return " ".join(texts).lower()
    except Exception as e:
        logger.debug("RapidOCR stamp check failed: %s", e)
        return ""


def _ocr_stamp_region_binarized(stamp_bgr: np.ndarray) -> str:
    """Run OCR on a binarized (thresholded) version of the stamp region.

    The 1st Edition stamp is dark text/circle on the card background.
    Binarizing helps OCR read the stamp text more reliably, especially
    when the background is noisy or low-contrast.
    """
    try:
        from cardprice.ml.ocr_matcher import get_rapid_engine
        engine = get_rapid_engine()

        gray = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to handle varying background brightness
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

        # Convert back to BGR for OCR engine
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        h, w = binary_bgr.shape[:2]
        scale = max(1, 150 // max(h, 1))
        if scale > 1:
            binary_bgr = cv2.resize(binary_bgr, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)

        binary_bgr = cv2.copyMakeBorder(binary_bgr, 20, 20, 20, 20,
                                        cv2.BORDER_REPLICATE)
        binary_bgr = cv2.resize(binary_bgr, None, fx=3, fy=3,
                                interpolation=cv2.INTER_CUBIC)

        result, _ = engine(binary_bgr)
        if not result:
            return ""

        texts = []
        for box, text, conf in result:
            if text and float(conf) > 0.3:
                texts.append(text.strip())

        return " ".join(texts).lower()
    except Exception as e:
        logger.debug("RapidOCR binarized stamp check failed: %s", e)
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


def _has_dark_circle_hough(stamp_bgr: np.ndarray) -> bool:
    """Detect dark circles using HoughCircles on the stamp region.

    The 1st Edition stamp is a small black circle (~30-40px diameter on a
    1008x1530 segment).  HoughCircles is more robust to partial occlusion
    and noisy backgrounds than contour-based circularity.

    Returns True if a dark circle of the expected size is found.
    """
    try:
        gray = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2GRAY)
        h_stamp, w_stamp = stamp_bgr.shape[:2]

        # Blur to reduce noise before circle detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Expected radius: the stamp circle is ~3-8% of region width
        min_radius = max(3, int(w_stamp * 0.05))
        max_radius = max(10, int(w_stamp * 0.40))
        min_dist = max(5, int(w_stamp * 0.10))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=80,
            param2=25,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None:
            return False

        # Check if any detected circle is dark (low mean intensity inside)
        for circle in circles[0]:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
            # Create a circular mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val < 100:  # dark circle
                logger.debug(
                    "HoughCircles found dark circle at (%d,%d) r=%d, "
                    "mean_intensity=%.1f",
                    cx, cy, r, mean_val,
                )
                return True
    except Exception as e:
        logger.debug("HoughCircles check failed: %s", e)
    return False


def _check_full_art(img_bgr: np.ndarray, era: int = 0) -> tuple[bool, float]:
    """Check if a card is full art by analyzing the outer edge strips.

    Full art cards have artwork extending to the card edges -- the outer ~5%
    of the image contains colorful, varied artwork rather than a uniform
    yellow/silver/white border.

    We extract the four edge strips (top, bottom, left, right), convert to
    HSV, and measure:
      1. Mean saturation -- high for colorful artwork, low for plain borders.
      2. Hue standard deviation -- high for varied artwork, low for uniform borders.
      3. Fraction of colorful pixels -- what % of edge pixels are saturated.

    A strip "passes" if at least 2 of the 3 metrics exceed their thresholds.
    The card is classified as full art if >= 3 of the 4 strips pass.

    Args:
        img_bgr: Card image in BGR format.
        era: Era number (1-9).  Full art is era-gated to era >= 5 (BW+).
             If 0 (unknown), the check runs without era gating.

    Returns:
        (is_full_art, confidence) -- confidence based on edges passing:
          - 3/4 edges: 0.70
          - 4/4 edges: 0.90
          - 0-2 edges: 0.0 (not full art)
    """
    # Era gate: full art cards only exist from Black & White onward.
    if era != 0 and era < FULL_ART_MIN_ERA:
        logger.debug("Full art check skipped: era %d < %d", era, FULL_ART_MIN_ERA)
        return False, 0.0

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, 0.0

    edge_h = max(3, int(h * FULL_ART_EDGE_FRAC))
    edge_w = max(3, int(w * FULL_ART_EDGE_FRAC))

    # Extract the four edge strips
    strips = {
        "top":    img_bgr[:edge_h, :, :],
        "bottom": img_bgr[h - edge_h:, :, :],
        "left":   img_bgr[:, :edge_w, :],
        "right":  img_bgr[:, w - edge_w:, :],
    }

    passing = 0
    for name, strip in strips.items():
        if strip.size == 0:
            continue

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)
        v_chan = hsv[:, :, 2]

        # Metric 1: mean saturation
        mean_sat = float(np.mean(s_chan))

        # Metric 2: hue std (only among sufficiently bright pixels to
        # avoid dark corners skewing the result)
        bright_mask = v_chan >= MIN_VALUE
        if np.sum(bright_mask) > 10:
            hue_std = float(np.std(h_chan[bright_mask]))
        else:
            hue_std = 0.0

        # Metric 3: fraction of colorful pixels
        total_pixels = s_chan.size
        colorful_count = int(np.sum(s_chan >= MIN_SATURATION))
        colorful_frac = colorful_count / total_pixels if total_pixels > 0 else 0.0

        # A strip passes if at least 2 of 3 metrics exceed threshold
        signals = 0
        if mean_sat >= FULL_ART_MEAN_SAT_THRESHOLD:
            signals += 1
        if hue_std >= FULL_ART_HUE_STD_THRESHOLD:
            signals += 1
        if colorful_frac >= FULL_ART_COLORFUL_FRAC_THRESHOLD:
            signals += 1

        strip_pass = signals >= 2
        if strip_pass:
            passing += 1

        logger.debug(
            "Full art %s strip: mean_sat=%.1f, hue_std=%.1f, "
            "colorful_frac=%.2f, signals=%d, pass=%s",
            name, mean_sat, hue_std, colorful_frac, signals, strip_pass,
        )

    is_full_art = passing >= FULL_ART_MIN_EDGES_PASSING
    # Confidence: 3/4 edges = 0.70, 4/4 edges = 0.90
    if is_full_art:
        full_art_conf = 0.70 if passing == 3 else 0.90
    else:
        full_art_conf = 0.0
    logger.debug("Full art check: %d/%d edges passing (need %d), result=%s, conf=%.2f",
                 passing, len(strips), FULL_ART_MIN_EDGES_PASSING, is_full_art,
                 full_art_conf)
    return is_full_art, full_art_conf


def _check_reverse_holo(img_bgr: np.ndarray, era: int = 0,
                        set_id: str = "") -> tuple[bool, float]:
    """Detect reverse holofoil by comparing color variance in border/text vs artwork.

    Reverse holo cards have holographic foil on the BORDER and TEXT BOX regions
    but NOT on the artwork.  This produces higher saturation and hue variance
    in the border/text areas compared to the artwork area.

    The detection measures HSV saturation std dev and hue std dev in three
    regions:
      1. Border strips (outer 6% on each side)
      2. Text box (bottom 35-45% of card, inside borders)
      3. Artwork (center top portion, for comparison)

    The card is classified as reverse holo if the border+text regions show
    significantly higher variance than the artwork region.

    Args:
        img_bgr: Card image in BGR format.
        era: Era number (1-9).  Reverse holo exists from era 2+, plus
             select era 1 sets (base6, ecard1-3).  If 0, no era gating.
        set_id: Set prefix (e.g. "base6", "ex5").  Used for era 1 overrides.

    Returns:
        (is_reverse_holo, confidence) -- confidence based on how far the
        border vs artwork contrast ratio exceeds the threshold.
    """
    # Era gate: reverse holo only from era 2+, or specific era 1 sets
    if era != 0:
        if era < REVERSE_HOLO_MIN_ERA:
            if set_id not in _ERA1_REVERSE_HOLO_SETS:
                logger.debug(
                    "Reverse holo check skipped: era %d, set %s not eligible",
                    era, set_id,
                )
                return False, 0.0

    h_img, w_img = img_bgr.shape[:2]
    if h_img < 100 or w_img < 80:
        return False, 0.0

    # --- Extract regions ---
    bf = REVERSE_HOLO_BORDER_FRAC

    # Border strips (4 strips, combined into one pixel array)
    border_top = img_bgr[:int(h_img * bf), :, :]
    border_bottom = img_bgr[int(h_img * (1.0 - bf)):, :, :]
    border_left = img_bgr[int(h_img * bf):int(h_img * (1.0 - bf)),
                          :int(w_img * bf), :]
    border_right = img_bgr[int(h_img * bf):int(h_img * (1.0 - bf)),
                           int(w_img * (1.0 - bf)):, :]

    # Stack all border pixels for aggregate statistics
    border_pixels = []
    for strip in [border_top, border_bottom, border_left, border_right]:
        if strip.size > 0:
            border_pixels.append(strip.reshape(-1, 3))

    if not border_pixels:
        return False, 0.0
    border_all = np.vstack(border_pixels)

    # Text box region (below artwork, inside borders)
    text_region = _extract_region(
        img_bgr,
        REVERSE_HOLO_TEXT_X0, REVERSE_HOLO_TEXT_Y0,
        REVERSE_HOLO_TEXT_X1, REVERSE_HOLO_TEXT_Y1,
    )

    # Artwork region (center-top)
    art_region = _extract_region(img_bgr, ART_X0, ART_Y0, ART_X1, ART_Y1)

    if text_region.size == 0 or art_region.size == 0:
        return False, 0.0

    # --- Compute HSV statistics ---
    def _region_hsv_stats(region_bgr: np.ndarray) -> tuple[float, float]:
        """Return (saturation_std, hue_std) for bright pixels."""
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(np.float32).ravel()
        s_ch = hsv[:, :, 1].astype(np.float32).ravel()
        v_ch = hsv[:, :, 2].ravel()

        bright = v_ch >= MIN_VALUE
        if np.sum(bright) < 30:
            return 0.0, 0.0
        return float(np.std(s_ch[bright])), float(np.std(h_ch[bright]))

    def _pixels_hsv_stats(pixels_bgr: np.ndarray) -> tuple[float, float]:
        """Same as _region_hsv_stats but for an Nx3 pixel array."""
        if len(pixels_bgr) < 30:
            return 0.0, 0.0
        row_img = pixels_bgr.reshape(1, -1, 3)
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        h_ch = hsv[0, :, 0].astype(np.float32)
        s_ch = hsv[0, :, 1].astype(np.float32)
        v_ch = hsv[0, :, 2]

        bright = v_ch >= MIN_VALUE
        if np.sum(bright) < 30:
            return 0.0, 0.0
        return float(np.std(s_ch[bright])), float(np.std(h_ch[bright]))

    border_sat_std, border_hue_std = _pixels_hsv_stats(border_all)
    text_sat_std, text_hue_std = _region_hsv_stats(text_region)
    art_sat_std, art_hue_std = _region_hsv_stats(art_region)

    # Combined border+text score: average of border and text box variances
    combined_sat_std = (border_sat_std + text_sat_std) / 2.0
    combined_hue_std = (border_hue_std + text_hue_std) / 2.0

    logger.debug(
        "Reverse holo -- border: sat_std=%.1f hue_std=%.1f | "
        "text: sat_std=%.1f hue_std=%.1f | "
        "art: sat_std=%.1f hue_std=%.1f | "
        "combined: sat_std=%.1f hue_std=%.1f",
        border_sat_std, border_hue_std,
        text_sat_std, text_hue_std,
        art_sat_std, art_hue_std,
        combined_sat_std, combined_hue_std,
    )

    # --- Decision logic ---
    # Condition 1: combined border+text variance exceeds thresholds.
    # Require BOTH saturation and hue variance to be elevated -- using OR
    # causes too many false positives on binder scans where plastic sleeves
    # create high saturation variance without the rainbow hue diversity
    # characteristic of real reverse holofoil.
    sat_passes = combined_sat_std >= REVERSE_HOLO_SAT_STD_THRESHOLD
    hue_passes = combined_hue_std >= REVERSE_HOLO_HUE_STD_THRESHOLD

    if not (sat_passes and hue_passes):
        logger.debug("Reverse holo: sat_passes=%s, hue_passes=%s (need both)",
                     sat_passes, hue_passes)
        return False, 0.0

    # Condition 1b: the TEXT region specifically must show elevated hue variance.
    # Real reverse holos have foil on the text box, producing rainbow hue shifts.
    # Binder sleeve reflections affect the outer border but not the inner text
    # area.  Require text_hue_std >= 15 to confirm the signal isn't just border
    # contamination from the binder sleeve environment.
    if text_hue_std < 15.0:
        logger.debug("Reverse holo: text_hue_std=%.1f too low (need >= 15.0), "
                     "likely binder sleeve artifact", text_hue_std)
        return False, 0.0

    # Condition 2: border+text variance must be higher than artwork variance
    # (distinguishes reverse holo from regular holo or colorful normal cards).
    # Require BOTH ratios to exceed threshold -- a single elevated ratio
    # is not sufficient evidence.
    sat_ratio = combined_sat_std / max(art_sat_std, 1.0)
    hue_ratio = combined_hue_std / max(art_hue_std, 1.0)

    ratio_passes = (sat_ratio >= REVERSE_HOLO_BORDER_ART_RATIO
                    and hue_ratio >= REVERSE_HOLO_BORDER_ART_RATIO)

    logger.debug(
        "Reverse holo -- sat_ratio=%.2f, hue_ratio=%.2f, "
        "sat_passes=%s, hue_passes=%s, ratio_passes=%s",
        sat_ratio, hue_ratio, sat_passes, hue_passes, ratio_passes,
    )

    if not ratio_passes:
        logger.debug("Reverse holo: border/art ratio too low")
        return False, 0.0

    # Confidence based on how far above threshold the ratios are.
    max_ratio = max(sat_ratio, hue_ratio)
    rh_conf = min(0.95, 0.60 + (max_ratio - REVERSE_HOLO_BORDER_ART_RATIO) * 0.30)
    logger.debug("Reverse holo detected: combined_sat_std=%.1f, "
                 "combined_hue_std=%.1f, sat_ratio=%.2f, hue_ratio=%.2f, conf=%.2f",
                 combined_sat_std, combined_hue_std, sat_ratio, hue_ratio, rh_conf)
    return True, rh_conf


# ---------------------------------------------------------------------------
# Shadowless detection (Base Set only)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Unlimited Base Set cards have a visible drop shadow on the right and bottom
# edges of the artwork frame -- a 3-5px dark band between the art border and
# the yellow card border.  Shadowless cards lack this band entirely.
#
# Detection: extract a narrow vertical strip along the right edge of the
# artwork frame, compute the gradient magnitude in the blue channel (best
# contrast for the dark shadow against yellow border), and use row-by-row
# median filtering for robustness against noise.
#
# Measured reference values:
#   Unlimited:  combined_gradient 44-112  (strong dark band)
#   Shadowless: combined_gradient 0-15    (smooth transition)
#   Threshold:  > 30 = Unlimited (has shadow)
#
# Bottom edge gradient is used as a confirmation signal.
# ---------------------------------------------------------------------------

# Right-edge strip: narrow band straddling the art frame's right border
_SHADOW_RIGHT_X0 = 0.87   # just inside the art frame right edge
_SHADOW_RIGHT_X1 = 0.93   # just outside into the border
_SHADOW_RIGHT_Y0 = 0.15   # skip the top corner
_SHADOW_RIGHT_Y1 = 0.50   # stop before bottom of art

# Bottom-edge strip: narrow band along the art frame's bottom border
_SHADOW_BOTTOM_X0 = 0.15
_SHADOW_BOTTOM_X1 = 0.85
_SHADOW_BOTTOM_Y0 = 0.53  # just above art frame bottom
_SHADOW_BOTTOM_Y1 = 0.59  # just below into the border

# Threshold for shadow gradient magnitude
_SHADOW_GRADIENT_THRESHOLD = 30.0


def _edge_gradient_magnitude(strip_bgr: np.ndarray, axis: int = 1) -> float:
    """Compute median-of-row gradient magnitude in the blue channel.

    Args:
        strip_bgr: BGR image strip (narrow rectangle along an edge).
        axis: 1 for horizontal gradient (right edge), 0 for vertical (bottom).

    Returns:
        Median of per-row (or per-col) max gradient magnitudes.
        Higher values indicate a sharp dark-to-light transition (shadow).
    """
    if strip_bgr.size == 0:
        return 0.0

    # Blue channel has best contrast for dark shadow against yellow border
    blue = strip_bgr[:, :, 0].astype(np.float32)

    if axis == 1:
        # Horizontal gradient (right edge shadow runs vertically)
        grad = np.abs(np.diff(blue, axis=1))
        if grad.size == 0:
            return 0.0
        # Max gradient per row, then median across rows for robustness
        row_maxes = np.max(grad, axis=1)
        return float(np.median(row_maxes))
    else:
        # Vertical gradient (bottom edge shadow runs horizontally)
        grad = np.abs(np.diff(blue, axis=0))
        if grad.size == 0:
            return 0.0
        col_maxes = np.max(grad, axis=0)
        return float(np.median(col_maxes))


def _check_shadowless(img_bgr: np.ndarray) -> tuple[bool | None, float]:
    """Detect whether a Base Set card is Shadowless or Unlimited.

    Analyses the right and bottom edges of the artwork frame for the
    characteristic drop shadow present on Unlimited prints.

    Returns:
        (result, confidence) where result is:
          True  -- card is Shadowless (no shadow detected)
          False -- card is Unlimited (shadow detected)
          None  -- inconclusive (image too small or edge region invalid)

        Confidence is based on distance from the threshold:
          - At threshold boundary (combined ~30): confidence ~0.50
          - Far below threshold (combined ~0, clearly shadowless): ~0.90
          - Far above threshold (combined ~80+, clearly unlimited): ~0.90
    """
    h, w = img_bgr.shape[:2]
    if h < 100 or w < 100:
        logger.debug("Image too small for shadowless detection: %dx%d", w, h)
        return None, 0.0

    # Extract right-edge strip
    right_strip = _extract_region(
        img_bgr, _SHADOW_RIGHT_X0, _SHADOW_RIGHT_Y0,
        _SHADOW_RIGHT_X1, _SHADOW_RIGHT_Y1,
    )
    right_grad = _edge_gradient_magnitude(right_strip, axis=1)

    # Extract bottom-edge strip
    bottom_strip = _extract_region(
        img_bgr, _SHADOW_BOTTOM_X0, _SHADOW_BOTTOM_Y0,
        _SHADOW_BOTTOM_X1, _SHADOW_BOTTOM_Y1,
    )
    bottom_grad = _edge_gradient_magnitude(bottom_strip, axis=0)

    # Combined: weighted average (right is primary signal, bottom confirms)
    combined = (right_grad * 0.7) + (bottom_grad * 0.3)

    # Confidence: distance from threshold, normalized.
    # At threshold (30): 0.50.  At 0 or 60+: approaches 0.90+.
    distance_from_threshold = abs(combined - _SHADOW_GRADIENT_THRESHOLD)
    shadow_conf = min(0.95, 0.50 + (distance_from_threshold / _SHADOW_GRADIENT_THRESHOLD) * 0.40)

    logger.debug(
        "Shadow detection -- right_grad=%.1f, bottom_grad=%.1f, combined=%.1f "
        "(threshold=%.1f, conf=%.2f)",
        right_grad, bottom_grad, combined, _SHADOW_GRADIENT_THRESHOLD,
        shadow_conf,
    )

    if combined > _SHADOW_GRADIENT_THRESHOLD:
        logger.debug("Shadow detected -> Unlimited")
        return False, shadow_conf  # has shadow = Unlimited
    else:
        logger.debug("No shadow detected -> Shadowless")
        return True, shadow_conf   # no shadow = Shadowless


def _has_1st_edition_text(ocr_text: str) -> tuple[bool, bool]:
    """Check if OCR text contains "1st" and/or "edition" tokens.

    Applies OCR confusion normalization: "l" -> "1", "|" -> "1".
    Also checks for fuzzy variants of "edition" that OCR commonly produces:
      - "eomon", "edmon", "edtion", "editon", "ediion", "ed1t1on", etc.

    Returns:
        (has_1st, has_edition) booleans.
    """
    raw = ocr_text.lower()
    normalized = _normalize_stamp_ocr(raw)

    # Check for "1st" in both raw and normalized text
    has_1st = "1st" in raw or "1st" in normalized

    # Check for "edition" - exact match first, then fuzzy variants
    has_edition = "edition" in raw or "edition" in normalized

    if not has_edition:
        # Common OCR garbles of "EDITION" (7 chars):
        # The stamp text is small and often noisy, producing garbled output.
        # Use Levenshtein-like matching: any 5+ char substring within
        # edit distance 2 of "edition".
        import re
        # Look for words that are close to "edition" (at least 4 chars matching)
        words = re.findall(r'[a-z0-9]{4,}', normalized)
        for word in words:
            if _fuzzy_edition_match(word):
                has_edition = True
                break

    return has_1st, has_edition


def _fuzzy_edition_match(word: str) -> bool:
    """Check if a word is a fuzzy match for 'edition'.

    Uses edit distance with OCR confusion awareness.  Common OCR misreads
    of "EDITION": "eomoh", "eomon", "edmon", "editon", "edtion".

    The approach:
    1. Check against known garbled patterns (empirically observed).
    2. Use Levenshtein edit distance <= 3 as the threshold.
    """
    target = "edition"
    if len(word) < 4 or len(word) > 10:
        return False

    # Known OCR garbles of "EDITION" (empirically observed from stamp OCR)
    _EDITION_GARBLES = {
        "eomon", "eomoh", "edmon", "editon", "edtion", "ediion",
        "ed1t1on", "editi0n", "editio", "editan", "editin",
        "edltion", "ed1tion", "edrtion", "ednion",
    }
    if word in _EDITION_GARBLES:
        return True

    # Levenshtein edit distance
    m, n = len(word), len(target)
    if abs(m - n) > 3:
        return False

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if word[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] <= 3


def _check_1st_edition(img_bgr: np.ndarray) -> tuple[bool, float]:
    """Check for 1st Edition stamp using OCR, contour, and HoughCircles.

    The 1st Edition stamp appears as a small black circle containing "1"
    with "EDITION" text below, located on the left side just below artwork.

    Detection strategy (multi-signal):
    1. OCR the wide stamp region with PaddleOCR -- check for "1st" or
       "edition" with OCR confusion normalization (l->1, fuzzy "edition").
    2. OCR the tight stamp region (more focused) as a second pass.
    3. Binarized OCR as a third pass (helps with low-contrast stamps).
    4. Look for a dark circular blob (contour-based) AND/OR dark circle
       (HoughCircles) plus partial OCR evidence ("1" in text).
    5. A circle alone (no OCR evidence at all) is NOT sufficient -- too
       many false positives from card artwork and shadows.

    Returns:
        (detected, confidence) -- detected is True if 1st Edition found,
        confidence reflects evidence strength:
          - 0.95 if both "1st" AND "edition" found in OCR text
          - 0.85 if only "1st" OR "edition" found
          - 0.75 if circle + normalized "1st" or "edition" found
          - 0.70 if circle (contour or Hough) + partial "1" digit found
          - 0.0 if not detected
    """
    # --- Wide stamp region (original, catches off-center stamps) ---
    stamp_region = _extract_region(img_bgr, STAMP_X0, STAMP_Y0,
                                   STAMP_X1, STAMP_Y1)
    if stamp_region.size == 0:
        return False, 0.0

    # Strategy 1: OCR the wide stamp region (raw + normalized)
    ocr_text = _ocr_stamp_region(stamp_region)
    has_1st, has_edition = _has_1st_edition_text(ocr_text)

    if has_1st and has_edition:
        logger.debug("1st Edition detected via OCR (both tokens): %r", ocr_text)
        return True, 0.95
    if has_1st or has_edition:
        logger.debug("1st Edition detected via OCR (one token): %r", ocr_text)
        return True, 0.85

    # --- Tight stamp region (focused, better for small stamps) ---
    stamp_tight = _extract_region(img_bgr, STAMP_TIGHT_X0, STAMP_TIGHT_Y0,
                                  STAMP_TIGHT_X1, STAMP_TIGHT_Y1)
    if stamp_tight.size > 0:
        ocr_tight = _ocr_stamp_region(stamp_tight)
        has_1st_t, has_edition_t = _has_1st_edition_text(ocr_tight)

        if has_1st_t and has_edition_t:
            logger.debug("1st Edition detected via tight OCR (both): %r",
                         ocr_tight)
            return True, 0.95
        if has_1st_t or has_edition_t:
            logger.debug("1st Edition detected via tight OCR (one): %r",
                         ocr_tight)
            return True, 0.85

        # Merge OCR text from both regions for circle confirmation
        combined_ocr = ocr_text + " " + ocr_tight
    else:
        combined_ocr = ocr_text

    # Strategy 2: Binarized OCR (helps with low-contrast stamps on noisy bg)
    ocr_bin = _ocr_stamp_region_binarized(stamp_tight if stamp_tight.size > 0
                                          else stamp_region)
    if ocr_bin:
        has_1st_b, has_edition_b = _has_1st_edition_text(ocr_bin)
        if has_1st_b and has_edition_b:
            logger.debug("1st Edition detected via binarized OCR (both): %r",
                         ocr_bin)
            return True, 0.90
        if has_1st_b or has_edition_b:
            logger.debug("1st Edition detected via binarized OCR (one): %r",
                         ocr_bin)
            return True, 0.80
        combined_ocr = combined_ocr + " " + ocr_bin

    # Strategy 3: Circle detection + partial OCR evidence
    has_blob = _has_dark_circular_blob(stamp_region)
    has_hough = _has_dark_circle_hough(stamp_tight if stamp_tight.size > 0
                                       else stamp_region)
    has_circle = has_blob or has_hough

    if has_circle:
        # Check normalized combined OCR for "1st" or "edition"
        has_1st_c, has_edition_c = _has_1st_edition_text(combined_ocr)

        if has_1st_c or has_edition_c:
            method = "blob" if has_blob else "hough"
            logger.debug("1st Edition detected via %s + normalized OCR: %r",
                         method, combined_ocr)
            return True, 0.75

        # Fallback: circle + raw "1" digit in raw OCR text (NOT normalized,
        # because l->1 normalization creates too many false "1" matches
        # from random card text like "atrocla" -> "atroc1a").
        if "1" in combined_ocr:
            method = "blob" if has_blob else "hough"
            logger.debug("1st Edition detected via %s + '1' in raw OCR: %r",
                         method, combined_ocr)
            return True, 0.70

    return False, 0.0


# ---------------------------------------------------------------------------
# Public API: detect_first_edition
# ---------------------------------------------------------------------------

def detect_first_edition(image_path: str) -> tuple[bool, float]:
    """Detect if a card has a 1st Edition stamp.

    The 1st Edition stamp is a small black circle with "1" and "EDITION"
    text, located on the left side of the card between the artwork and the
    text box, roughly at x: 5-12%, y: 55-65% of card dimensions.  The stamp
    is approximately 30-40px diameter on a 1008x1530 segment.

    Detection uses multiple complementary signals with OCR confusion
    normalization (l->1 substitution, fuzzy "edition" matching):
      1. PaddleOCR on wide stamp region + confusion normalization
      2. PaddleOCR on tight stamp region + confusion normalization
      3. Binarized (thresholded) OCR for low-contrast stamps
      4. Contour-based dark circular blob + partial OCR evidence
      5. HoughCircles dark circle + partial OCR evidence

    A circle detection alone is NOT sufficient (too many false positives
    from card artwork and shadows).  OCR evidence is always required,
    even if just a partial "1" digit alongside a detected circle.

    Args:
        image_path: Path to the card image (phone photo or segment).

    Returns:
        (is_first_edition, confidence) where:
          - is_first_edition: True if a 1st Edition stamp was detected.
          - confidence: Detection confidence (0.0 to 0.95):
              0.95 -- both "1st" AND "edition" found in OCR
              0.85 -- only "1st" OR "edition" found
              0.90 -- both found via binarized OCR
              0.80 -- one found via binarized OCR
              0.75 -- circle + normalized "1st"/"edition"
              0.70 -- circle + partial "1" digit in OCR
              0.0  -- not detected
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return _check_1st_edition(img)


def _stamp_region_is_blank(img_bgr: np.ndarray) -> tuple[bool, float]:
    """Check if the 1st Edition stamp region is blank (no stamp present).

    Unlimited WotC-era cards have NO stamp in the region where 1st Edition
    cards have the "1" circle + "EDITION" text.  This function checks if
    the tight stamp region is relatively uniform (no dark circular features,
    low contrast, no significant dark blobs).

    Returns:
        (is_blank, confidence) -- is_blank True means no stamp detected
        (consistent with Unlimited).  confidence 0.0-0.95.
    """
    stamp_tight = _extract_region(img_bgr, STAMP_TIGHT_X0, STAMP_TIGHT_Y0,
                                  STAMP_TIGHT_X1, STAMP_TIGHT_Y1)
    if stamp_tight.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(stamp_tight, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Check for dark circular features (which would indicate a stamp)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.count_nonzero(dark_mask) / (h * w)

    # Check contrast (standard deviation of intensity)
    std_val = gray.std()

    # A blank stamp region has:
    # - Very few dark pixels (< 5% of area)
    # - Low contrast (std < 30)
    # A region with a stamp has:
    # - Significant dark area (the circle is ~10-20% of tight region)
    # - Higher contrast (dark circle on lighter background)

    if dark_ratio < 0.05 and std_val < 30:
        conf = min(0.90, 0.60 + (1.0 - dark_ratio) * 0.20 + (30 - std_val) / 30 * 0.10)
        logger.debug("Stamp region blank: dark_ratio=%.3f, std=%.1f -> Unlimited (conf=%.2f)",
                     dark_ratio, std_val, conf)
        return True, conf

    # Moderate evidence: low dark ratio OR low contrast
    if dark_ratio < 0.10 and std_val < 40:
        conf = 0.55
        logger.debug("Stamp region likely blank: dark_ratio=%.3f, std=%.1f (conf=%.2f)",
                     dark_ratio, std_val, conf)
        return True, conf

    logger.debug("Stamp region NOT blank: dark_ratio=%.3f, std=%.1f",
                 dark_ratio, std_val)
    return False, 0.0


def detect_edition_status(image_path: str) -> dict:
    """Detect whether a WotC-era card is 1st Edition or Unlimited.

    Combines 1st Edition stamp detection with blank-region detection
    to provide a definitive edition classification.

    Args:
        image_path: Path to the card image.

    Returns:
        Dict with:
          - "edition": "1st_edition", "unlimited", or "unknown"
          - "confidence": float 0.0-0.95
          - "has_stamp": bool -- True if 1st Edition stamp detected
          - "stamp_blank": bool -- True if stamp region appears blank
          - "details": str -- human-readable explanation
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    is_1st, stamp_conf = _check_1st_edition(img)
    is_blank, blank_conf = _stamp_region_is_blank(img)

    if is_1st:
        return {
            "edition": "1st_edition",
            "confidence": stamp_conf,
            "has_stamp": True,
            "stamp_blank": False,
            "details": f"1st Edition stamp detected (conf={stamp_conf:.2f})",
        }

    if is_blank:
        return {
            "edition": "unlimited",
            "confidence": blank_conf,
            "has_stamp": False,
            "stamp_blank": True,
            "details": f"No stamp in expected region (conf={blank_conf:.2f})",
        }

    return {
        "edition": "unknown",
        "confidence": 0.0,
        "has_stamp": False,
        "stamp_blank": False,
        "details": "Could not determine edition status",
    }


# ---------------------------------------------------------------------------
# EX-era stamped reverse holo detection
# ---------------------------------------------------------------------------
# EX Team Rocket Returns (ex7) through EX Power Keepers (ex16) reverse holo
# cards have the set logo stamped in the bottom-right area of the artwork.
# The stamp is semi-transparent with the set name text.
#
# Detection: OCR the artwork bottom-right region for known set name text.
# The stamp text is typically the set name in small caps, e.g. "DELTA SPECIES",
# "POWER KEEPERS", etc.  We also check for a higher-than-expected text density
# in that region compared to normal artwork.
# ---------------------------------------------------------------------------

# Sets that have stamped reverse holos (EX Team Rocket Returns through Power Keepers)
STAMPED_SETS = frozenset({
    "ex7", "ex8", "ex9", "ex10", "ex11", "ex12", "ex13", "ex14", "ex15", "ex16",
})

# Known stamp text fragments for each set (lowercase, for fuzzy matching)
STAMPED_SET_TEXT: dict[str, list[str]] = {
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

# All known stamp words across all sets (for set-agnostic detection)
_ALL_STAMP_WORDS = frozenset({
    "team", "rocket", "returns", "deoxys", "emerald", "unseen", "forces",
    "delta", "species", "legend", "maker", "holon", "phantoms", "crystal",
    "guardians", "dragon", "frontiers", "power", "keepers",
})

# Stamp region: bottom-right quadrant of the artwork area.
# The stamp sits roughly at x: 55-88%, y: 35-55% of card dimensions.
STAMP_ART_X0, STAMP_ART_Y0 = 0.50, 0.30
STAMP_ART_X1, STAMP_ART_Y1 = 0.90, 0.58


def _check_stamped(img_bgr: np.ndarray,
                   set_id: str = "") -> tuple[bool, float]:
    """Check for EX-era set logo stamp on a reverse holo card.

    The stamp is a semi-transparent set logo/text overlaid on the card art,
    in the bottom-right area of the artwork window.

    Detection strategy:
    1. Extract the stamp region (bottom-right of artwork).
    2. Run OCR on the region.
    3. Check if any known stamp text fragments match the OCR output.
    4. If set_id is known, check for that set's specific stamp text.

    Args:
        img_bgr: Card image in BGR format.
        set_id: Set prefix (e.g. "ex11").  If provided, matches against
            that set's stamp text.  If empty, matches against all known stamps.

    Returns:
        (is_stamped, confidence) where:
          - is_stamped: True if a stamp was detected.
          - confidence: 0.90 if set-specific text found, 0.75 if generic
            stamp text found, 0.0 if not detected.
    """
    stamp_region = _extract_region(img_bgr, STAMP_ART_X0, STAMP_ART_Y0,
                                   STAMP_ART_X1, STAMP_ART_Y1)
    if stamp_region.size == 0:
        return False, 0.0

    # OCR the stamp region
    ocr_text = _ocr_stamp_region(stamp_region)
    if not ocr_text:
        return False, 0.0

    logger.debug("Stamped OCR text: %r (set=%s)", ocr_text, set_id)

    # Check for set-specific stamp text
    if set_id and set_id in STAMPED_SET_TEXT:
        expected_words = STAMPED_SET_TEXT[set_id]
        matches = sum(1 for w in expected_words if w in ocr_text)
        if matches >= 1:
            logger.debug("Stamped: matched %d/%d words for %s",
                         matches, len(expected_words), set_id)
            conf = 0.90 if matches >= 2 else 0.80
            return True, conf

    # Check for any known stamp text (set-agnostic)
    found_words = [w for w in _ALL_STAMP_WORDS if w in ocr_text]
    if len(found_words) >= 2:
        logger.debug("Stamped: found generic stamp words: %s", found_words)
        return True, 0.75
    elif len(found_words) == 1:
        # Single word match -- too low confidence for standalone detection
        # but could be combined with other signals
        logger.debug("Stamped: single word match: %s (too weak alone)", found_words)
        return False, 0.0

    return False, 0.0


def detect_stamped(image_path: str | Path,
                   set_id: str = "") -> tuple[bool, float]:
    """Detect if a card has an EX-era set logo stamp on the artwork.

    These stamped cards appear in EX Team Rocket Returns (ex7) through
    EX Power Keepers (ex16).  The reverse holo cards in these sets have
    the set logo/name stamped semi-transparently in the bottom-right
    area of the card artwork.

    Args:
        image_path: Path to the card image.
        set_id: Set prefix (e.g. "ex11" for Delta Species).  If provided,
            detection looks for that set's specific stamp text.

    Returns:
        (is_stamped, confidence) where:
          - is_stamped: True if stamp detected.
          - confidence: 0.80-0.90 for set-specific match,
                       0.75 for generic match, 0.0 if not detected.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Only check on sets that actually had stamps
    if set_id and set_id not in STAMPED_SETS:
        return False, 0.0

    return _check_stamped(img, set_id=set_id)


# ---------------------------------------------------------------------------
# Promo stamp detection
# ---------------------------------------------------------------------------
# Promo cards across all eras have a distinctive black star symbol that
# replaces the normal set symbol.  The star shape has very low "solidity"
# (ratio of contour area to convex hull area) because the points of the
# star create concavities.  This is the primary discriminator:
#
#   Promo star: solidity ~0.25-0.40, circularity ~0.07-0.18
#   Normal set symbols: solidity ~0.55-0.99, circularity ~0.10-0.73
#
# Position varies by era:
#   WotC (era 1):        right side, near set symbol area (x:76-98%, y:44-60%)
#   EX-XY (eras 2-6):   bottom-right corner (x:82-99%, y:91-99%)
#   SM-SV (eras 7-9):   bottom-left corner (x:2-22%, y:88-97%)
#
# For WotC promos, we also look for "PROMO" text via OCR as backup.
# ---------------------------------------------------------------------------

# Promo card set prefixes across all eras.
PROMO_SETS = frozenset({
    "basep",   # WotC Black Star Promos
    "np",      # Nintendo (EX-era) Black Star Promos
    "dpp",     # Diamond & Pearl Promos
    "hsp",     # HGSS Promos
    "bwp",     # Black & White Promos
    "xyp",     # XY Promos
    "smp",     # Sun & Moon Promos
    "swshp",   # Sword & Shield Promos
    "svp",     # Scarlet & Violet Promos
})

# Regions to check for the promo star by position group.
# Format: (x0, y0, x1, y1) as fractions of card dimensions.
_PROMO_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "left":  (0.02, 0.88, 0.22, 0.97),   # SM/SWSH/SV promos
    "right": (0.82, 0.91, 0.99, 0.99),   # EX/DP/HGSS/BW/XY promos
    "wotc":  (0.76, 0.44, 0.98, 0.60),   # WotC promos (near set symbol area)
}

# Era -> which region(s) to check for the promo star.
_ERA_PROMO_REGION: dict[int, list[str]] = {
    1: ["wotc"],       # WotC era
    2: ["right"],      # EX era
    3: ["right"],      # DP era
    4: ["right"],      # HGSS era
    5: ["right"],      # BW era
    6: ["right"],      # XY era
    7: ["left"],       # SM era
    8: ["left"],       # SWSH era
    9: ["left"],       # SV era
}

# Solidity threshold: promo stars have very low solidity due to concavities
# between the star points.  Normal set symbols are much more solid.
_PROMO_STAR_MAX_SOLIDITY = 0.45

# Area bounds (at 3x upscale) for a promo star contour.
_PROMO_STAR_MIN_AREA = 80
_PROMO_STAR_MAX_AREA = 500

# Upscale factor for promo stamp region analysis.
_PROMO_UPSCALE = 3

# Darkness threshold for binarization when looking for the black star.
_PROMO_DARK_THRESHOLD = 100


def _has_promo_star(region_bgr: np.ndarray) -> tuple[bool, float, dict]:
    """Check if a region contains a black star shape (promo stamp).

    The promo star has a distinctive shape with low solidity (~0.25-0.40)
    because the concavities between star points reduce the area relative
    to the convex hull.  Normal set symbols have solidity > 0.55.

    Args:
        region_bgr: BGR image of the region to check.

    Returns:
        (found, confidence, details) where:
          - found: True if a promo-star-shaped dark blob was detected.
          - confidence: 0.0-0.95 based on how well the shape matches.
          - details: dict with "area", "solidity", "circularity" of best match.
    """
    if region_bgr.size == 0:
        return False, 0.0, {}

    # Upscale for better contour detection on small reference images
    region_up = cv2.resize(region_bgr, None,
                           fx=_PROMO_UPSCALE, fy=_PROMO_UPSCALE,
                           interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(region_up, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, _PROMO_DARK_THRESHOLD, 255,
                                 cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_match: tuple[float, float, float, float] | None = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _PROMO_STAR_MIN_AREA or area > _PROMO_STAR_MAX_AREA:
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

        if solidity >= _PROMO_STAR_MAX_SOLIDITY:
            continue  # Too solid -- not a star shape

        # Confidence based on how star-like the shape is:
        # - Very low solidity (0.25-0.35) = high confidence
        # - Moderate solidity (0.35-0.45) = lower confidence
        if solidity < 0.35:
            conf = 0.90
        elif solidity < 0.40:
            conf = 0.80
        else:
            conf = 0.65

        if best_match is None or conf > best_match[3]:
            best_match = (solidity, circularity, area, conf)

    if best_match is not None:
        sol, circ, area, conf = best_match
        logger.debug("Promo star detected: solidity=%.3f, circularity=%.3f, "
                     "area=%.0f, confidence=%.2f", sol, circ, area, conf)
        return True, conf, {
            "solidity": round(sol, 3),
            "circularity": round(circ, 3),
            "area": round(area, 0),
        }

    return False, 0.0, {}


def _check_promo_stamp(img_bgr: np.ndarray,
                       era: int = 0,
                       set_id: str = "") -> tuple[bool, float, str | None]:
    """Check if a card image has a promo stamp (black star symbol).

    Checks the appropriate region based on era.  When era is unknown (0),
    checks all possible regions.  For WotC-era promos, also attempts OCR
    for "PROMO" text as a backup signal.

    Args:
        img_bgr: Card image in BGR format.
        era: Era number (1-9).  0 = unknown.
        set_id: Set prefix (e.g. "svp", "basep").

    Returns:
        (is_promo, confidence, stamp_position) where:
          - is_promo: True if promo stamp detected.
          - confidence: 0.0-0.95.
          - stamp_position: "left", "right", or "wotc" indicating where the
            stamp was found.  None if not detected.
    """
    # Determine which regions to check
    if era > 0 and era in _ERA_PROMO_REGION:
        regions_to_check = _ERA_PROMO_REGION[era]
    elif set_id in PROMO_SETS:
        # Infer era from promo set prefix
        _set_era_map = {
            "basep": 1, "np": 2, "dpp": 3, "hsp": 4,
            "bwp": 5, "xyp": 6, "smp": 7, "swshp": 8, "svp": 9,
        }
        inferred_era = _set_era_map.get(set_id, 0)
        regions_to_check = _ERA_PROMO_REGION.get(inferred_era, ["left", "right"])
    else:
        # Unknown era/set: check all regions
        regions_to_check = ["left", "right", "wotc"]

    best_result: tuple[bool, float, str | None] = (False, 0.0, None)

    for region_key in regions_to_check:
        x0, y0, x1, y1 = _PROMO_REGIONS[region_key]
        region = _extract_region(img_bgr, x0, y0, x1, y1)

        found, conf, _details = _has_promo_star(region)
        if found and conf > best_result[1]:
            best_result = (True, conf, region_key)

    # WotC backup: try OCR for "PROMO" text in the bottom-right area
    if not best_result[0] and (era == 1 or set_id == "basep" or era == 0):
        wotc_region = _extract_region(img_bgr, 0.60, 0.82, 0.99, 0.98)
        ocr_text = _ocr_stamp_region(wotc_region)
        if "promo" in ocr_text:
            logger.debug("WotC promo detected via OCR: %r", ocr_text)
            best_result = (True, 0.85, "wotc")

    return best_result


def detect_promo_stamp(image_path: str | Path,
                       set_id: str | None = None,
                       era: int = 0) -> dict:
    """Detect if a card has a promo stamp (black star promo symbol).

    Modern promo cards have a distinctive black star symbol that replaces
    the normal set symbol.  This function checks for that star shape using
    contour analysis of dark blobs in the expected stamp region.

    The stamp position varies by era:
      - WotC (era 1): Mid-right, near set symbol area + "PROMO" text
      - EX through XY (eras 2-6): Bottom-right corner
      - SM through SV (eras 7-9): Bottom-left corner

    Args:
        image_path: Path to the card image.
        set_id: Optional set prefix (e.g. "svp", "basep").  Helps determine
            which region to check.
        era: Era number (1-9).  0 = unknown (checks all regions).

    Returns:
        Dict with keys:
          - "is_promo": bool -- True if promo stamp detected.
          - "confidence": float -- 0.0-0.95.
          - "stamp_position": "left" | "right" | "wotc" | None -- where the
            stamp was found relative to the card.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    effective_set = set_id or ""
    is_promo, conf, position = _check_promo_stamp(img, era=era, set_id=effective_set)

    if is_promo:
        logger.info("Detected promo stamp for %s (conf=%.2f, position=%s)",
                    image_path, conf, position)

    return {
        "is_promo": is_promo,
        "confidence": round(conf, 2),
        "stamp_position": position,
    }


# ---------------------------------------------------------------------------
# Gold / Secret Rare and Rainbow Rare detection
# ---------------------------------------------------------------------------
# Gold/secret rare cards have a distinctive gold color scheme that dominates
# the entire card (border, text areas, and often artwork background).  The
# gold hue sits in the ~15-45 range in OpenCV HSV (yellow-gold).
#
# Rainbow rare cards have high saturation across multiple distinct hue peaks,
# producing a visible rainbow gradient across the card surface.
#
# Both types only exist in modern eras (Sun & Moon onwards, era >= 7).
# ---------------------------------------------------------------------------

# Gold hue range in OpenCV HSV (0-180 scale)
GOLD_HUE_LOW = 15
GOLD_HUE_HIGH = 45

# Minimum fraction of card pixels in the gold hue range to classify as gold.
# Gold cards have the gold tint across the majority of the card surface.
GOLD_COVERAGE_THRESHOLD = 0.40

# Minimum saturation for gold pixels (gold is saturated, not washed out)
GOLD_MIN_SATURATION = 40

# Minimum brightness for gold pixels (gold is bright, not dark)
GOLD_MIN_VALUE = 80

# Minimum gold coverage on the border strips specifically.  Gold cards have
# distinctively gold borders unlike any normal card.
GOLD_BORDER_THRESHOLD = 0.50

# Rainbow rare: minimum number of distinct hue segments (out of 6) with
# significant pixel count.  Rainbow rares span most of the hue wheel.
RAINBOW_MIN_PEAKS = 4

# Rainbow rare: minimum fraction of pixels at high saturation
RAINBOW_MIN_SAT_COVERAGE = 0.35

# Minimum era for gold/rainbow rare detection
GOLD_RAINBOW_MIN_ERA = 7


def _check_gold_rare(img_bgr: np.ndarray, era: int = 0) -> tuple[str | None, float]:
    """Detect gold/secret rare or rainbow rare cards from HSV color analysis.

    Gold cards have a dominant yellow-gold hue (HSV hue ~15-45) across the
    majority of the card surface, including borders and text areas.

    Rainbow rare cards have high saturation spread across many different hues,
    producing a visible rainbow gradient.

    Args:
        img_bgr: Card image in BGR format.
        era: Era number.  Gold/rainbow rares only exist in era >= 7.
             If era < 7 and era != 0, returns None immediately.

    Returns:
        (variant, confidence) -- variant is "gold", "rainbow_rare", or None.
        Confidence for gold is based on gold pixel coverage (higher = more confident).
        Confidence for rainbow is based on active hue segment count.
    """
    # Only applicable for modern era cards (Sun & Moon onwards)
    if era != 0 and era < GOLD_RAINBOW_MIN_ERA:
        return None, 0.0

    h_img, w_img = img_bgr.shape[:2]
    if h_img < 50 or w_img < 50:
        return None, 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    total_pixels = h_img * w_img

    # --- Gold detection ---
    # Find pixels with gold hue + sufficient saturation + brightness
    gold_mask = (
        (h_chan >= GOLD_HUE_LOW) & (h_chan <= GOLD_HUE_HIGH)
        & (s_chan >= GOLD_MIN_SATURATION)
        & (v_chan >= GOLD_MIN_VALUE)
    )
    gold_count = int(np.count_nonzero(gold_mask))
    gold_coverage = gold_count / total_pixels

    # Check the border strips specifically -- gold cards have gold borders
    # which is very distinctive (normal cards have grey/white/yellow borders
    # but at low saturation).
    border_regions_hsv = [
        _extract_region(hsv, 0.0, 0.0, 1.0, 0.08),    # top
        _extract_region(hsv, 0.0, 0.92, 1.0, 1.0),     # bottom
        _extract_region(hsv, 0.0, 0.08, 0.08, 0.92),    # left
        _extract_region(hsv, 0.92, 0.08, 1.0, 0.92),    # right
    ]

    border_gold_pixels = 0
    border_total_pixels = 0
    for region in border_regions_hsv:
        if region.size == 0:
            continue
        bh, bs, bv = region[:, :, 0], region[:, :, 1], region[:, :, 2]
        b_gold = (
            (bh >= GOLD_HUE_LOW) & (bh <= GOLD_HUE_HIGH)
            & (bs >= GOLD_MIN_SATURATION)
            & (bv >= GOLD_MIN_VALUE)
        )
        border_gold_pixels += int(np.count_nonzero(b_gold))
        border_total_pixels += region.shape[0] * region.shape[1]

    border_gold_coverage = (
        border_gold_pixels / border_total_pixels
        if border_total_pixels > 0 else 0.0
    )

    logger.debug(
        "Gold check: overall_coverage=%.3f, border_coverage=%.3f",
        gold_coverage, border_gold_coverage,
    )

    # Gold card: high gold coverage overall AND strong gold borders
    if (gold_coverage >= GOLD_COVERAGE_THRESHOLD
            and border_gold_coverage >= GOLD_BORDER_THRESHOLD):
        # Confidence scales with how far above thresholds the coverage is.
        # At threshold (0.40 overall, 0.50 border) = 0.70 confidence.
        # At 0.60 overall + 0.70 border = ~0.90 confidence.
        gold_excess = ((gold_coverage - GOLD_COVERAGE_THRESHOLD) / 0.30
                       + (border_gold_coverage - GOLD_BORDER_THRESHOLD) / 0.30) / 2.0
        gold_conf = min(0.95, 0.70 + gold_excess * 0.25)
        logger.info(
            "Detected gold/secret rare (coverage=%.2f, border=%.2f, conf=%.2f)",
            gold_coverage, border_gold_coverage, gold_conf,
        )
        return "gold", gold_conf

    # --- Rainbow rare detection ---
    # Rainbow rares have high saturation across multiple distinct hue ranges.
    # We divide the hue space into 6 segments of 30 degrees each and check
    # how many segments have significant representation among saturated pixels.
    sat_mask = (s_chan >= 60) & (v_chan >= MIN_VALUE)
    sat_pixels = h_chan[sat_mask]
    sat_coverage = len(sat_pixels) / total_pixels

    if sat_coverage >= RAINBOW_MIN_SAT_COVERAGE and len(sat_pixels) >= 100:
        # 6 hue segments: 0-30, 30-60, 60-90, 90-120, 120-150, 150-180
        segment_counts = np.zeros(6, dtype=int)
        for i in range(6):
            lo = i * 30
            hi = (i + 1) * 30
            segment_counts[i] = int(np.count_nonzero(
                (sat_pixels >= lo) & (sat_pixels < hi)
            ))

        # A segment is "present" if it holds >= 5% of the saturated pixels
        min_segment_pixels = len(sat_pixels) * 0.05
        active_segments = int(np.sum(segment_counts >= min_segment_pixels))

        logger.debug(
            "Rainbow check: sat_coverage=%.3f, active_segments=%d, "
            "segment_counts=%s",
            sat_coverage, active_segments, segment_counts.tolist(),
        )

        if active_segments >= RAINBOW_MIN_PEAKS:
            # Confidence: 4 segments = 0.70, 5 = 0.82, 6 = 0.92
            rainbow_conf = min(0.95, 0.50 + active_segments * 0.07
                               + sat_coverage * 0.20)
            logger.info(
                "Detected rainbow rare (sat_coverage=%.2f, segments=%d, conf=%.2f)",
                sat_coverage, active_segments, rainbow_conf,
            )
            return "rainbow_rare", rainbow_conf

    return None, 0.0


def detect_variant(image_path: str | Path, era: int = 0,
                   card_id: str | None = None) -> str:
    """Detect the variant of a Pokemon card from a photo.

    Args:
        image_path: Path to the card image (phone photo or scan).
        era: Era number (1-9) for era-gated checks.  0 = unknown (no gating).
        card_id: Optional card identifier (e.g. "base1-4").  When the card
            belongs to base1, shadowless detection is enabled.

    Returns:
        One of: "normal", "holofoil", "reverse_holofoil", "1st_edition",
        "promo", "full_art", "shadowless", "gold", "rainbow_rare".

    Detection priority:
      1. 1st Edition stamp (highest priority -- overrides all others).
      1b. EX-era stamped reverse holo (ex7-ex16, returns "reverse_holofoil").
      1c. Promo stamp (black star symbol on promo set cards).
      2. Gold / rainbow rare (era >= 7 only, checked early since gold
         cards also trigger holo/full-art detectors).
      3. Shadowless (base1 only -- right/bottom edge gradient analysis).
      4. Full art (artwork extends to card edges, no border).  Era-gated
         to era >= 5 (Black & White, 2011+).
      5. Reverse holofoil (border/text variance analysis).  Era-gated to
         era >= 2 (EX era+), plus base6/ecard sets in era 1.
      6. Holographic analysis (holofoil vs reverse_holofoil vs normal).
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    logger.debug("Analyzing variant for %s (shape=%s, era=%d, card_id=%s)",
                 image_path, img.shape, era, card_id)

    # --- Derive set prefix for era/set-specific gating ---
    set_prefix = (card_id or "").split("-")[0] if card_id else ""

    # --- 1st Edition check (highest priority, era-gated) ---
    # Only check for 1st Edition stamp on known WotC sets that actually had
    # 1st Edition print runs.  When era/card_id is unknown (era=0, no card_id),
    # require high-confidence OCR (>= 0.85) to avoid false positives from
    # random "1" digits in attack text or HP values.
    if set_prefix in FIRST_EDITION_SETS:
        stamp_detected, stamp_conf = _check_1st_edition(img)
        if stamp_detected:
            logger.info("Detected variant: 1st_edition for %s (conf=%.2f)",
                        image_path, stamp_conf)
            return "1st_edition"
    elif not set_prefix:
        # Unknown card: only trust high-confidence OCR detections
        stamp_detected, stamp_conf = _check_1st_edition(img)
        if stamp_detected and stamp_conf >= 0.85:
            logger.info("Detected variant: 1st_edition for %s (conf=%.2f, unknown card)",
                        image_path, stamp_conf)
            return "1st_edition"

    # --- EX-era stamped reverse holo check (ex7-ex16 only) ---
    # Stamped cards have a set logo overlaid on the artwork. This is a
    # sub-type of reverse_holofoil but visually distinct.  We detect
    # the stamp text via OCR on the artwork bottom-right region.
    # Note: for pricing purposes, stamped = reverse_holofoil (same TCGCSV
    # subtype), but we report it for collection tracking.
    if set_prefix in STAMPED_SETS:
        stamped_detected, stamped_conf = _check_stamped(img, set_id=set_prefix)
        if stamped_detected:
            logger.info("Detected EX-era stamp for %s (conf=%.2f), "
                        "reporting as reverse_holofoil",
                        image_path, stamped_conf)
            # Stamped reverse holos are priced as reverse_holofoil
            return "reverse_holofoil"

    # --- Promo stamp check (all eras) ---
    # Promo cards have a black star symbol replacing the normal set symbol.
    # Only check when the card is from a known promo set, or when the set
    # is unknown (era=0).
    if set_prefix in PROMO_SETS:
        promo_detected, promo_conf, promo_pos = _check_promo_stamp(
            img, era=era, set_id=set_prefix)
        if promo_detected:
            logger.info("Detected variant: promo for %s (conf=%.2f, pos=%s)",
                        image_path, promo_conf, promo_pos)
            return "promo"
    elif not set_prefix and era == 0:
        # Unknown card: check all regions but require high confidence
        promo_detected, promo_conf, promo_pos = _check_promo_stamp(
            img, era=0, set_id="")
        if promo_detected and promo_conf >= 0.85:
            logger.info("Detected variant: promo for %s (conf=%.2f, pos=%s, "
                        "unknown card)", image_path, promo_conf, promo_pos)
            return "promo"

    # --- Gold / Rainbow Rare check (era >= 7 only) ---
    # When era is unknown (0), skip gold/rainbow detection entirely.
    # These are rare variants that produce many false positives on binder
    # scans due to warm lighting and color casts.  Only check when we
    # have confirmed era context.
    if era >= GOLD_RAINBOW_MIN_ERA:
        gold_variant, gold_conf = _check_gold_rare(img, era)
        if gold_variant is not None:
            logger.info("Detected variant: %s for %s (conf=%.2f)",
                        gold_variant, image_path, gold_conf)
            return gold_variant

    # --- Shadowless check (base1 only, after 1st edition) ---
    # 1st Edition Base Set cards are never Shadowless (they predate it),
    # so this runs only after the 1st edition check passes.
    if set_prefix == "base1":
        shadowless_result, shadow_conf = _check_shadowless(img)
        if shadowless_result is True:
            logger.info("Detected variant: shadowless for %s (conf=%.2f)",
                        image_path, shadow_conf)
            return "shadowless"
        # shadowless_result is False (Unlimited) or None (inconclusive):
        # fall through to holo analysis which will return normal/holofoil

    # --- Full art check (before holo -- full art cards often trigger holo) ---
    # When era is unknown (0), skip full art detection.  Full art cards only
    # exist from Black & White onward (era >= 5) and the edge-strip analysis
    # produces many false positives on binder scans (warm lighting causes
    # high saturation in the border areas).
    if era >= FULL_ART_MIN_ERA:
        fa_detected, fa_conf = _check_full_art(img, era=era)
        if fa_detected:
            logger.info("Detected variant: full_art for %s (conf=%.2f)",
                        image_path, fa_conf)
            return "full_art"

    # --- Reverse holo check (after full art, before general holo analysis) ---
    # Reverse holo has foil on border/text but NOT artwork.  Check this before
    # the general holo analysis so we can give a definitive answer when the
    # border/text variance signal is strong.
    # When era is unknown (0), still run reverse holo but require the era
    # to be at least 2 (EX era+) or a known era 1 reverse holo set.
    # With era=0, we run the check but the _check_reverse_holo function
    # itself does not era-gate when era=0, so we add extra caution here.
    if era >= REVERSE_HOLO_MIN_ERA or (era == 1 and set_prefix in _ERA1_REVERSE_HOLO_SETS):
        rh_detected, rh_conf = _check_reverse_holo(img, era=era, set_id=set_prefix)
        if rh_detected:
            logger.info("Detected variant: reverse_holofoil for %s (conf=%.2f)",
                        image_path, rh_conf)
            return "reverse_holofoil"
    elif era == 0 and not set_prefix:
        # Unknown era: still run reverse holo but require higher confidence
        rh_detected, rh_conf = _check_reverse_holo(img, era=0, set_id="")
        if rh_detected and rh_conf >= 0.75:
            logger.info("Detected variant: reverse_holofoil for %s (conf=%.2f, unknown era)",
                        image_path, rh_conf)
            return "reverse_holofoil"

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


def detect_variant_detailed(image_path: str | Path, era: int = 0,
                           card_id: str | None = None) -> dict:
    """Like detect_variant() but returns detailed analysis for debugging.

    Args:
        image_path: Path to the card image.
        era: Era number (1-9) for era-gated checks.  0 = unknown.

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
      - is_full_art: bool
      - is_reverse_holo: bool
      - gold_rare_result: str | None -- "gold", "rainbow_rare", or None
      - is_shadowless: bool | None  (only for base1 cards)
      - shadow_right_grad: float | None
      - shadow_bottom_grad: float | None
      - shadow_combined: float | None
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Era-gated 1st Edition check: only on WotC sets that had 1st Ed runs,
    # or high-confidence OCR when card is unknown.
    set_prefix = (card_id or "").split("-")[0] if card_id else ""
    if set_prefix in FIRST_EDITION_SETS:
        has_stamp, stamp_conf = _check_1st_edition(img)
    elif not set_prefix:
        has_stamp, stamp_conf = _check_1st_edition(img)
        if has_stamp and stamp_conf < 0.85:
            has_stamp = False  # Reject low-confidence detections on unknown cards
    else:
        has_stamp, stamp_conf = False, 0.0

    # EX-era stamped check
    if set_prefix in STAMPED_SETS:
        has_ex_stamp, ex_stamp_conf = _check_stamped(img, set_id=set_prefix)
    else:
        has_ex_stamp, ex_stamp_conf = False, 0.0

    # Promo stamp check
    if set_prefix in PROMO_SETS:
        is_promo, promo_conf, promo_position = _check_promo_stamp(
            img, era=era, set_id=set_prefix)
    elif not set_prefix and era == 0:
        is_promo, promo_conf, promo_position = _check_promo_stamp(
            img, era=0, set_id="")
        if is_promo and promo_conf < 0.85:
            is_promo = False
    else:
        is_promo, promo_conf, promo_position = False, 0.0, None

    # Gold/rainbow: only when era is known and >= 7
    if era >= GOLD_RAINBOW_MIN_ERA:
        gold_rare_result, gold_conf = _check_gold_rare(img, era)
    else:
        gold_rare_result, gold_conf = None, 0.0

    # Full art: only when era is known and >= 5
    if era >= FULL_ART_MIN_ERA:
        is_full_art, full_art_conf = _check_full_art(img, era=era)
    else:
        is_full_art, full_art_conf = False, 0.0

    # Reverse holo: era-gated, or high-confidence when era unknown
    if era >= REVERSE_HOLO_MIN_ERA or (era == 1 and set_prefix in _ERA1_REVERSE_HOLO_SETS):
        is_reverse_holo, rh_conf = _check_reverse_holo(img, era=era, set_id=set_prefix)
    elif era == 0:
        is_reverse_holo, rh_conf = _check_reverse_holo(img, era=0, set_id="")
        if is_reverse_holo and rh_conf < 0.75:
            is_reverse_holo = False
    else:
        is_reverse_holo, rh_conf = False, 0.0

    # Shadowless analysis (base1 only)
    is_shadowless = None
    shadow_right_grad = None
    shadow_bottom_grad = None
    shadow_combined = None
    if set_prefix == "base1" and not has_stamp:
        h, w = img.shape[:2]
        if h >= 100 and w >= 100:
            right_strip = _extract_region(
                img, _SHADOW_RIGHT_X0, _SHADOW_RIGHT_Y0,
                _SHADOW_RIGHT_X1, _SHADOW_RIGHT_Y1,
            )
            shadow_right_grad = _edge_gradient_magnitude(right_strip, axis=1)
            bottom_strip = _extract_region(
                img, _SHADOW_BOTTOM_X0, _SHADOW_BOTTOM_Y0,
                _SHADOW_BOTTOM_X1, _SHADOW_BOTTOM_Y1,
            )
            shadow_bottom_grad = _edge_gradient_magnitude(bottom_strip, axis=0)
            shadow_combined = (shadow_right_grad * 0.7) + (shadow_bottom_grad * 0.3)
            is_shadowless = shadow_combined <= _SHADOW_GRADIENT_THRESHOLD

    art_region = _extract_region(img, ART_X0, ART_Y0, ART_X1, ART_Y1)
    border_region = _extract_region(img, 0.05, BORDER_Y0, 0.95, 0.95)

    art_combined, art_spread, art_noise = _holo_score(art_region)
    border_combined, border_spread, border_noise = _holo_score(border_region)

    art_sat = _saturation_std(art_region)
    border_sat = _saturation_std(border_region)

    # Determine variant using same logic as detect_variant
    # Priority: 1st Ed > Stamped > Promo > Gold/Rainbow > Shadowless > Full Art > Reverse Holo > Holo
    if has_stamp:
        variant = "1st_edition"
    elif has_ex_stamp:
        variant = "reverse_holofoil"  # stamped is a sub-type of reverse_holofoil for pricing
    elif is_promo:
        variant = "promo"
    elif gold_rare_result is not None:
        variant = gold_rare_result
    elif is_shadowless is True:
        variant = "shadowless"
    elif is_full_art:
        variant = "full_art"
    elif is_reverse_holo:
        variant = "reverse_holofoil"
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
        "stamp_confidence": round(stamp_conf, 2),
        "has_ex_era_stamp": has_ex_stamp,
        "ex_stamp_confidence": round(ex_stamp_conf, 2),
        "is_promo": is_promo,
        "promo_confidence": round(promo_conf, 2),
        "promo_position": promo_position,
        "is_full_art": is_full_art,
        "is_reverse_holo": is_reverse_holo,
        "gold_rare_result": gold_rare_result,
        "is_shadowless": is_shadowless,
        "shadow_right_grad": (round(shadow_right_grad, 2)
                              if shadow_right_grad is not None else None),
        "shadow_bottom_grad": (round(shadow_bottom_grad, 2)
                               if shadow_bottom_grad is not None else None),
        "shadow_combined": (round(shadow_combined, 2)
                            if shadow_combined is not None else None),
    }
