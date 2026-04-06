"""Era-specific stamp/prerelease region lookup.

Provides normalized crop coordinates for stamp detection based on card era
and set. The pipeline uses this to know exactly WHERE to crop for each card
before running stamp classifiers or OCR.

Regions are specified as fractional coordinates [0.0, 1.0] of the card
image dimensions:
    x_start, x_end: fraction of card width  (0 = left edge)
    y_start, y_end: fraction of card height  (0 = top edge)

Verified against real binder scans:
    - Skitty (ex14 Crystal Guardians stamp)
    - Vibrava (ex15 Dragon Frontiers stamp)
    - Chikorita (ex11 Delta Species stamp)
    - Misty's Seadra (gym2 prerelease stamp)
    - Aerodactyl (base3 prerelease stamp)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stamp region definitions
# ---------------------------------------------------------------------------

STAMP_REGIONS: dict[str, dict] = {
    "ex_era_stamp": {
        # EX Team Rocket Returns (ex7) through EX Power Keepers (ex16).
        # Set logo stamp overlaid on bottom-right of artwork area.
        # Verified: Skitty/Crystal Guardians, Vibrava/Dragon Frontiers,
        #           Chikorita/Delta Species.
        "x_start": 0.53,
        "x_end": 0.93,
        "y_start": 0.33,
        "y_end": 0.60,
        "description": "Bottom-right of artwork area, set logo stamp",
    },
    "wotc_prerelease": {
        # WotC-era prerelease stamps (Gym Heroes/Challenge, Fossil, etc.).
        # "PRERELEASE" text in bottom-right of artwork, just above the
        # Pokemon species/length/weight info line.
        # Verified: Misty's Seadra (gym2), Aerodactyl (base3).
        "x_start": 0.50,
        "x_end": 0.93,
        "y_start": 0.33,
        "y_end": 0.58,
        "description": "Bottom-right of artwork area, PRERELEASE text",
    },
    "wotc_1st_edition": {
        # 1st Edition stamp: small black circle with "1" and "EDITION" text.
        # Located on the left side, just below the artwork frame.
        # Matches variant_tree.json region [0.02, 0.44, 0.24, 0.65].
        "x_start": 0.02,
        "x_end": 0.24,
        "y_start": 0.44,
        "y_end": 0.65,
        "description": "Left side below artwork, 1st Edition circle stamp",
    },
    "wotc_1st_edition_tight": {
        # Tighter region focused on the expected 1st Edition stamp location.
        # Better for small-stamp detection when card alignment is good.
        "x_start": 0.03,
        "x_end": 0.15,
        "y_start": 0.53,
        "y_end": 0.67,
        "description": "Tight crop around 1st Edition stamp",
    },
}

# ---------------------------------------------------------------------------
# Set-to-region mapping
# ---------------------------------------------------------------------------

# EX-era sets with set logo stamps on reverse holos (ex7-ex16)
_STAMPED_SETS = frozenset([
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
])

# WotC sets that had prerelease promos
_PRERELEASE_SETS = frozenset([
    "base1", "base2", "base3", "base4", "base5",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
    "ecard1", "ecard2", "ecard3",
    # EX-era prerelease promos
    "ex1", "ex2", "ex3", "ex4", "ex5", "ex6",
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
    # DP-era prerelease promos
    "dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7",
    "pl1", "pl2", "pl3", "pl4",
])

# 1st Edition eligible sets
_FIRST_EDITION_SETS = frozenset([
    "base1", "base2", "base3", "base5",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
])


def _extract_set_id(card_id: str) -> str:
    """Extract set prefix from card_id like 'base1-4/holofoil' -> 'base1'."""
    bare = card_id.split("/")[0]
    return bare.rsplit("-", 1)[0] if "-" in bare else bare


def get_stamp_region(
    card_id: str = "",
    set_id: str = "",
    stamp_type: str = "",
) -> Optional[dict]:
    """Return the crop region to check for stamps based on card era/set.

    Priority:
      1. If stamp_type is specified, return that region directly.
      2. If set_id is in the stamped EX sets (ex7-ex16), return ex_era_stamp.
      3. For 1st Edition eligible sets, return wotc_1st_edition.
      4. For prerelease-eligible sets, return wotc_prerelease.
      5. None if no stamp region applies.

    Args:
        card_id: Full card identifier (e.g. "ex14-3/normal"). Used to
            extract set_id if set_id is not provided.
        set_id: Set identifier (e.g. "ex14"). Takes priority over card_id.
        stamp_type: Explicit stamp type key (e.g. "ex_era_stamp").

    Returns:
        Dict with x_start, x_end, y_start, y_end, description.
        None if no stamp region applies.
    """
    # Explicit stamp type override
    if stamp_type:
        region = STAMP_REGIONS.get(stamp_type)
        if region is None:
            logger.warning("Unknown stamp_type: %s", stamp_type)
        return region

    # Resolve set_id
    if not set_id and card_id:
        set_id = _extract_set_id(card_id)

    if not set_id:
        return None

    # EX-era stamped sets take priority (most distinctive)
    if set_id in _STAMPED_SETS:
        return STAMP_REGIONS["ex_era_stamp"]

    # 1st Edition eligible
    if set_id in _FIRST_EDITION_SETS:
        return STAMP_REGIONS["wotc_1st_edition"]

    # Prerelease eligible
    if set_id in _PRERELEASE_SETS:
        return STAMP_REGIONS["wotc_prerelease"]

    return None


def get_all_stamp_regions(
    card_id: str = "",
    set_id: str = "",
) -> list[tuple[str, dict]]:
    """Return ALL applicable stamp regions for a card.

    A card can have multiple stamp regions to check (e.g., an ex7 card
    could be checked for both the set logo stamp and a prerelease stamp).

    Returns:
        List of (region_name, region_dict) tuples.
    """
    if not set_id and card_id:
        set_id = _extract_set_id(card_id)

    if not set_id:
        return []

    regions = []

    if set_id in _STAMPED_SETS:
        regions.append(("ex_era_stamp", STAMP_REGIONS["ex_era_stamp"]))

    if set_id in _FIRST_EDITION_SETS:
        regions.append(("wotc_1st_edition", STAMP_REGIONS["wotc_1st_edition"]))
        regions.append(("wotc_1st_edition_tight", STAMP_REGIONS["wotc_1st_edition_tight"]))

    if set_id in _PRERELEASE_SETS:
        regions.append(("wotc_prerelease", STAMP_REGIONS["wotc_prerelease"]))

    return regions


def crop_stamp_region(
    image: np.ndarray,
    card_id: str = "",
    set_id: str = "",
    stamp_type: str = "",
) -> Optional[np.ndarray]:
    """Crop the stamp region from a card image.

    Uses get_stamp_region() to determine where to crop, then extracts
    that region from the image.

    Args:
        image: Card image as BGR numpy array (H, W, 3).
        card_id: Full card identifier.
        set_id: Set identifier.
        stamp_type: Explicit stamp type key.

    Returns:
        Cropped BGR numpy array, or None if no region applies or
        the crop would be empty.
    """
    region = get_stamp_region(card_id=card_id, set_id=set_id, stamp_type=stamp_type)
    if region is None:
        return None

    return _crop_region(image, region)


def crop_all_stamp_regions(
    image: np.ndarray,
    card_id: str = "",
    set_id: str = "",
) -> list[tuple[str, np.ndarray]]:
    """Crop ALL applicable stamp regions from a card image.

    Returns:
        List of (region_name, cropped_image) tuples.
    """
    regions = get_all_stamp_regions(card_id=card_id, set_id=set_id)
    results = []
    for name, region in regions:
        crop = _crop_region(image, region)
        if crop is not None:
            results.append((name, crop))
    return results


def _crop_region(image: np.ndarray, region: dict) -> Optional[np.ndarray]:
    """Extract a normalized region from a card image.

    Args:
        image: BGR numpy array (H, W, 3) or (H, W).
        region: Dict with x_start, x_end, y_start, y_end as [0,1] fractions.

    Returns:
        Cropped array, or None if the crop would be empty.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    x0 = int(w * region["x_start"])
    x1 = int(w * region["x_end"])
    y0 = int(h * region["y_start"])
    y1 = int(h * region["y_end"])

    # Clamp to image bounds
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    return crop
