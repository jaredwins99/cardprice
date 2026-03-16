"""Variant tree helper: query possible variants and detection methods by card/set/era.

Loads data/variant_tree.json and provides a clean Python API for querying
what variants to check for a given card, and which detection method to use.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TREE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "variant_tree.json"

# ---------------------------------------------------------------------------
# Internal: load and index the tree
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_tree() -> dict:
    """Load and return the variant tree JSON (cached)."""
    with open(_TREE_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_set_index() -> dict[str, dict]:
    """Build set_id -> era info index from the tree.

    Returns dict mapping set_id to:
      {"era_key": str, "era_number": int, "era_name": str,
       "possible_variants": dict}
    """
    tree = _load_tree()
    index: dict[str, dict] = {}
    for era_key, era_info in tree["eras"].items():
        era_num = era_info["era_number"]
        era_name = era_info["name"]
        variants = era_info["possible_variants"]
        for set_id in era_info["sets"]:
            index[set_id] = {
                "era_key": era_key,
                "era_number": era_num,
                "era_name": era_name,
                "possible_variants": variants,
            }
    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_possible_variants(card_id: str, set_id: str | None = None) -> list[str]:
    """Return list of variant types to check for this card based on era/set.

    Uses the variant tree to determine which variants are possible for the
    card's set.  Filters variants by set-specific restrictions (e.g.,
    1st_edition only on certain WotC sets, shadowless only on base1).

    Args:
        card_id: Card identifier like "base1-4" or "ex11-55".
        set_id: Optional set prefix override.  If not provided, extracted
            from card_id.

    Returns:
        List of variant type strings (e.g. ["normal", "holofoil",
        "1st_edition", "1st_edition_holofoil"]).  Returns ["normal", "holofoil"]
        as fallback if the set is not found in the tree.
    """
    if set_id is None:
        # Extract set prefix from card_id: "base1-4/holofoil" -> "base1"
        bare = card_id.split("/")[0]
        set_id = bare.rsplit("-", 1)[0] if "-" in bare else bare

    index = _build_set_index()
    if set_id not in index:
        logger.debug("Set %s not in variant tree, returning defaults", set_id)
        return ["normal", "holofoil"]

    era_info = index[set_id]
    variants = era_info["possible_variants"]

    result = []
    for variant_type, spec in variants.items():
        # Check if this variant is restricted to specific sets
        variant_sets = spec.get("sets")
        if variant_sets == "all" or variant_sets is None:
            result.append(variant_type)
        elif isinstance(variant_sets, list) and set_id in variant_sets:
            result.append(variant_type)
        # else: variant not applicable to this set

    return result if result else ["normal", "holofoil"]


def get_detection_method(variant_type: str) -> str:
    """Return which detection method to use for this variant type.

    Args:
        variant_type: Variant string (e.g. "holofoil", "1st_edition",
            "stamped", "full_art").

    Returns:
        Detection method name string (e.g. "holo_detector", "stamp_ocr",
        "stamped_detector", "default").
    """
    tree = _load_tree()
    methods = tree.get("detection_methods", {})

    # First check the variant tree eras for detection info
    for era_info in tree["eras"].values():
        if variant_type in era_info["possible_variants"]:
            spec = era_info["possible_variants"][variant_type]
            detection = spec.get("detection")
            if detection:
                return detection

    # Fallback: check detection_methods section directly
    if variant_type in methods:
        return variant_type

    return "default"


def get_variant_info(variant_type: str, set_id: str = "") -> dict | None:
    """Return full variant spec from the tree for a given type and set.

    Args:
        variant_type: Variant string (e.g. "stamped", "1st_edition").
        set_id: Optional set prefix for set-specific info.

    Returns:
        Dict with keys like "description", "detection", "region", "sets", "notes".
        None if the variant type is not found.
    """
    index = _build_set_index()
    if set_id and set_id in index:
        variants = index[set_id]["possible_variants"]
        if variant_type in variants:
            return dict(variants[variant_type])

    # Search all eras for this variant type
    tree = _load_tree()
    for era_info in tree["eras"].values():
        if variant_type in era_info["possible_variants"]:
            return dict(era_info["possible_variants"][variant_type])

    return None


def get_stamped_sets() -> dict[str, dict]:
    """Return info about sets that have stamped reverse holos.

    Returns:
        Dict mapping set_id to {"name": str, "stamp_text": str}.
    """
    tree = _load_tree()
    stamped = tree.get("stamped_sets", {})
    return dict(stamped.get("sets", {}))


def is_stamped_set(set_id: str) -> bool:
    """Check if a set has stamped reverse holo cards.

    Args:
        set_id: Set prefix (e.g. "ex11").

    Returns:
        True if the set has stamped reverse holos (EX era ex7-ex16).
    """
    tree = _load_tree()
    stamped = tree.get("stamped_sets", {})
    return set_id in stamped.get("sets", {})


def get_era_info(set_id: str) -> dict | None:
    """Return era info for a set from the variant tree.

    Args:
        set_id: Set prefix (e.g. "base1", "ex11", "sv3").

    Returns:
        Dict with "era_key", "era_number", "era_name", "possible_variants".
        None if set not found.
    """
    index = _build_set_index()
    return dict(index[set_id]) if set_id in index else None
