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
_STAMP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "stamp_positions.json"

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


# ---------------------------------------------------------------------------
# Stamp position helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_stamp_positions() -> dict:
    """Load and return the stamp positions JSON (cached)."""
    with open(_STAMP_PATH) as f:
        return json.load(f)


def _extract_set_id(card_id: str) -> str:
    """Extract the set prefix from a card_id like 'base1-4' -> 'base1'."""
    bare = card_id.split("/")[0]
    return bare.rsplit("-", 1)[0] if "-" in bare else bare


def get_stamp_checks(card_id: str, era: str = "") -> list[dict]:
    """Return which stamps to check for and where, based on card and era.

    Consults data/stamp_positions.json to determine which stamp types are
    applicable to this card's era and set, and returns their detection
    regions and methods.

    Args:
        card_id: Card identifier like "base1-4" or "ex11-55".
        era: Era key (e.g. "wotc_base", "ex_era"). If empty, resolved
            from card_id via the variant tree set index.

    Returns:
        List of dicts, each with keys:
          - stamp_type: str (e.g. "1st_edition", "prerelease")
          - description: str
          - position: dict with "x_range" and "y_range" (normalized 0-1)
          - alt_position: dict or None (secondary search region)
          - detection_method: str
          - detection_details: dict with method-specific info
          - visual: str (human-readable description of what to look for)

        Returns empty list if no stamps are applicable.
    """
    set_id = _extract_set_id(card_id)

    # Resolve era from set index if not provided
    if not era:
        index = _build_set_index()
        if set_id in index:
            era = index[set_id]["era_key"]
        else:
            logger.debug("Cannot resolve era for %s, returning empty stamp checks", card_id)
            return []

    stamp_data = _load_stamp_positions()
    stamp_types = stamp_data.get("stamp_types", {})
    era_matrix = stamp_data.get("era_stamp_matrix", {})

    # Get the list of applicable stamp type keys from the era matrix
    era_entry = era_matrix.get(era, {})
    applicable_stamp_keys = set(era_entry.get("applicable_stamps", []))

    results: list[dict] = []
    for stamp_key, spec in stamp_types.items():
        # Check if this stamp type applies to the current era
        stamp_eras = spec.get("eras", [])
        era_match = "all" in stamp_eras or era in stamp_eras
        matrix_match = stamp_key in applicable_stamp_keys

        if not (era_match or matrix_match):
            continue

        # Check if the stamp is restricted to specific sets
        eligible = spec.get("eligible_sets")
        if eligible is not None and eligible != "all_prerelease":
            if set_id not in eligible:
                continue

        # Build the position dict (flatten x_range/y_range from nested position)
        pos = spec.get("position", {})
        position = {
            "x_range": pos.get("x_range", [0.0, 1.0]),
            "y_range": pos.get("y_range", [0.0, 1.0]),
        }

        # Alt position (optional second region to check)
        alt_pos_raw = spec.get("alt_position") or spec.get("tight_position")
        alt_position = None
        if alt_pos_raw:
            alt_position = {
                "x_range": alt_pos_raw.get("x_range", [0.0, 1.0]),
                "y_range": alt_pos_raw.get("y_range", [0.0, 1.0]),
            }

        results.append({
            "stamp_type": stamp_key,
            "description": spec.get("description", ""),
            "position": position,
            "alt_position": alt_position,
            "detection_method": spec.get("detection_method", "unknown"),
            "detection_details": spec.get("detection_details", {}),
            "visual": spec.get("visual", ""),
        })

    return results


def get_stamp_region(stamp_type: str) -> dict | None:
    """Return the normalized position region for a specific stamp type.

    Args:
        stamp_type: Stamp type key (e.g. "1st_edition", "ex_set_stamp").

    Returns:
        Dict with "x_range" and "y_range" as [min, max] lists (0-1 normalized),
        or None if the stamp type is not found.
    """
    stamp_data = _load_stamp_positions()
    spec = stamp_data.get("stamp_types", {}).get(stamp_type)
    if not spec:
        return None

    pos = spec.get("position", {})
    return {
        "x_range": pos.get("x_range", [0.0, 1.0]),
        "y_range": pos.get("y_range", [0.0, 1.0]),
    }
