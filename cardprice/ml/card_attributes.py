"""O(1) card attribute lookup by card_id.

Built lazily on first access: queries all cards from dim_cards, maps each to
its era via variant_tree.json, computes possible variants and detection checks.

Usage:
    from cardprice.ml.card_attributes import get_card_attrs, get_variant_checks

    attrs = get_card_attrs("base1-4/holofoil")
    attrs.era            # "wotc_base"
    attrs.is_1st_edition_eligible  # True
    attrs.possible_variants        # ["normal", "holofoil", "1st_edition", ...]
    attrs.variant_checks           # [{"variant": "holofoil", "method": ...}, ...]
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from cardprice.ml.variant_detector import (
    ERA_VALID_VARIANTS,
    SET_SPECIAL_VARIANTS,
    get_valid_variants,
)

logger = logging.getLogger(__name__)

_TREE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "variant_tree.json"

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CardAttrs:
    """Pre-computed attributes for a single card."""

    card_id: str
    set_id: str
    era: str  # "wotc_base", "ex_era", "dp_era", etc.
    era_start_year: int
    era_end_year: int
    possible_variants: list[str]
    variant_checks: list[dict] = field(default_factory=list)
    is_stamped_eligible: bool = False
    is_1st_edition_eligible: bool = False
    has_reverse_holo: bool = False
    rarity: str = ""
    supertype: str = ""


# ---------------------------------------------------------------------------
# Lazy-loaded cache
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_card_attrs: dict[str, CardAttrs] | None = None


def _parse_years(years_str: str) -> tuple[int, int]:
    """Parse '1999-2003' or '2023-present' into (start, end) ints."""
    parts = years_str.split("-")
    start = int(parts[0])
    end_str = parts[1] if len(parts) > 1 else parts[0]
    end = 9999 if end_str == "present" else int(end_str)
    return start, end


def _load_variant_tree() -> dict:
    with open(_TREE_PATH) as f:
        return json.load(f)


def _build_set_to_era(tree: dict) -> dict[str, dict]:
    """Map set_id -> {era_key, era_number, years_start, years_end, variants_spec}."""
    index: dict[str, dict] = {}
    for era_key, era_info in tree["eras"].items():
        start, end = _parse_years(era_info["years"])
        for set_id in era_info["sets"]:
            index[set_id] = {
                "era_key": era_key,
                "era_number": era_info["era_number"],
                "years_start": start,
                "years_end": end,
                "variants_spec": era_info["possible_variants"],
            }
    return index


def _build_variant_checks(
    possible: list[str],
    variants_spec: dict,
    set_id: str,
    stamped_sets: dict,
) -> list[dict]:
    """Build ordered list of variant detection steps for a card."""
    checks: list[dict] = []
    for v in possible:
        if v == "normal":
            # Normal is the default fallback, no active check needed
            continue

        spec = variants_spec.get(v, {})
        detection = spec.get("detection", "default")

        check: dict = {
            "variant": v,
            "method": detection,
        }

        # Add region info if present
        region = spec.get("region")
        if region:
            check["region"] = region

        # Add stamp region for stamped variants
        stamp_region = spec.get("stamp_region")
        if stamp_region:
            check["stamp_region"] = stamp_region

        # For stamped cards, add the stamp text
        if v == "stamped" and set_id in stamped_sets:
            check["stamp_text"] = stamped_sets[set_id].get("stamp_text", "")

        checks.append(check)

    return checks


def _extract_set_id(card_id: str) -> str:
    """Extract set prefix from card_id like 'base1-4/holofoil' -> 'base1'."""
    bare = card_id.split("/")[0]
    return bare.rsplit("-", 1)[0] if "-" in bare else bare


_FIRST_EDITION_SETS = frozenset([
    "base1", "base2", "base3", "base5",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
])

_STAMPED_SETS = frozenset([
    "ex7", "ex8", "ex9", "ex10", "ex11",
    "ex12", "ex13", "ex14", "ex15", "ex16",
])


def _build_attrs() -> dict[str, CardAttrs]:
    """Query dim_cards and build the full lookup dict."""
    from cardprice.db.session import SessionLocal
    from sqlalchemy import text

    tree = _load_variant_tree()
    set_to_era = _build_set_to_era(tree)
    stamped_sets = tree.get("stamped_sets", {}).get("sets", {})

    attrs: dict[str, CardAttrs] = {}

    with SessionLocal() as session:
        rows = session.execute(
            text("SELECT card_id, set_id, rarity, supertype FROM dim_cards")
        ).fetchall()

    logger.info("Building card_attributes lookup for %d cards", len(rows))

    for card_id, set_id, rarity, supertype in rows:
        era_info = set_to_era.get(set_id)

        if era_info is None:
            # Unknown set -- use defaults
            era_key = "unknown"
            era_number = 0
            start_year = 0
            end_year = 0
            variants_spec: dict = {}
        else:
            era_key = era_info["era_key"]
            era_number = era_info["era_number"]
            start_year = era_info["years_start"]
            end_year = era_info["years_end"]
            variants_spec = era_info["variants_spec"]

        # Compute possible variants using the same logic as variant_detector
        possible_set = get_valid_variants(set_id, era_number)
        possible = sorted(possible_set)

        has_reverse = "reverse_holofoil" in possible_set
        is_stamped = set_id in _STAMPED_SETS
        is_1st_ed = set_id in _FIRST_EDITION_SETS

        checks = _build_variant_checks(possible, variants_spec, set_id, stamped_sets)

        attrs[card_id] = CardAttrs(
            card_id=card_id,
            set_id=set_id,
            era=era_key,
            era_start_year=start_year,
            era_end_year=end_year,
            possible_variants=possible,
            variant_checks=checks,
            is_stamped_eligible=is_stamped,
            is_1st_edition_eligible=is_1st_ed,
            has_reverse_holo=has_reverse,
            rarity=rarity or "",
            supertype=supertype or "",
        )

    logger.info("Card attributes lookup ready: %d entries", len(attrs))
    return attrs


def _ensure_loaded() -> dict[str, CardAttrs]:
    """Lazy-load the card attributes dict (thread-safe)."""
    global _card_attrs
    if _card_attrs is not None:
        return _card_attrs
    with _lock:
        # Double-check after acquiring lock
        if _card_attrs is not None:
            return _card_attrs
        _card_attrs = _build_attrs()
        return _card_attrs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level dict, populated lazily. Access via get_card_attrs() for safety,
# or directly after ensuring it's loaded.
CARD_ATTRS: dict[str, CardAttrs] = {}  # Sentinel; replaced on first access


def get_card_attrs(card_id: str) -> CardAttrs | None:
    """O(1) lookup of card attributes including era and variant checks.

    Args:
        card_id: Full card identifier (e.g. "base1-4/holofoil" or "sv3-55/normal").

    Returns:
        CardAttrs dataclass with era, possible_variants, variant_checks, etc.
        None if the card_id is not found in dim_cards.
    """
    store = _ensure_loaded()
    return store.get(card_id)


def get_variant_checks(card_id: str) -> list[dict]:
    """Return the specific variant detection steps for this card.

    Args:
        card_id: Full card identifier.

    Returns:
        List of dicts, each with keys:
          - "variant": variant type string
          - "method": detection method name
          - "region": optional [x0, y0, x1, y1] normalized coords
          - "stamp_region": optional stamp detection region
          - "stamp_text": optional expected stamp text (for stamped cards)
        Empty list if card_id is not found.
    """
    attrs = get_card_attrs(card_id)
    if attrs is None:
        return []
    return attrs.variant_checks


def get_all_attrs() -> dict[str, CardAttrs]:
    """Return the full card attributes dict (lazy-loaded).

    This is the backing store -- do not mutate.
    """
    return _ensure_loaded()


def reload() -> None:
    """Force rebuild of the card attributes cache.

    Useful after DB changes or during development.
    """
    global _card_attrs
    with _lock:
        _card_attrs = None
    _ensure_loaded()
