"""Page-level context inference for binder card identification.

When scanning a binder page, cards on the same page tend to come from the same
set or era.  This module exploits that insight:

1. ``identify_page_context`` examines already-identified cards to determine the
   most likely set(s) and era for the page.
2. ``rerank_with_context`` re-orders a candidate list for an unidentified card,
   boosting candidates that belong to the inferred set/era.

Typical flow::

    results = [identify_card(img) for img in segmented_cards]
    ctx = identify_page_context(results)
    for r in results:
        if r["confidence"] < threshold and r.get("raw_response", {}).get("top_alternatives"):
            r["raw_response"]["top_alternatives"] = rerank_with_context(
                r["raw_response"]["top_alternatives"], ctx
            )
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Era classification
# ---------------------------------------------------------------------------

# Maps set_id prefix patterns -> era name.  Ordered so the first match wins.
# Derived from the official Pokemon TCG set groupings (series column in dim_sets).
_SERIES_TO_ERA: dict[str, str] = {
    # WotC era
    "base": "wotc",
    "jungle": "wotc",
    "fossil": "wotc",
    "base2": "wotc",
    "gym": "wotc",
    "neo": "wotc",
    "si": "wotc",          # Southern Islands
    "basep": "wotc",       # Base Set promos
    # e-Card era
    "ecard": "e-card",
    # EX era
    "ex": "ex",
    "tk": "ex",            # trainer kits
    "pop": "ex",           # POP series
    # Diamond & Pearl era
    "dp": "dp",
    "pl": "dp",            # Platinum series
    # HeartGold SoulSilver era
    "hgss": "hgss",
    "col": "hgss",         # Call of Legends
    # Black & White era
    "bw": "bw",
    # XY era
    "xy": "xy",
    "g1": "xy",            # Generations
    # Sun & Moon era
    "sm": "sm",
    "det": "sm",           # Detective Pikachu
    # Sword & Shield era
    "swsh": "swsh",
    # Scarlet & Violet era
    "sv": "sv",
}


# Adjacent eras: binder pages often mix cards from neighboring eras.
# Era filter should treat these as compatible when filtering candidates.
_ADJACENT_ERAS: dict[str, set[str]] = {
    "wotc": {"e-card"},
    "e-card": {"wotc", "ex"},
    "ex": {"e-card", "dp"},
    "dp": {"ex", "hgss"},
    "hgss": {"dp", "bw"},
    "bw": {"hgss", "xy"},
    "xy": {"bw", "sm"},
    "sm": {"xy", "swsh"},
    "swsh": {"sm", "sv"},
    "sv": {"swsh"},
}


def _eras_compatible(era1: str, era2: str) -> bool:
    """Check if two eras are the same or adjacent."""
    if era1 == era2:
        return True
    return era2 in _ADJACENT_ERAS.get(era1, set())


def _extract_set_id(card_id: str) -> Optional[str]:
    """Extract the set_id portion from a card_id like 'base1-4/holofoil'.

    card_id format: ``{set_id}-{number}/{variant}``
    We strip the variant (after ``/``) then remove the trailing ``-number``.
    """
    if not card_id:
        return None
    # Strip variant
    base = card_id.split("/", 1)[0]
    # Remove trailing -number  (e.g. "base1-4" -> "base1")
    parts = base.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else base


def _era_for_set(set_id: str) -> Optional[str]:
    """Classify a set_id into an era string using prefix matching."""
    if not set_id:
        return None
    sid = set_id.lower()
    for prefix, era in _SERIES_TO_ERA.items():
        if sid.startswith(prefix):
            return era
    return None


def _era_for_set_from_db(set_id: str) -> Optional[str]:
    """Look up the series/era for a set_id via the database (fallback).

    Returns the ``series`` column from ``dim_sets``, or None if unavailable.
    This is called only when the prefix heuristic fails, so it is rarely used.
    """
    try:
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text

        with SessionLocal() as session:
            row = session.execute(
                text("SELECT series FROM dim_sets WHERE set_id = :sid"),
                {"sid": set_id},
            ).fetchone()
            if row and row.series:
                return row.series.lower().replace(" ", "-").replace("&", "")
    except Exception as exc:
        logger.debug("DB era lookup failed for %s: %s", set_id, exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_page_context(card_results: list[dict]) -> dict:
    """Analyze identified cards on a page to determine likely set/era context.

    Args:
        card_results: List of identification result dicts.  Each dict should
            have at least ``card_id`` (str | None) and ``confidence`` (float).
            Unidentified cards have ``card_id=None``.

    Returns:
        A context dict::

            {
                "likely_sets": ["ecard3", "ecard2"],   # ordered by frequency
                "era": "e-card",                       # most common era, or None
                "confidence": 0.8,                     # how consistent the page is
                "set_counts": {"ecard3": 5, "ecard2": 1},
                "era_counts": {"e-card": 6},
                "total_identified": 6,
                "total_cards": 9,
            }

        ``confidence`` is the fraction of identified cards that belong to the
        top era.  If only 1 card is identified, confidence is capped at 0.5.
    """
    total_cards = len(card_results)
    set_counter: Counter[str] = Counter()
    era_counter: Counter[str] = Counter()

    for r in card_results:
        cid = r.get("card_id")
        conf = r.get("confidence", 0.0)
        if not cid or conf <= 0:
            continue

        set_id = _extract_set_id(cid)
        if set_id:
            # Weight by confidence so low-confidence guesses don't dominate.
            set_counter[set_id] += conf
            era = _era_for_set(set_id) or _era_for_set_from_db(set_id)
            if era:
                era_counter[era] += conf

    total_identified = sum(1 for r in card_results if r.get("card_id") and r.get("confidence", 0) > 0)

    if not set_counter:
        return {
            "likely_sets": [],
            "era": None,
            "confidence": 0.0,
            "set_counts": {},
            "era_counts": {},
            "total_identified": 0,
            "total_cards": total_cards,
        }

    # Rank sets by weighted count
    likely_sets = [s for s, _ in set_counter.most_common()]
    top_era = era_counter.most_common(1)[0][0] if era_counter else None

    # Confidence: fraction of weighted votes that agree on the top era.
    total_weight = sum(era_counter.values())
    top_era_weight = era_counter.most_common(1)[0][1] if era_counter else 0
    era_agreement = top_era_weight / total_weight if total_weight > 0 else 0.0

    # Discount confidence if only 1 card is identified (not enough evidence).
    if total_identified == 1:
        era_agreement = min(era_agreement, 0.5)
    elif total_identified == 2:
        era_agreement = min(era_agreement, 0.7)

    # Unweighted counts for display
    raw_set_counts: Counter[str] = Counter()
    raw_era_counts: Counter[str] = Counter()
    for r in card_results:
        cid = r.get("card_id")
        if not cid or r.get("confidence", 0) <= 0:
            continue
        sid = _extract_set_id(cid)
        if sid:
            raw_set_counts[sid] += 1
            era = _era_for_set(sid) or _era_for_set_from_db(sid)
            if era:
                raw_era_counts[era] += 1

    context = {
        "likely_sets": likely_sets,
        "era": top_era,
        "confidence": round(era_agreement, 3),
        "set_counts": dict(raw_set_counts.most_common()),
        "era_counts": dict(raw_era_counts.most_common()),
        "total_identified": total_identified,
        "total_cards": total_cards,
    }

    logger.info(
        "Page context: sets=%s era=%s confidence=%.2f (%d/%d identified)",
        likely_sets[:3], top_era, era_agreement, total_identified, total_cards,
    )
    return context


def rerank_with_context(candidates: list, context: dict) -> list:
    """Re-rank match candidates favoring the page's set/era context.

    Args:
        candidates: List of candidate tuples or dicts.  Supported formats:

            - ``(card_id, score)`` tuples (from DINOv2/CLIP matchers)
            - ``{"card_id": str, "score": float, ...}`` dicts

        context: Dict returned by ``identify_page_context``.  Must contain
            ``likely_sets`` and ``era``.

    Returns:
        A new list in the same format, re-sorted by adjusted score.
        The original scores are preserved; only ordering changes.

    The boost logic:
        - Same set as page majority  ->  +0.15 * context_confidence
        - Same era but different set ->  +0.08 * context_confidence
        - Different era              ->  no change

    This means a DINO candidate at 0.60 similarity from the same set can
    overtake a 0.68 candidate from a different era when context confidence
    is high -- which matches the real-world prior that binder pages are
    organized by set.
    """
    if not candidates or not context.get("likely_sets"):
        return candidates

    likely_sets = set(context.get("likely_sets", []))
    page_era = context.get("era")
    ctx_conf = context.get("confidence", 0.0)

    # Boost magnitudes scaled by context confidence
    SET_BOOST = 0.15 * ctx_conf
    ERA_BOOST = 0.08 * ctx_conf

    def _adjust(card_id: str, original_score: float) -> float:
        """Compute adjusted score for ranking."""
        sid = _extract_set_id(card_id)
        if not sid:
            return original_score

        if sid in likely_sets:
            return original_score + SET_BOOST

        cand_era = _era_for_set(sid)
        if cand_era and cand_era == page_era:
            return original_score + ERA_BOOST

        return original_score

    # Detect format: tuple vs dict
    if candidates and isinstance(candidates[0], (list, tuple)):
        # (card_id, score) format
        scored = [(c[0], c[1], _adjust(c[0], float(c[1]))) for c in candidates]
        scored.sort(key=lambda x: x[2], reverse=True)
        # Return in original format (preserve tuple/list type)
        fmt = type(candidates[0])
        return [fmt((cid, orig)) for cid, orig, _ in scored]

    elif candidates and isinstance(candidates[0], dict):
        # {"card_id": ..., "score": ...} format
        # Also support "similarity" key used by some matchers
        score_key = "score"
        if score_key not in candidates[0]:
            for alt in ("similarity", "confidence", "distance"):
                if alt in candidates[0]:
                    score_key = alt
                    break

        id_key = "card_id"
        if id_key not in candidates[0]:
            # Some results use just "id"
            for alt in ("id", "cid"):
                if alt in candidates[0]:
                    id_key = alt
                    break

        scored = []
        for c in candidates:
            cid = c.get(id_key, "")
            orig = float(c.get(score_key, 0))
            adj = _adjust(cid, orig)
            scored.append((c, adj))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    # Unknown format -- return unchanged
    logger.warning("rerank_with_context: unknown candidate format %s", type(candidates[0]))
    return candidates
