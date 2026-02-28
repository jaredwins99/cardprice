"""Condition-based pricing model for Pokemon cards.

Models how card condition (raw grades and professional grading) affects
market price relative to a Near Mint baseline.  Provides both hardcoded
default multipliers and data-driven premium analysis from subtype price
spreads in fact_market_prices.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default condition multipliers (relative to NM = 1.0)
# ---------------------------------------------------------------------------

CONDITION_MULTIPLIERS: dict[str, float] = {
    # Professional grading -- PSA
    "PSA 10": 3.0,
    "PSA 9": 1.5,
    "PSA 8": 1.1,
    "PSA 7": 0.9,
    # Professional grading -- BGS (Beckett)
    "BGS 10": 5.0,
    "BGS 9.5": 2.5,
    "BGS 9": 1.5,
    # Professional grading -- CGC
    "CGC 10": 2.5,
    "CGC 9.5": 1.8,
    "CGC 9": 1.3,
    # Raw conditions
    "NM": 1.0,
    "LP": 0.75,
    "MP": 0.5,
    "HP": 0.3,
    "DMG": 0.1,
}

# Mapping from grade_authority + grade to the canonical key above
_GRADE_KEY_MAP: dict[tuple[str | None, str | None], str] = {}
for _key in CONDITION_MULTIPLIERS:
    parts = _key.split(maxsplit=1)
    if len(parts) == 2:
        _GRADE_KEY_MAP[(parts[0], parts[1])] = _key
    else:
        _GRADE_KEY_MAP[(None, _key)] = _key


def _resolve_condition_key(
    condition: str | None = None,
    grade_authority: str | None = None,
    grade: str | None = None,
) -> str:
    """Resolve inventory fields into a CONDITION_MULTIPLIERS key.

    Supports three calling conventions:
      - condition="PSA 10"          (combined string)
      - condition="NM"              (raw condition)
      - grade_authority="PSA", grade="10"  (split, as stored in user_inventory)
    """
    if grade_authority and grade:
        key = f"{grade_authority} {grade}"
        if key in CONDITION_MULTIPLIERS:
            return key

    if condition and condition in CONDITION_MULTIPLIERS:
        return condition

    # Fallback: try parsing "PSA 10" style from condition string
    if condition:
        parts = condition.strip().split(maxsplit=1)
        if len(parts) == 2:
            combined = f"{parts[0].upper()} {parts[1]}"
            if combined in CONDITION_MULTIPLIERS:
                return combined

    return "NM"  # safe default


def get_multiplier(
    condition: str | None = None,
    grade_authority: str | None = None,
    grade: str | None = None,
    custom_multipliers: dict[str, float] | None = None,
) -> tuple[str, float]:
    """Return (resolved_key, multiplier) for the given condition/grade."""
    table = custom_multipliers if custom_multipliers else CONDITION_MULTIPLIERS
    key = _resolve_condition_key(condition, grade_authority, grade)
    return key, table.get(key, 1.0)


# ---------------------------------------------------------------------------
# 1. estimate_condition_price
# ---------------------------------------------------------------------------

_LATEST_PRICE_SQL = text("""
    SELECT market_price, low_price, high_price, subtype_name, price_date
    FROM fact_market_prices
    WHERE card_id = :card_id
      AND market_price > 0
    ORDER BY price_date DESC
    LIMIT 1
""")


def estimate_condition_price(
    card_id: str,
    condition: str,
    session: Session,
    *,
    grade_authority: str | None = None,
    grade: str | None = None,
    custom_multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate price for *card_id* in the given *condition*.

    Parameters
    ----------
    card_id:
        dim_cards.card_id (e.g. ``"base1-4/holofoil"``).
    condition:
        A key from CONDITION_MULTIPLIERS, e.g. ``"PSA 10"`` or ``"LP"``.
    session:
        An open SQLAlchemy Session.
    grade_authority / grade:
        Optional split form; takes priority over *condition* when both
        are supplied.
    custom_multipliers:
        Override the default multiplier table.

    Returns
    -------
    dict with keys: base_price, condition, multiplier, estimated_price,
    price_range_low, price_range_high, subtype, price_date.
    Returns ``None`` values when no price data exists.
    """
    cond_key, multiplier = get_multiplier(
        condition, grade_authority, grade, custom_multipliers
    )

    row = session.execute(_LATEST_PRICE_SQL, {"card_id": card_id}).fetchone()

    if row is None:
        return {
            "card_id": card_id,
            "base_price": None,
            "condition": cond_key,
            "multiplier": multiplier,
            "estimated_price": None,
            "price_range_low": None,
            "price_range_high": None,
            "subtype": None,
            "price_date": None,
        }

    base = float(row.market_price)
    low = float(row.low_price) if row.low_price else base * 0.8
    high = float(row.high_price) if row.high_price else base * 1.2

    return {
        "card_id": card_id,
        "base_price": round(base, 2),
        "condition": cond_key,
        "multiplier": multiplier,
        "estimated_price": round(base * multiplier, 2),
        "price_range_low": round(low * multiplier, 2),
        "price_range_high": round(high * multiplier, 2),
        "subtype": row.subtype_name,
        "price_date": str(row.price_date),
    }


# ---------------------------------------------------------------------------
# 2. value_inventory
# ---------------------------------------------------------------------------

_INVENTORY_SQL = text("""
    SELECT
        ui.id            AS inv_id,
        ui.card_id,
        ui.quantity,
        ui.condition,
        ui.grade_authority,
        ui.grade,
        ui.acquisition_price,
        dc.name          AS card_name,
        dc.set_id
    FROM user_inventory ui
    JOIN dim_cards dc ON dc.card_id = ui.card_id
""")


def value_inventory(
    session: Session,
    *,
    custom_multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Value every card in ``user_inventory``.

    Returns
    -------
    dict with keys:
        total_value       – float, sum of estimated prices * quantity
        total_cost        – float, sum of acquisition prices (where known)
        card_count        – int, total quantity across all rows
        cards             – list of per-item dicts
        value_by_condition – dict mapping condition tier -> subtotal
    """
    rows = session.execute(_INVENTORY_SQL).fetchall()

    cards: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    card_count = 0
    value_by_condition: dict[str, float] = defaultdict(float)

    for row in rows:
        qty = row.quantity or 1
        card_count += qty

        est = estimate_condition_price(
            card_id=row.card_id,
            condition=row.condition or "NM",
            session=session,
            grade_authority=row.grade_authority,
            grade=row.grade,
            custom_multipliers=custom_multipliers,
        )

        line_value = (est["estimated_price"] or 0.0) * qty
        total_value += line_value
        value_by_condition[est["condition"]] += line_value

        acq = float(row.acquisition_price) if row.acquisition_price else 0.0
        total_cost += acq * qty

        cards.append(
            {
                "inv_id": row.inv_id,
                "card_id": row.card_id,
                "card_name": row.card_name,
                "set_id": row.set_id,
                "quantity": qty,
                "condition": est["condition"],
                "multiplier": est["multiplier"],
                "base_price": est["base_price"],
                "estimated_price": est["estimated_price"],
                "line_value": round(line_value, 2),
                "acquisition_price": acq if acq else None,
            }
        )

    return {
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "unrealized_gain": round(total_value - total_cost, 2),
        "card_count": card_count,
        "cards": cards,
        "value_by_condition": dict(value_by_condition),
    }


# ---------------------------------------------------------------------------
# 3. analyze_condition_premiums
# ---------------------------------------------------------------------------

_SUBTYPE_PREMIUM_SQL = text("""
    WITH base AS (
        SELECT
            card_id,
            subtype_name,
            AVG(market_price) AS avg_market,
            AVG(low_price)    AS avg_low,
            AVG(high_price)   AS avg_high,
            COUNT(*)          AS sample_size
        FROM fact_market_prices
        WHERE market_price > 0
          AND card_id IS NOT NULL
        GROUP BY card_id, subtype_name
    ),
    normal AS (
        SELECT card_id, avg_market AS normal_price
        FROM base
        WHERE subtype_name = 'Normal'
          AND avg_market > 0
    )
    SELECT
        b.subtype_name,
        COUNT(*)                                       AS card_count,
        AVG(b.avg_market)                              AS avg_price,
        AVG(b.avg_low)                                 AS avg_low,
        AVG(b.avg_high)                                AS avg_high,
        AVG(b.avg_market / n.normal_price)             AS avg_premium_ratio,
        PERCENTILE_CONT(0.5) WITHIN GROUP
            (ORDER BY b.avg_market / n.normal_price)   AS median_premium_ratio,
        MIN(b.avg_market / n.normal_price)             AS min_premium_ratio,
        MAX(b.avg_market / n.normal_price)             AS max_premium_ratio
    FROM base b
    JOIN normal n ON n.card_id = b.card_id
    WHERE b.subtype_name != 'Normal'
    GROUP BY b.subtype_name
    ORDER BY avg_premium_ratio DESC
""")

_SUBTYPE_OVERVIEW_SQL = text("""
    SELECT
        subtype_name,
        COUNT(*)          AS row_count,
        AVG(market_price) AS avg_market,
        AVG(low_price)    AS avg_low,
        AVG(high_price)   AS avg_high
    FROM fact_market_prices
    WHERE market_price > 0
    GROUP BY subtype_name
    ORDER BY row_count DESC
""")


def analyze_condition_premiums(session: Session) -> dict[str, Any]:
    """Analyze data-driven premium ratios from subtype price data.

    Compares each non-Normal subtype's average market price to the Normal
    subtype for the same card_id, giving an empirical multiplier.

    Returns
    -------
    dict with keys:
        subtypes_overview – list of dicts with aggregate stats per subtype
        premium_ratios    – list of dicts per subtype showing avg/median/min/max
                           premium relative to 'Normal'
        suggested_multipliers – dict mapping subtype_name -> recommended multiplier
                               (uses median to be robust against outliers)
    """
    # Overview of all subtypes
    overview_rows = session.execute(_SUBTYPE_OVERVIEW_SQL).fetchall()
    subtypes_overview = [
        {
            "subtype_name": r.subtype_name,
            "row_count": r.row_count,
            "avg_market": round(float(r.avg_market), 2),
            "avg_low": round(float(r.avg_low), 2) if r.avg_low else None,
            "avg_high": round(float(r.avg_high), 2) if r.avg_high else None,
        }
        for r in overview_rows
    ]

    # Per-subtype premium ratios relative to Normal
    premium_rows = session.execute(_SUBTYPE_PREMIUM_SQL).fetchall()
    premium_ratios = []
    suggested_multipliers: dict[str, float] = {"Normal": 1.0}

    for r in premium_rows:
        median = float(r.median_premium_ratio) if r.median_premium_ratio else 1.0
        premium_ratios.append(
            {
                "subtype_name": r.subtype_name,
                "card_count": r.card_count,
                "avg_price": round(float(r.avg_price), 2),
                "avg_premium_ratio": round(float(r.avg_premium_ratio), 3),
                "median_premium_ratio": round(median, 3),
                "min_premium_ratio": round(float(r.min_premium_ratio), 3),
                "max_premium_ratio": round(float(r.max_premium_ratio), 3),
            }
        )
        suggested_multipliers[r.subtype_name] = round(median, 2)

    return {
        "subtypes_overview": subtypes_overview,
        "premium_ratios": premium_ratios,
        "suggested_multipliers": suggested_multipliers,
    }
