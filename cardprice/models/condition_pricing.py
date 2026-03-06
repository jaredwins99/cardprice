"""Condition-based pricing model for Pokemon cards.

Models how card condition (raw grades and professional grading) affects
market price relative to a Near Mint baseline.  Provides both hardcoded
default multipliers and data-driven premium analysis from subtype price
spreads in fact_market_prices.

Research basis (March 2026):
    Raw conditions:
        - PokeTrace condition guide: LP 70-90%, MP 40-70%, HP 20-40%, DMG 5-20%
        - PokeScope condition guide: LP ~80%, HP discount 60-80% off NM
        - TCGPlayer seller defaults: LP ~75-85%, MP ~50%, HP ~25-35%
        - Midpoints used for default multipliers

    Graded cards (relative to raw NM):
        - PSA 10: 3-5x raw (modern), 5-10x (vintage). Default 3.5x.
        - PSA 9:  2-3x raw (30-50% of PSA 10 value). Default 1.8x.
        - PSA 8:  1.5-2x raw. Default 1.3x.
        - PSA 7:  0.8-1.2x raw (below NM, slab adds marginal value). Default 0.95x.
        - BGS 10 Black Label: 2-3x PSA 10 (extreme rarity). Default 8.0x.
        - BGS 9.5: ~equivalent to PSA 10, slight discount. Default 2.8x.
        - BGS 9:  ~equivalent to PSA 9, slight discount. Default 1.6x.
        - CGC 10: ~equivalent to PSA 10 with 20-35% discount. Default 2.5x.
        - CGC 9.5: 30-50% below PSA 10. Default 2.0x.
        - CGC 9:  slightly below PSA 9. Default 1.4x.

    Value-tier variation:
        - Bulk cards (<$5 raw): graded premiums compressed (PSA 10 ~2-3x)
        - Mid-range ($5-50): standard multipliers apply
        - High-value ($50-500): slightly above standard (PSA 10 ~4-5x)
        - Chase/vintage ($500+): large premiums (PSA 10 ~5-10x)
        - Raw condition discounts are more uniform across tiers

    Sources:
        - PokeTrace (poketrace.com/blog/raw-pokemon-card-conditions)
        - PokeScope (pokescope.app/condition-guide)
        - PKMhobby (pkmhobby.com/blogs/cards/graded-pokemon-card-values)
        - PokeInvest (pokeinvest.io/card-market-insights)
        - OG Cards (ogcards.com/blogs/pokemon-cards/should-you-grade-your-pokemon-cards)
        - Shop Cards USA (shopcardsusa.com/blogs/news/psa-10-vs-bgs-black-label-10-comparison)
        - TCGPlayer seller conditioning standards (March 2025 update)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default condition multipliers (relative to NM = 1.0)
#
# Each entry: (multiplier, ci_low, ci_high) where ci = 80% confidence interval
# The CI reflects observed market variance across card types and eras.
# ---------------------------------------------------------------------------

CONDITION_MULTIPLIERS_WITH_CI: dict[str, tuple[float, float, float]] = {
    # Professional grading -- PSA
    #                       (default, ci_low, ci_high)
    "PSA 10":              (3.50, 2.50, 5.00),
    "PSA 9":               (1.80, 1.30, 2.50),
    "PSA 8":               (1.30, 1.10, 1.70),
    "PSA 7":               (0.95, 0.80, 1.15),
    "PSA 6":               (0.75, 0.60, 0.90),
    "PSA 5":               (0.60, 0.45, 0.75),
    "PSA 4":               (0.45, 0.30, 0.60),
    "PSA 3":               (0.35, 0.20, 0.50),
    "PSA 2":               (0.25, 0.15, 0.40),
    "PSA 1":               (0.15, 0.08, 0.25),
    # Professional grading -- BGS (Beckett)
    "BGS 10":              (8.00, 5.00, 15.00),  # Black Label; extreme variance
    "BGS 9.5":             (2.80, 2.00, 4.00),
    "BGS 9":               (1.60, 1.20, 2.20),
    "BGS 8.5":             (1.15, 0.90, 1.40),
    "BGS 8":               (0.95, 0.75, 1.20),
    # Professional grading -- CGC
    "CGC 10":              (2.50, 1.80, 3.50),
    "CGC 9.5":             (2.00, 1.50, 2.80),
    "CGC 9":               (1.40, 1.10, 1.80),
    "CGC 8.5":             (1.05, 0.85, 1.30),
    "CGC 8":               (0.90, 0.70, 1.10),
    # Raw conditions
    "NM":                  (1.00, 0.90, 1.10),
    "LP":                  (0.80, 0.70, 0.90),
    "MP":                  (0.55, 0.40, 0.70),
    "HP":                  (0.30, 0.20, 0.40),
    "DMG":                 (0.12, 0.05, 0.20),
}

# Flat multiplier dict for backward compatibility
CONDITION_MULTIPLIERS: dict[str, float] = {
    k: v[0] for k, v in CONDITION_MULTIPLIERS_WITH_CI.items()
}

# ---------------------------------------------------------------------------
# Value-tier adjustment factors
#
# Graded card premiums vary by the raw NM value of the card.  These tiers
# apply a scaling factor to graded multipliers (raw conditions are unaffected
# because LP/MP/HP/DMG discounts are relatively uniform across price points).
# ---------------------------------------------------------------------------

VALUE_TIERS: list[tuple[str, float, float, float]] = [
    # (label, max_raw_price, graded_scale_factor, description)
    #  graded_scale_factor multiplies the graded multiplier delta above 1.0
    ("bulk",      5.0,   0.65),   # PSA 10 on a $2 card: 1 + (3.5-1)*0.65 = 2.63x
    ("low",      20.0,   0.80),   # PSA 10 on a $15 card: 1 + 2.5*0.80 = 3.00x
    ("mid",      50.0,   1.00),   # Standard multipliers
    ("high",    500.0,   1.15),   # PSA 10 on a $200 card: 1 + 2.5*1.15 = 3.88x
    ("chase", 99999.0,   1.50),   # PSA 10 on a $1000 card: 1 + 2.5*1.50 = 4.75x
]


def get_value_tier_scale(raw_nm_price: float | None) -> float:
    """Return the graded-multiplier scale factor for a given raw NM price."""
    if raw_nm_price is None or raw_nm_price <= 0:
        return 1.0
    for _label, max_price, scale in VALUE_TIERS:
        if raw_nm_price <= max_price:
            return scale
    return VALUE_TIERS[-1][2]


# ---------------------------------------------------------------------------
# Continuous grade-to-multiplier mapping
#
# Maps a numeric grade (1.0 - 10.0) to a multiplier using a double-sigmoid
# that captures both the steep "PSA 10 cliff" AND the gradual rise at lower
# grades.  A single sigmoid cannot fit both behaviors simultaneously.
#
#   multiplier(g) = a + b1*sigmoid(s1*(g - c1)) + b2*sigmoid(s2*(g - c2))
#
# The first sigmoid (steep, s1=1.85) models the PSA 9->10 cliff.
# The second sigmoid (gradual, s2=0.23) models the smooth rise from 1->8.
#
# Least-squares fit to PSA anchor points (max error <5% at integer grades):
#   g=10 -> 3.50, g=9 -> 1.80, g=8 -> 1.30, g=7 -> 0.95,
#   g=6 -> 0.75, g=5 -> 0.60, g=4 -> 0.45, g=3 -> 0.35,
#   g=2 -> 0.25, g=1 -> 0.15
# ---------------------------------------------------------------------------

# Double-sigmoid parameters (optimized via scipy Nelder-Mead)
_DS_A  = -0.1196    # offset
_DS_B1 = 17.3404    # amplitude of steep (cliff) sigmoid
_DS_S1 = 1.8496     # steepness of cliff sigmoid
_DS_C1 = 11.2346    # inflection of cliff sigmoid
_DS_B2 = 11.9955    # amplitude of gradual sigmoid
_DS_S2 = 0.2338     # steepness of gradual sigmoid
_DS_C2 = 16.8435    # inflection of gradual sigmoid


def _sigmoid(x: float) -> float:
    """Standard sigmoid function, clamped to avoid overflow."""
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _psa_curve(grade: float) -> float:
    """Evaluate the double-sigmoid PSA grade->multiplier curve."""
    return (
        _DS_A
        + _DS_B1 * _sigmoid(_DS_S1 * (grade - _DS_C1))
        + _DS_B2 * _sigmoid(_DS_S2 * (grade - _DS_C2))
    )


def grade_to_multiplier(
    grade: float,
    authority: str = "PSA",
    raw_nm_price: float | None = None,
) -> float:
    """Convert a continuous numeric grade to a price multiplier.

    Parameters
    ----------
    grade : float
        Numeric grade from 1.0 to 10.0 (supports half-grades like 8.5).
    authority : str
        Grading authority: "PSA", "BGS", or "CGC". BGS/CGC grades are
        normalized to PSA-equivalent before applying the curve.
    raw_nm_price : float | None
        If provided, applies value-tier scaling to the graded premium.

    Returns
    -------
    float
        Price multiplier relative to raw NM = 1.0.

    Examples
    --------
    >>> grade_to_multiplier(10.0)        # PSA 10 -> ~3.50x
    3.499
    >>> grade_to_multiplier(9.0)         # PSA 9  -> ~1.81x
    1.807
    >>> grade_to_multiplier(8.5)         # PSA 8.5 -> ~1.48x
    1.483
    >>> grade_to_multiplier(9.5, "BGS")  # BGS 9.5 -> ~PSA 10
    3.499
    >>> grade_to_multiplier(10, "CGC")   # CGC 10 -> ~PSA 9.7
    2.557
    """
    grade = max(1.0, min(10.0, float(grade)))

    # Normalize non-PSA grades to PSA-equivalent scale
    # BGS 9.5 ~ PSA 10, BGS 10 Black Label is above the curve
    # CGC grades run ~0.3 below PSA equivalent in market perception
    if authority.upper() == "BGS":
        if grade >= 10.0:
            # BGS 10 Black Label -- off the curve, use lookup
            mult = CONDITION_MULTIPLIERS.get("BGS 10", 8.0)
        else:
            # BGS 9.5 ~ PSA 10, BGS 9 ~ PSA 9.5
            grade = min(10.0, grade + 0.5)
            mult = _psa_curve(grade)
    elif authority.upper() == "CGC":
        # CGC grades perceived ~0.3 below PSA; CGC 10 ~ PSA 9.7
        grade = min(10.0, grade - 0.3)
        mult = _psa_curve(grade)
    else:
        # PSA (default)
        mult = _psa_curve(grade)

    # Apply value-tier scaling to the premium portion (above 1.0)
    if raw_nm_price is not None and mult > 1.0:
        scale = get_value_tier_scale(raw_nm_price)
        mult = 1.0 + (mult - 1.0) * scale

    return round(mult, 3)

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
    raw_nm_price: float | None = None,
) -> tuple[str, float]:
    """Return (resolved_key, multiplier) for the given condition/grade.

    Parameters
    ----------
    raw_nm_price : float | None
        If provided and the card is graded, applies value-tier scaling
        to adjust the premium up or down based on the card's base value.
    """
    table = custom_multipliers if custom_multipliers else CONDITION_MULTIPLIERS
    key = _resolve_condition_key(condition, grade_authority, grade)
    mult = table.get(key, 1.0)

    # Apply value-tier scaling for graded cards (raw conditions are uniform)
    if raw_nm_price is not None and mult > 1.0 and grade_authority:
        scale = get_value_tier_scale(raw_nm_price)
        # Rescale only the premium above 1.0
        mult = 1.0 + (mult - 1.0) * scale

    return key, round(mult, 3)


def get_multiplier_with_ci(
    condition: str | None = None,
    grade_authority: str | None = None,
    grade: str | None = None,
    raw_nm_price: float | None = None,
) -> tuple[str, float, float, float]:
    """Return (key, multiplier, ci_low, ci_high) with confidence interval.

    The CI represents an 80% confidence interval around the multiplier,
    reflecting observed market variance across card types and eras.
    """
    key = _resolve_condition_key(condition, grade_authority, grade)
    mult, ci_low, ci_high = CONDITION_MULTIPLIERS_WITH_CI.get(
        key, (1.0, 0.9, 1.1)
    )

    # Apply value-tier scaling for graded cards
    if raw_nm_price is not None and grade_authority:
        scale = get_value_tier_scale(raw_nm_price)
        if mult > 1.0:
            mult = 1.0 + (mult - 1.0) * scale
        if ci_low > 1.0:
            ci_low = 1.0 + (ci_low - 1.0) * scale
        if ci_high > 1.0:
            ci_high = 1.0 + (ci_high - 1.0) * scale

    return key, round(mult, 3), round(ci_low, 3), round(ci_high, 3)


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
