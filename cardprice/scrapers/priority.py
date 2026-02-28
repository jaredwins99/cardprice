"""Build and maintain the card_scrape_priority table.

Ranks cards by a composite priority score so the scraper focuses on
the most valuable / volatile / relevant cards first.

Score components (all normalized 0-1, then weighted):
  - market_price   : higher price  => higher priority   (weight 0.35)
  - volatility     : stddev of market_price over last 30 days (weight 0.30)
  - rarity         : rarer cards scored higher           (weight 0.20)
  - set_recency    : newer sets scored higher            (weight 0.15)
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from cardprice.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Rarity tiers — higher number = rarer = more priority
RARITY_TIERS = {
    "Common": 0.1,
    "Uncommon": 0.2,
    "Rare": 0.4,
    "Rare Holo": 0.5,
    "Rare Holo EX": 0.6,
    "Rare Holo GX": 0.6,
    "Rare Holo V": 0.6,
    "Rare Holo VMAX": 0.7,
    "Rare Holo VSTAR": 0.7,
    "Rare Ultra": 0.75,
    "Rare Secret": 0.8,
    "Rare Rainbow": 0.85,
    "Rare Shiny": 0.8,
    "Rare ACE": 0.7,
    "Illustration Rare": 0.85,
    "Special Illustration Rare": 0.9,
    "Hyper Rare": 0.95,
    "Double Rare": 0.7,
    "Ultra Rare": 0.8,
    "Shiny Rare": 0.75,
    "Shiny Ultra Rare": 0.9,
    "ACE SPEC Rare": 0.7,
    "Promo": 0.5,
}
DEFAULT_RARITY_SCORE = 0.3


# The big query: one pass to get latest price, 30-day volatility, rarity, and
# set release_date for every card that has at least one market price row.
PRIORITY_SQL = text("""
WITH latest_prices AS (
    SELECT DISTINCT ON (fmp.card_id)
        fmp.card_id,
        fmp.market_price
    FROM fact_market_prices fmp
    WHERE fmp.card_id IS NOT NULL
      AND fmp.market_price IS NOT NULL
    ORDER BY fmp.card_id, fmp.price_date DESC
),
volatility AS (
    SELECT
        fmp.card_id,
        STDDEV(fmp.market_price) AS price_stddev,
        AVG(fmp.market_price)    AS price_avg
    FROM fact_market_prices fmp
    WHERE fmp.card_id IS NOT NULL
      AND fmp.market_price IS NOT NULL
      AND fmp.price_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY fmp.card_id
),
combined AS (
    SELECT
        lp.card_id,
        lp.market_price,
        COALESCE(v.price_stddev, 0)  AS price_stddev,
        COALESCE(v.price_avg, lp.market_price) AS price_avg,
        dc.rarity,
        ds.release_date
    FROM latest_prices lp
    JOIN dim_cards dc ON dc.card_id = lp.card_id
    LEFT JOIN dim_sets ds ON ds.set_id = dc.set_id
    LEFT JOIN volatility v ON v.card_id = lp.card_id
)
SELECT
    card_id,
    market_price,
    price_stddev,
    price_avg,
    rarity,
    release_date
FROM combined
ORDER BY market_price DESC NULLS LAST
LIMIT :top_n
""")

UPSERT_SQL = text("""
INSERT INTO card_scrape_priority (card_id, priority_score, last_scraped, scrape_count)
VALUES (:card_id, :score, NULL, 0)
ON CONFLICT (card_id) DO UPDATE
    SET priority_score = EXCLUDED.priority_score
""")


def _normalize(value: float, max_val: float) -> float:
    """Normalize a value to 0-1 range given the max in the batch."""
    if max_val <= 0:
        return 0.0
    return min(value / max_val, 1.0)


def _rarity_score(rarity: str | None) -> float:
    if rarity is None:
        return DEFAULT_RARITY_SCORE
    return RARITY_TIERS.get(rarity, DEFAULT_RARITY_SCORE)


def _recency_score(release_date, min_date, max_date) -> float:
    """Score 0-1 based on how recent the set is relative to the range."""
    if release_date is None or min_date is None or max_date is None:
        return 0.3  # neutral default
    span = (max_date - min_date).days
    if span <= 0:
        return 1.0
    days_from_min = (release_date - min_date).days
    return days_from_min / span


def build_priority_queue(session, top_n: int = 2000) -> dict:
    """Query market data and upsert composite priority scores.

    Returns a summary dict with counts and score stats.
    """
    logger.info("Building priority queue (top_n=%d)...", top_n)

    rows = session.execute(PRIORITY_SQL, {"top_n": top_n}).fetchall()
    if not rows:
        logger.warning("No market price data found — priority table not updated.")
        return {"cards_scored": 0, "min_score": None, "max_score": None}

    # Collect maximums for normalization
    max_price = max(float(r.market_price) for r in rows)
    max_stddev = max(float(r.price_stddev) for r in rows) or 1.0

    # Collect release date range
    dates = [r.release_date for r in rows if r.release_date is not None]
    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None

    # Weights
    W_PRICE = 0.35
    W_VOLATILITY = 0.30
    W_RARITY = 0.20
    W_RECENCY = 0.15

    scored = []
    for r in rows:
        price_norm = _normalize(float(r.market_price), max_price)
        vol_norm = _normalize(float(r.price_stddev), max_stddev)
        rarity_norm = _rarity_score(r.rarity)
        recency_norm = _recency_score(r.release_date, min_date, max_date)

        score = (
            W_PRICE * price_norm
            + W_VOLATILITY * vol_norm
            + W_RARITY * rarity_norm
            + W_RECENCY * recency_norm
        )
        scored.append({"card_id": r.card_id, "score": round(score, 6)})

    # Sort by score descending before upserting
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Upsert in batch
    for item in scored:
        session.execute(UPSERT_SQL, item)
    session.commit()

    scores = [s["score"] for s in scored]
    summary = {
        "cards_scored": len(scored),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "avg_score": round(sum(scores) / len(scores), 4),
    }
    logger.info(
        "Priority queue built: %d cards scored (min=%.4f, max=%.4f, avg=%.4f)",
        summary["cards_scored"],
        summary["min_score"],
        summary["max_score"],
        summary["avg_score"],
    )
    return summary
