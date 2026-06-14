"""Scrape-eligibility filter for the daily TCGPlayer scraper.

The scraper's daily budget runs ~5.5 hr for 5000 products via Playwright.
Most of the 20k catalog is sub-dollar bulk that turns over slowly and
provides little decision-useful price signal.  This module computes the
subset of `tcg_product_id`s that are *worth* scraping today, based on
recent sale history in `data/tcgplayer_sales.db`.

Filter rule (user-specified, final):
    Exclude bulk. A card is bulk iff it's effectively sub-$1 in every
    observed signal. Include iff ANY of:
      (a) >= `min_lp_sales` sales strictly above `$min_price` in any
          condition within `lookback_days` (the card has trades >$1),
      (b) the product's TCGCSV-aggregate market_price > `$min_price`
          in Postgres `fact_market_prices` (catches high-value chase
          cards that trade rarely; without this clause, a $2400 Treecko
          Star that hasn't traded in 90 days gets wrongly excluded), OR
      (c) the product has zero sales in the entire window (no data to
          filter against; give it a chance).
    NM-at-exactly-$1 cards are excluded because the threshold is
    strict-greater-than (a card "at $1 NM but never hitting $1 LP" is
    bulk by the user's definition).

Plus a new-release grace window: products whose set's release_date is
within `grace_days` days from the Postgres `dim_sets` table are always
eligible, regardless of history.  This avoids permanently excluding
newly-released cards that haven't had time to accumulate sales.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SALES_DB = Path(__file__).resolve().parents[2] / "data" / "tcgplayer_sales.db"


def eligible_product_ids(
    conn: sqlite3.Connection | None = None,
    lookback_days: int = 90,
    min_lp_sales: int = 2,
    min_price: float = 1.0,
) -> set[int]:
    """Return scrape-eligible tcg_product_ids per the final rule.

    Eligible iff (within `lookback_days`):
      (a) ANY condition has >= `min_lp_sales` sales strictly above
          `$min_price` (the card has trades >$1), OR
      (b) TCGCSV-aggregate market_price > `$min_price` (catches chase
          cards that trade rarely; pulled from Postgres), OR
      (c) zero sales in window (no data to filter against; give it a
          chance).

    "Bulk" cards (no sales >$1 anywhere, no market_price >$1, but at
    least one sub-$1 sale) are excluded. Threshold is strict-greater-
    than. Reads SQLite (sales) + Postgres (market_price).
    """
    own_conn = False
    if conn is None:
        if not _SALES_DB.exists():
            return set()
        conn = sqlite3.connect(f"file:{_SALES_DB}?mode=ro", uri=True)
        own_conn = True

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        # (a) ANY condition has >= min_lp_sales sales strictly above min_price.
        any_qual = {int(r[0]) for r in conn.execute(
            """
            SELECT tcg_product_id FROM tcgplayer_sales
            WHERE sale_price > ?
              AND sale_date >= ?
            GROUP BY tcg_product_id
            HAVING COUNT(*) >= ?
            """,
            (min_price, cutoff, min_lp_sales),
        )}

        # (c) zero-history products: in scrape_log but no sales in window.
        ever_seen = {int(r[0]) for r in conn.execute(
            "SELECT DISTINCT tcg_product_id FROM scrape_log"
        )}
        any_sales_in_window = {int(r[0]) for r in conn.execute(
            "SELECT DISTINCT tcg_product_id FROM tcgplayer_sales WHERE sale_date >= ?",
            (cutoff,),
        )}
        no_history = ever_seen - any_sales_in_window

        # (b) TCGCSV market_price > min_price (Postgres).
        market_qual: set[int] = set()
        try:
            from sqlalchemy import text as sa_text
            from cardprice.db.session import SessionLocal
            session = SessionLocal()
            try:
                # Use the MAX market_price across all subtypes for each
                # tcg_product_id (cards have Normal + Reverse Holofoil etc.
                # rows; the rare/holo printing is what actually drives the
                # "is this card worth >$1" decision).
                rows = session.execute(
                    sa_text(
                        """
                        SELECT DISTINCT dc.tcg_product_id
                        FROM dim_cards dc
                        WHERE dc.tcg_product_id IS NOT NULL
                          AND (
                            SELECT MAX(f.market_price)
                            FROM fact_market_prices f
                            WHERE f.tcg_product_id = dc.tcg_product_id
                              AND f.market_price IS NOT NULL
                          ) > :min_price
                        """
                    ),
                    {"min_price": min_price},
                ).fetchall()
                market_qual = {int(r[0]) for r in rows}
            finally:
                session.close()
        except Exception as e:
            logger.warning("market_price clause failed (%s); skipping clause (b)", e)

        result = any_qual | market_qual | no_history
        logger.debug(
            "eligibility: any_qual=%d, market_qual=%d, no_history=%d, total=%d",
            len(any_qual), len(market_qual), len(no_history), len(result),
        )
        return result
    finally:
        if own_conn:
            conn.close()


def new_release_product_ids(grace_days: int = 90) -> set[int]:
    """Return tcg_product_ids belonging to sets released within the last
    `grace_days` days.  Grants newly-released products a free pass past
    the sales-history filter.

    Reads from Postgres `dim_cards` joined to `dim_sets.release_date`.
    Lazy-imports SQLAlchemy so the helper stays cheap to import in
    contexts that don't touch Postgres.
    """
    from sqlalchemy import text as sa_text

    from cardprice.db.session import SessionLocal

    session = SessionLocal()
    try:
        rows = session.execute(
            sa_text(
                """
                SELECT DISTINCT dc.tcg_product_id
                FROM dim_cards dc
                JOIN dim_sets ds ON ds.set_id = dc.set_id
                WHERE dc.tcg_product_id IS NOT NULL
                  AND ds.release_date IS NOT NULL
                  AND ds.release_date >= CURRENT_DATE - (:grace || ' days')::interval
                """
            ),
            {"grace": str(grace_days)},
        ).fetchall()
    finally:
        session.close()

    return {int(r[0]) for r in rows}
