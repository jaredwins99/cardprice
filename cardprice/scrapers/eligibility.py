"""Scrape-eligibility filter for the daily TCGPlayer scraper.

The scraper's daily budget runs ~5.5 hr for 5000 products via Playwright.
Most of the 20k catalog is sub-dollar bulk that turns over slowly and
provides little decision-useful price signal.  This module computes the
subset of `tcg_product_id`s that are *worth* scraping today, based on
recent sale history in `data/tcgplayer_sales.db`.

Filter rule (user-specified, final):
    Exclude bulk. NM is the price ceiling -- a card trading only as NM
    near $1 is functionally bulk because most copies aren't NM. But a
    card trading only as NM at $5+ (multiple times) is genuinely valued.
    So we use a DUAL THRESHOLD:
      - Non-NM: just 1 sale above $1 is enough (LP/MP/HP/DMG above $1
        proves real value across the condition range).
      - NM:     2 sales above $5 needed (higher count AND higher dollar
        amount, because NM-only doesn't generalize to played copies).
    Include iff ANY of:
      (a1) >= `non_nm_min_sales` non-NM sales > `$non_nm_min_price`,
      (a2) >= `nm_min_sales` NM sales > `$nm_min_price`,
      (b)  MAX(market_price) across subtypes > `$chase_market_price`
           (chase-card escape for rare-trading high-value cards), OR
      (c)  zero sales in window (no data, give it a chance).

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
    non_nm_min_sales: int = 1,
    non_nm_min_price: float = 1.0,
    nm_min_sales: int = 2,
    nm_min_price: float = 5.0,
    chase_market_price: float = 5.0,
) -> set[int]:
    """Return scrape-eligible tcg_product_ids per the dual-threshold rule.

    Eligible iff (within `lookback_days`) ANY of:
      (a1) >= `non_nm_min_sales` sales > `$non_nm_min_price` in any
           non-NM condition (LP/MP/HP/DMG),
      (a2) >= `nm_min_sales` NM sales > `$nm_min_price`,
      (b)  MAX(market_price) across subtypes > `$chase_market_price`,
      (c)  zero sales in window.

    The dual threshold catches both kinds of valuable cards: ones with
    real played-copy value (non-NM evidence) and ones that trade only
    as preserved NM but at higher dollar amounts.
    Reads SQLite (sales) + Postgres (market_price).
    """
    own_conn = False
    if conn is None:
        if not _SALES_DB.exists():
            return set()
        conn = sqlite3.connect(f"file:{_SALES_DB}?mode=ro", uri=True)
        own_conn = True

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        # (a1) Non-NM condition has >= non_nm_min_sales sales strictly
        # above non_nm_min_price (default: 1 LP-or-worse sale >$1).
        non_nm_qual = {int(r[0]) for r in conn.execute(
            """
            SELECT tcg_product_id FROM tcgplayer_sales
            WHERE sale_price > ?
              AND sale_date >= ?
              AND condition <> 'Near Mint'
            GROUP BY tcg_product_id
            HAVING COUNT(*) >= ?
            """,
            (non_nm_min_price, cutoff, non_nm_min_sales),
        )}

        # (a2) NM has >= nm_min_sales sales strictly above nm_min_price
        # (default: 2 NM sales >$5). Catches genuinely valuable cards that
        # trade only in preserved NM but at meaningful dollar amounts.
        nm_qual = {int(r[0]) for r in conn.execute(
            """
            SELECT tcg_product_id FROM tcgplayer_sales
            WHERE sale_price > ?
              AND sale_date >= ?
              AND condition = 'Near Mint'
            GROUP BY tcg_product_id
            HAVING COUNT(*) >= ?
            """,
            (nm_min_price, cutoff, nm_min_sales),
        )}

        any_qual = non_nm_qual | nm_qual

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
                # Chase-card escape: MAX(market_price) across subtypes
                # > chase_market_price ($5 default). Higher threshold than
                # min_price ($1) because clause (a) already catches anything
                # with LP/MP/HP/DMG sales above $1. Clause (b) is here only
                # to keep rare-trading high-value cards (NM-only sellers like
                # $2400 Treecko Star) eligible. At >$5 a card's LP would
                # normally clear $1 too, so anything in this clause but not
                # in (a) is a card with sparse recent sales -- exactly the
                # case we want to still track.
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
                          ) > :chase_market_price
                        """
                    ),
                    {"chase_market_price": chase_market_price},
                ).fetchall()
                market_qual = {int(r[0]) for r in rows}
            finally:
                session.close()
        except Exception as e:
            logger.warning("market_price clause failed (%s); skipping clause (b)", e)

        result = any_qual | market_qual | no_history
        logger.debug(
            "eligibility: non_nm_qual=%d, nm_qual=%d, market_qual=%d, "
            "no_history=%d, total=%d",
            len(non_nm_qual), len(nm_qual), len(market_qual),
            len(no_history), len(result),
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
