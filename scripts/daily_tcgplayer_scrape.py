#!/usr/bin/env python3
"""Daily TCGPlayer sales scraper.

Pulls high-priority product IDs from PostgreSQL, scrapes their recent sales
via Playwright, and stores results in data/tcgplayer_sales.db.

Usage:
    python scripts/daily_tcgplayer_scrape.py              # default 500 cards
    python scripts/daily_tcgplayer_scrape.py --limit 1000  # custom batch size
    python scripts/daily_tcgplayer_scrape.py --stale-days 3 # re-scrape after 3 days

Designed to be run via cron:
    0 2 * * * cd /home/godli/cardprice && python scripts/daily_tcgplayer_scrape.py >> data/logs/tcgplayer_scrape.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text as sa_text

from cardprice.db.session import SessionLocal
from cardprice.scrapers.tcgplayer_sales import DB_PATH, scrape_batch, _get_db

log = logging.getLogger("daily_tcgplayer_scrape")


def get_product_ids(limit: int = 500, stale_days: int = 7) -> list[int]:
    """Get product IDs to scrape, prioritized by market value.

    Strategy:
    1. Never-scraped products first (ordered by latest market_price DESC)
    2. Stale products (scraped > stale_days ago, ordered by market_price DESC)
    """
    # Get already-scraped product IDs and their last scrape time from SQLite
    scraped: dict[int, str] = {}
    if DB_PATH.exists():
        sconn = sqlite3.connect(str(DB_PATH))
        try:
            rows = sconn.execute(
                "SELECT tcg_product_id, last_scraped FROM scrape_log"
            ).fetchall()
            scraped = {r[0]: r[1] for r in rows}
        except sqlite3.OperationalError:
            pass
        finally:
            sconn.close()

    scraped_ids = set(scraped.keys())

    session = SessionLocal()
    try:
        # Get all product IDs with their latest market price
        result = session.execute(sa_text("""
            SELECT DISTINCT dc.tcg_product_id,
                   COALESCE(
                       (SELECT fmp.market_price
                        FROM fact_market_prices fmp
                        WHERE fmp.tcg_product_id = dc.tcg_product_id
                        ORDER BY fmp.price_date DESC LIMIT 1),
                       0
                   ) AS latest_price
            FROM dim_cards dc
            WHERE dc.tcg_product_id IS NOT NULL
            ORDER BY latest_price DESC
        """)).fetchall()
    finally:
        session.close()

    all_products = [(r[0], float(r[1])) for r in result]

    # Split into never-scraped and stale
    never_scraped = []
    stale = []
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).isoformat()

    for pid, price in all_products:
        if pid not in scraped_ids:
            never_scraped.append(pid)
        elif scraped.get(pid, "") < cutoff:
            stale.append(pid)

    # Prioritize never-scraped, then stale (both already sorted by price DESC)
    candidates = never_scraped + stale
    selected = candidates[:limit]

    log.info(
        "Product selection: %d never-scraped, %d stale (>%dd), selected %d",
        len(never_scraped), len(stale), stale_days, len(selected),
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily TCGPlayer sales scraper")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max products to scrape per run (default: 500)")
    parser.add_argument("--stale-days", type=int, default=7,
                        help="Re-scrape products older than N days (default: 7)")
    parser.add_argument("--delay-min", type=float, default=2.0,
                        help="Min delay between scrapes in seconds (default: 2.0)")
    parser.add_argument("--delay-max", type=float, default=4.0,
                        help="Max delay between scrapes in seconds (default: 4.0)")
    parser.add_argument("--max-errors", type=int, default=10,
                        help="Stop after N consecutive errors (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    log.info("=== Daily TCGPlayer scrape starting ===")

    product_ids = get_product_ids(limit=args.limit, stale_days=args.stale_days)
    if not product_ids:
        log.info("No products to scrape, exiting")
        return

    log.info("Scraping %d products...", len(product_ids))
    results = scrape_batch(
        product_ids,
        delay_range=(args.delay_min, args.delay_max),
        max_errors=args.max_errors,
    )

    # Summary
    success = sum(1 for v in results.values() if v >= 0)
    errors = sum(1 for v in results.values() if v < 0)
    total_sales = sum(v for v in results.values() if v > 0)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    log.info("=== Daily TCGPlayer scrape complete ===")
    log.info("  Products: %d/%d succeeded, %d errors", success, len(product_ids), errors)
    log.info("  Sales stored: %d new records", total_sales)
    log.info("  Elapsed: %.0fs (%.1f products/min)", elapsed, len(product_ids) / (elapsed / 60) if elapsed > 0 else 0)

    # Print top earners for visibility
    conn = _get_db()
    top = conn.execute("""
        SELECT tcg_product_id, COUNT(*) as cnt,
               AVG(sale_price) as avg_price,
               GROUP_CONCAT(DISTINCT condition) as conditions
        FROM tcgplayer_sales
        WHERE scraped_at > datetime('now', '-1 day')
        GROUP BY tcg_product_id
        ORDER BY avg_price DESC
        LIMIT 10
    """).fetchall()
    conn.close()

    if top:
        log.info("  Top cards scraped today:")
        for row in top:
            log.info("    Product %d: %d sales, avg $%.2f, conditions: %s",
                     row[0], row[1], row[2], row[3])


if __name__ == "__main__":
    main()
