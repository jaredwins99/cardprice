#!/usr/bin/env python3
"""Daily TCGPlayer sales scraper.

Pulls high-priority product IDs from PostgreSQL, scrapes their recent sales
via Playwright, and stores results in data/tcgplayer_sales.db.

Usage:
    python scripts/scrapers/daily_tcgplayer_scrape.py              # default 500 cards
    python scripts/scrapers/daily_tcgplayer_scrape.py --limit 1000  # custom batch size
    python scripts/scrapers/daily_tcgplayer_scrape.py --stale-days 3 # re-scrape after 3 days

Designed to be run via cron:
    0 2 * * * cd /home/godli/cardprice && python scripts/scrapers/daily_tcgplayer_scrape.py >> data/logs/tcgplayer_scrape.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text as sa_text

from cardprice.db.session import SessionLocal
from cardprice.scrapers.tcgplayer_sales import DB_PATH, scrape_batch, _get_db
from cardprice.scrapers.velocity import (
    compute_velocity,
    get_last_scraped_at,
    select_by_velocity,
    EXPECTED_SALES_THRESHOLD,
)

log = logging.getLogger("daily_tcgplayer_scrape")


def get_product_ids(limit: int = 500, threshold: float = EXPECTED_SALES_THRESHOLD) -> list[int]:
    """Get product IDs to scrape, prioritized by expected information gain.

    Replaces uniform stale-days rotation with velocity-aware scheduling:

    1. Compute per-product sales-per-day from observed history.
    2. For each product, compute expected_new_sales = velocity * days_since_last_scrape.
    3. Skip if expected_new_sales < threshold (default 1.0).
       Re-scraping a product that's had zero new sales burns Playwright
       time for a no-op UPSERT — exactly what we want to avoid.
    4. Never-scraped products always qualify (unknown velocity).
    5. Rank survivors by expected_new_sales DESC (most overdue first),
       with market_price as tie-breaker.

    This replaces the previous "every card every 7 days uniformly"
    strategy, which overstaffed low-velocity commons and under-sampled
    high-velocity high-value cards.
    """
    # Pull market price for every product from Postgres
    session = SessionLocal()
    try:
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

    candidates = [(int(r[0]), float(r[1])) for r in result]

    # Velocity + last-scraped timestamps from SQLite
    velocity = compute_velocity()
    last_scraped = get_last_scraped_at()

    selected = select_by_velocity(
        candidates,
        velocity=velocity,
        last_scraped=last_scraped,
        limit=limit,
        threshold=threshold,
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily TCGPlayer sales scraper")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max products to scrape per run (default: 500)")
    parser.add_argument("--threshold", type=float, default=EXPECTED_SALES_THRESHOLD,
                        help=("Skip products with fewer than this many new "
                              "sales expected since last scrape (default: 1.0). "
                              "Never-scraped products always qualify."))
    parser.add_argument("--delay-min", type=float, default=2.0,
                        help="Min delay between scrapes in seconds (default: 2.0)")
    parser.add_argument("--delay-max", type=float, default=4.0,
                        help="Max delay between scrapes in seconds (default: 4.0)")
    parser.add_argument("--max-errors", type=int, default=10,
                        help="Stop after N consecutive errors (default: 10)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel browser instances (default: 2)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    log.info("=== Daily TCGPlayer scrape starting ===")

    product_ids = get_product_ids(limit=args.limit, threshold=args.threshold)
    if not product_ids:
        log.info("No products to scrape, exiting")
        return

    log.info("Scraping %d products...", len(product_ids))
    results = scrape_batch(
        product_ids,
        delay_range=(args.delay_min, args.delay_max),
        max_errors=args.max_errors,
        workers=args.workers,
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
