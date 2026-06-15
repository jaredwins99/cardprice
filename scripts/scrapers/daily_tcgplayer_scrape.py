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
from cardprice.scrapers.eligibility import (
    eligible_product_ids,
    new_release_product_ids,
)
from cardprice.scrapers.tcgplayer_sales import DB_PATH, scrape_batch, _get_db
from cardprice.scrapers.velocity import (
    compute_velocity,
    get_last_scraped_at,
    select_by_velocity,
    EXPECTED_SALES_THRESHOLD,
)

log = logging.getLogger("daily_tcgplayer_scrape")


def get_product_ids(
    limit: int = 500,
    threshold: float = EXPECTED_SALES_THRESHOLD,
    *,
    use_filter: bool = True,
    lookback_days: int = 90,
    non_nm_min_sales: int = 1,
    non_nm_min_price: float = 1.0,
    nm_min_sales: int = 2,
    nm_min_price: float = 5.0,
    chase_market_price: float = 5.0,
    grace_days: int = 90,
) -> list[int]:
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
    pre_filter_count = len(candidates)

    # Apply the bulk filter (unless explicitly disabled). Dual threshold:
    # non-NM sales >$1 (1+), or NM sales >$5 (2+), or market_price >$5
    # chase escape, or zero history. See cardprice.scrapers.eligibility.
    if use_filter:
        eligible = eligible_product_ids(
            lookback_days=lookback_days,
            non_nm_min_sales=non_nm_min_sales,
            non_nm_min_price=non_nm_min_price,
            nm_min_sales=nm_min_sales,
            nm_min_price=nm_min_price,
            chase_market_price=chase_market_price,
        )
        grace = new_release_product_ids(grace_days=grace_days)
        keep = eligible | grace
        candidates = [(pid, price) for pid, price in candidates if pid in keep]
        log.info(
            "Eligibility filter: pre=%d, eligible_by_sales=%d, "
            "new_release_grace=%d, post=%d "
            "(lookback=%dd, non_nm>=%d@$%.2f, nm>=%d@$%.2f, chase_market>$%.2f, grace=%dd)",
            pre_filter_count, len(eligible), len(grace), len(candidates),
            lookback_days, non_nm_min_sales, non_nm_min_price,
            nm_min_sales, nm_min_price, chase_market_price, grace_days,
        )
    else:
        log.info("Eligibility filter DISABLED (--no-filter); candidates=%d", pre_filter_count)

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
    parser.add_argument("--lookback-days", type=int, default=90,
                        help="Sales lookback window for eligibility filter (default: 90)")
    parser.add_argument("--non-nm-min-sales", type=int, default=1,
                        help="Minimum non-NM sales above non-NM-min-price (default: 1)")
    parser.add_argument("--non-nm-min-price", type=float, default=1.0,
                        help="Minimum non-NM sale price to count (default: 1.0)")
    parser.add_argument("--nm-min-sales", type=int, default=2,
                        help="Minimum NM sales above nm-min-price (default: 2)")
    parser.add_argument("--nm-min-price", type=float, default=5.0,
                        help="Minimum NM sale price to count (default: 5.0)")
    parser.add_argument("--chase-market-price", type=float, default=5.0,
                        help="MAX(market_price)>this is the chase-card escape (default: 5.0)")
    parser.add_argument("--grace-days", type=int, default=90,
                        help="New-release grace window in days (default: 90)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable the sub-dollar bulk filter (legacy behaviour)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    log.info("=== Daily TCGPlayer scrape starting ===")

    product_ids = get_product_ids(
        limit=args.limit,
        threshold=args.threshold,
        use_filter=not args.no_filter,
        lookback_days=args.lookback_days,
        non_nm_min_sales=args.non_nm_min_sales,
        non_nm_min_price=args.non_nm_min_price,
        nm_min_sales=args.nm_min_sales,
        nm_min_price=args.nm_min_price,
        chase_market_price=args.chase_market_price,
        grace_days=args.grace_days,
    )
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
