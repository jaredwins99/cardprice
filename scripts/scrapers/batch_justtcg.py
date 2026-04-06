#!/usr/bin/env python3
"""Batch fetch JustTCG per-condition pricing for the card catalog.

Pulls product IDs from PostgreSQL ordered by market price (highest first),
batches them 20 at a time via JustTCG API, stores to data/justtcg_prices.db.

Free tier: 100 requests/day, 10/min, 20 cards/batch = 2,000 cards/day.

Usage:
    export JUSTTCG_API_KEY=tcg_...
    python scripts/scrapers/batch_justtcg.py                # default 2000 cards (100 batches)
    python scripts/scrapers/batch_justtcg.py --limit 500    # fewer cards
    python scripts/scrapers/batch_justtcg.py --resume       # skip already-fetched cards
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text as sa_text

from cardprice.db.session import SessionLocal
from cardprice.scrapers.justtcg_prices import (
    JustTCGClient,
    get_db,
    DB_PATH,
    print_price_breakdown,
)

log = logging.getLogger("batch_justtcg")


def get_product_ids_by_value(limit: int = 2000, skip_fetched: bool = False) -> list[int]:
    """Get product IDs ordered by market price DESC."""
    session = SessionLocal()
    try:
        rows = session.execute(sa_text("""
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

    all_ids = [(r[0], float(r[1])) for r in rows]

    if skip_fetched and DB_PATH.exists():
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        try:
            already = {r[0] for r in conn.execute(
                "SELECT DISTINCT tcg_product_id FROM justtcg_prices"
            ).fetchall()}
        except Exception:
            already = set()
        finally:
            conn.close()
        before = len(all_ids)
        all_ids = [(pid, price) for pid, price in all_ids if pid not in already]
        log.info("Skipping %d already-fetched, %d remaining", before - len(all_ids), len(all_ids))

    selected = all_ids[:limit]
    if selected:
        log.info(
            "Selected %d products, price range $%.2f - $%.2f",
            len(selected), selected[0][1], selected[-1][1],
        )
    return [pid for pid, _ in selected]


def main():
    parser = argparse.ArgumentParser(description="Batch JustTCG price fetch")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max cards to fetch (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Cards per API request (default: 20, max 20 on free)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cards already in justtcg_prices.db")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    log.info("=== JustTCG batch fetch starting ===")

    product_ids = get_product_ids_by_value(limit=args.limit, skip_fetched=args.resume)
    if not product_ids:
        log.info("No products to fetch")
        return

    client = JustTCGClient()
    db = get_db()

    total_variants = 0
    total_batches = 0
    errors = 0

    # Process in batches
    for i in range(0, len(product_ids), args.batch_size):
        batch = product_ids[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_expected = (len(product_ids) + args.batch_size - 1) // args.batch_size

        try:
            n = client.fetch_and_store_batch(batch, db=db)
            total_variants += n
            total_batches += 1
            log.info(
                "Batch %d/%d: %d cards -> %d variants (quota: %s monthly, %s daily)",
                batch_num, total_expected, len(batch), n,
                client.requests_remaining, client.daily_remaining,
            )

            # Check if we're about to hit quota
            if client.daily_remaining is not None and client.daily_remaining <= 1:
                log.warning("Daily quota exhausted, stopping")
                break
            if client.requests_remaining is not None and client.requests_remaining <= 1:
                log.warning("Monthly quota exhausted, stopping")
                break

        except Exception as e:
            errors += 1
            log.error("Batch %d failed: %s", batch_num, e)
            if errors >= 5:
                log.error("Too many errors, stopping")
                break
            time.sleep(10)  # back off on errors

    db.close()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info("=== JustTCG batch fetch complete ===")
    log.info("  Batches: %d (%d errors)", total_batches, errors)
    log.info("  Variants stored: %d", total_variants)
    log.info("  Cards covered: %d / %d", min(total_batches * args.batch_size, len(product_ids)), len(product_ids))
    log.info("  Elapsed: %.0fs", elapsed)
    log.info("  Quota remaining: %s monthly, %s daily",
             client.requests_remaining, client.daily_remaining)


if __name__ == "__main__":
    main()
