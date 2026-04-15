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
    ConsecutiveFailureLimit,
    get_db,
    DB_PATH,
    print_price_breakdown,
)
from cardprice.scrapers.velocity import (
    compute_velocity,
    value_weighted_priority,
)

log = logging.getLogger("batch_justtcg")


def get_product_ids_by_value(
    limit: int = 2000,
    skip_fetched: bool = False,
    game: str = "pokemon",
) -> list[int]:
    """Get product IDs ordered by market price DESC.

    For game='pokemon' uses dim_cards joined with fact_market_prices.
    For game='pokemon-japan' uses dim_cards_jp (no price ordering — JP has
    no fact_market_prices coverage; orders by tcg_product_id).
    """
    session = SessionLocal()
    try:
        if game == "pokemon":
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
            all_ids = [(r[0], float(r[1])) for r in rows]
        elif game == "pokemon-japan":
            rows = session.execute(sa_text("""
                SELECT tcg_product_id
                FROM dim_cards_jp
                WHERE tcg_product_id IS NOT NULL
                ORDER BY tcg_product_id
            """)).fetchall()
            all_ids = [(r[0], 0.0) for r in rows]
        else:
            raise ValueError(f"Unknown game: {game}")
    finally:
        session.close()

    if skip_fetched and DB_PATH.exists():
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        try:
            already = {r[0] for r in conn.execute(
                "SELECT DISTINCT tcg_product_id FROM justtcg_prices WHERE game = ?",
                (game,),
            ).fetchall()}
        except Exception:
            already = set()
        finally:
            conn.close()
        before = len(all_ids)
        all_ids = [(pid, price) for pid, price in all_ids if pid not in already]
        log.info("[%s] Skipping %d already-fetched, %d remaining",
                 game, before - len(all_ids), len(all_ids))

    # Value-weighted velocity priority: price * sales-per-day ranks products
    # by how much USD moves per day, i.e. where condition pricing matters most.
    # Free-tier quota is 1000/month, so we want every call spent on a product
    # whose prices are actually changing. For English (pokemon) game we have
    # velocity data; JP has none so we fall back to the original price order.
    if game == "pokemon":
        velocity = compute_velocity()
        scored = value_weighted_priority(all_ids, velocity=velocity)
        selected_pairs = scored[:limit]
        if selected_pairs:
            log.info(
                "[%s] Selected %d by price*velocity, score range $%.3f - $%.3f",
                game, len(selected_pairs), selected_pairs[0][1], selected_pairs[-1][1],
            )
        return [pid for pid, _ in selected_pairs]

    selected = all_ids[:limit]
    if selected:
        log.info(
            "[%s] Selected %d products, price range $%.2f - $%.2f",
            game, len(selected), selected[0][1], selected[-1][1],
        )
    return [pid for pid, _ in selected]


def run_pass(client, db, game: str, limit: int, batch_size: int, resume: bool) -> tuple[int, int, int]:
    """Run one full scrape pass for a game. Returns (batches, variants, errors)."""
    log.info("=== JustTCG pass starting: game=%s ===", game)
    product_ids = get_product_ids_by_value(limit=limit, skip_fetched=resume, game=game)
    if not product_ids:
        log.info("[%s] No products to fetch", game)
        return 0, 0, 0

    total_variants = 0
    total_batches = 0
    errors = 0

    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_expected = (len(product_ids) + batch_size - 1) // batch_size

        try:
            n = client.fetch_and_store_batch(batch, db=db, game=game)
            total_variants += n
            total_batches += 1
            log.info(
                "[%s] Batch %d/%d: %d cards -> %d variants (quota: %s monthly, %s daily)",
                game, batch_num, total_expected, len(batch), n,
                client.requests_remaining, client.daily_remaining,
            )

            if client.daily_remaining is not None and client.daily_remaining <= 1:
                log.warning("[%s] Daily quota exhausted, stopping", game)
                break
            if client.requests_remaining is not None and client.requests_remaining <= 1:
                log.warning("[%s] Monthly quota exhausted, stopping", game)
                break

        except ConsecutiveFailureLimit as e:
            # Hard stop — API is wedged.  Exit the whole pass cleanly so
            # the script doesn't sit in a retry loop for hours.
            log.error("[%s] %s", game, e)
            log.error("[%s] Aborting pass — try again later when API recovers", game)
            break

        except Exception as e:
            errors += 1
            log.error("[%s] Batch %d failed: %s", game, batch_num, e)
            if errors >= 5:
                log.error("[%s] Too many errors, stopping pass", game)
                break
            import time as _t
            _t.sleep(10)

    return total_batches, total_variants, errors


def main():
    parser = argparse.ArgumentParser(description="Batch JustTCG price fetch")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max cards to fetch (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Cards per API request (default: 20, max 20 on free)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cards already in justtcg_prices.db")
    parser.add_argument("--include-jp", action="store_true",
                        help="Also run a Japanese (pokemon-japan) pass against dim_cards_jp")
    parser.add_argument("--jp-limit", type=int, default=None,
                        help="Override --limit for the JP pass (default: same as --limit)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    log.info("=== JustTCG batch fetch starting ===")

    client = JustTCGClient()
    db = get_db()

    en_b, en_v, en_e = run_pass(
        client, db, "pokemon", args.limit, args.batch_size, args.resume,
    )

    jp_b = jp_v = jp_e = 0
    quota_left = (
        (client.daily_remaining is None or client.daily_remaining > 1)
        and (client.requests_remaining is None or client.requests_remaining > 1)
    )
    if args.include_jp and quota_left:
        jp_limit = args.jp_limit if args.jp_limit is not None else args.limit
        jp_b, jp_v, jp_e = run_pass(
            client, db, "pokemon-japan", jp_limit, args.batch_size, args.resume,
        )
    elif args.include_jp:
        log.warning("Skipping JP pass: quota exhausted after EN pass")

    db.close()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    log.info("=== JustTCG batch fetch complete ===")
    log.info("  EN: %d batches, %d variants, %d errors", en_b, en_v, en_e)
    if args.include_jp:
        log.info("  JP: %d batches, %d variants, %d errors", jp_b, jp_v, jp_e)
    log.info("  Elapsed: %.0fs", elapsed)
    log.info("  Quota remaining: %s monthly, %s daily",
             client.requests_remaining, client.daily_remaining)


if __name__ == "__main__":
    main()
