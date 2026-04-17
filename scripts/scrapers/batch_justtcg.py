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


def _velocity_refresh_days(velocity_per_day: float) -> int:
    """Compute how many days to wait before re-fetching JustTCG prices
    for a product with a given sales velocity.

    Hot cards move fast — their condition premiums shift more frequently
    and are worth checking weekly or better.  Slow cards rarely see NM/LP
    price swings so monthly is fine.  Quota budget (~1000/month) forces a
    long floor on the cold tier.
    """
    if velocity_per_day >= 2.0:
        return 3   # hot: ~10 refreshes per month
    if velocity_per_day >= 0.5:
        return 7   # warm: ~4 refreshes per month
    if velocity_per_day >= 0.1:
        return 14  # median: 2/month
    return 30      # cold: 1/month (quota floor)


def _load_last_fetched(game: str) -> dict[int, str]:
    """Return {tcg_product_id: latest ISO fetched_at} for a game.

    Empty dict if DB doesn't exist or table is empty.
    """
    if not DB_PATH.exists():
        return {}
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            """
            SELECT tcg_product_id, MAX(fetched_at) AS last
            FROM justtcg_prices
            WHERE game = ?
            GROUP BY tcg_product_id
            """,
            (game,),
        ).fetchall()
        return {pid: ts for pid, ts in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def get_product_ids_by_value(
    limit: int = 2000,
    refresh_days: int | None = None,
    game: str = "pokemon",
) -> list[int]:
    """Select JustTCG targets with velocity-aware refresh cadence.

    Replaces the old binary --resume behaviour (fetch each product once
    and never again).  Each product gets an individual refresh interval
    based on its TCGPlayer sales velocity: hot cards every 3 days, warm
    every 7, median every 14, cold every 30.  Products whose last fetch
    is older than their interval become candidates.

    Within the candidate set, orders by price*velocity so the 1000/month
    quota gets spent on products whose USD-flow-per-day is highest.

    Parameters
    ----------
    limit : int
        Max products to return this run.
    refresh_days : int | None
        If set, overrides the velocity-based per-product interval with a
        uniform wait.  Useful for initial coverage passes (refresh_days=0
        means "fetch anything not fetched yet").
    game : str
        'pokemon' or 'pokemon-japan'.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

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

    last_fetched = _load_last_fetched(game)
    # Velocity is computed from TCGPlayer sales data, which only exists for
    # English (pokemon). JP products have no velocity signal.
    velocity = compute_velocity() if game == "pokemon" else {}

    # Filter: a product is eligible if never-fetched OR past its refresh interval.
    never_fetched = 0
    due = 0
    skipped_fresh = 0
    eligible: list[tuple[int, float]] = []

    for pid, price in all_ids:
        last = last_fetched.get(pid)
        if last is None:
            never_fetched += 1
            eligible.append((pid, price))
            continue
        # Per-product interval: explicit override, else velocity-derived.
        if refresh_days is not None:
            interval = refresh_days
        else:
            interval = _velocity_refresh_days(velocity.get(pid, 0.0))
        try:
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            days_old = (now - last_dt).total_seconds() / 86400.0
        except Exception:
            days_old = 9e9  # unparseable → treat as very stale
        if days_old >= interval:
            due += 1
            eligible.append((pid, price))
        else:
            skipped_fresh += 1

    log.info(
        "[%s] refresh eligibility: never_fetched=%d, due=%d, skipped_fresh=%d",
        game, never_fetched, due, skipped_fresh,
    )

    # Value-weighted velocity priority: spend quota where USD-flow is highest.
    if game == "pokemon":
        scored = value_weighted_priority(eligible, velocity=velocity)
        selected_pairs = scored[:limit]
        if selected_pairs:
            log.info(
                "[%s] Selected %d by price*velocity, score range $%.3f - $%.3f",
                game, len(selected_pairs), selected_pairs[0][1], selected_pairs[-1][1],
            )
        return [pid for pid, _ in selected_pairs]

    selected = eligible[:limit]
    if selected:
        log.info(
            "[%s] Selected %d products, price range $%.2f - $%.2f",
            game, len(selected), selected[0][1], selected[-1][1],
        )
    return [pid for pid, _ in selected]


def run_pass(
    client,
    db,
    game: str,
    limit: int,
    batch_size: int,
    refresh_days: int | None,
) -> tuple[int, int, int]:
    """Run one full scrape pass for a game. Returns (batches, variants, errors)."""
    log.info("=== JustTCG pass starting: game=%s ===", game)
    product_ids = get_product_ids_by_value(
        limit=limit, refresh_days=refresh_days, game=game,
    )
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
                        help=("Deprecated alias for --refresh-days 0. Kept for "
                              "backwards compat with the existing cron entry. "
                              "Prefer --refresh-days."))
    parser.add_argument("--refresh-days", type=int, default=None,
                        help=("Uniform wait between re-fetches per product. "
                              "If unset, uses velocity-aware per-product "
                              "intervals (3/7/14/30 days for hot/warm/"
                              "median/cold)."))
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

    # --resume is the legacy cron flag — treat as "fetch anything not
    # fetched yet" (refresh_days=0 means any gap qualifies).  When
    # --refresh-days is passed explicitly, it wins.  When neither is set,
    # the per-product velocity intervals apply.
    refresh_days = args.refresh_days
    if refresh_days is None and args.resume:
        refresh_days = 0  # backwards-compat: only never-fetched products

    en_b, en_v, en_e = run_pass(
        client, db, "pokemon", args.limit, args.batch_size, refresh_days,
    )

    jp_b = jp_v = jp_e = 0
    quota_left = (
        (client.daily_remaining is None or client.daily_remaining > 1)
        and (client.requests_remaining is None or client.requests_remaining > 1)
    )
    if args.include_jp and quota_left:
        jp_limit = args.jp_limit if args.jp_limit is not None else args.limit
        jp_b, jp_v, jp_e = run_pass(
            client, db, "pokemon-japan", jp_limit, args.batch_size, refresh_days,
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
