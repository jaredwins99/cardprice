#!/usr/bin/env python3
"""Preview the scrape-eligibility filter without actually scraping.

Prints:
  - Total known products in the Postgres catalog
  - Eligible-by-sales (>= N LP sales >= $X in lookback)
  - Eligible-by-new-release-grace (set release_date within grace)
  - Combined eligible total + savings ratio vs the legacy 5000-product run
  - Top 20 cards EXCLUDED by the filter (sanity check we aren't nuking
    anything that matters)
  - Top 20 cards NEWLY ELIGIBLE (mostly the same as before for high-end;
    interesting for the boundary cases)

Usage:
    python scripts/scrapers/preview_scrape_eligibility.py
    python scripts/scrapers/preview_scrape_eligibility.py --lookback-days 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import text as sa_text

from cardprice.db.session import SessionLocal
from cardprice.scrapers.eligibility import (
    eligible_product_ids,
    new_release_product_ids,
)


def _fetch_catalog() -> list[tuple[int, str, float]]:
    """All products in dim_cards with latest market price (NULL -> 0)."""
    session = SessionLocal()
    try:
        rows = session.execute(sa_text("""
            SELECT dc.tcg_product_id,
                   dc.name || COALESCE(' (' || ds.name || ')', '') AS label,
                   COALESCE(
                       (SELECT fmp.market_price
                        FROM fact_market_prices fmp
                        WHERE fmp.tcg_product_id = dc.tcg_product_id
                        ORDER BY fmp.price_date DESC LIMIT 1),
                       0
                   ) AS latest_price
            FROM dim_cards dc
            LEFT JOIN dim_sets ds ON ds.set_id = dc.set_id
            WHERE dc.tcg_product_id IS NOT NULL
        """)).fetchall()
    finally:
        session.close()
    return [(int(r[0]), str(r[1]), float(r[2])) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--min-lp-sales", type=int, default=2)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--grace-days", type=int, default=90)
    p.add_argument("--legacy-budget", type=int, default=5000,
                   help="Compare savings vs this baseline scrape volume")
    args = p.parse_args()

    catalog = _fetch_catalog()
    catalog_by_pid: dict[int, tuple[str, float]] = {
        pid: (label, price) for pid, label, price in catalog
    }
    total = len(catalog)

    eligible = eligible_product_ids(
        lookback_days=args.lookback_days,
        min_lp_sales=args.min_lp_sales,
        min_price=args.min_price,
    )
    grace = new_release_product_ids(grace_days=args.grace_days)
    eligible_in_catalog = eligible & set(catalog_by_pid)
    grace_in_catalog = grace & set(catalog_by_pid)
    combined = eligible_in_catalog | grace_in_catalog
    excluded = set(catalog_by_pid) - combined

    print(f"Filter params: lookback={args.lookback_days}d, "
          f"min_lp_sales={args.min_lp_sales}, "
          f"min_price=${args.min_price:.2f}, "
          f"grace={args.grace_days}d")
    print(f"")
    print(f"Catalog (dim_cards w/ tcg_product_id):  {total:>7}")
    print(f"Eligible by sales filter:               {len(eligible_in_catalog):>7}")
    print(f"Eligible by new-release grace:          {len(grace_in_catalog):>7}")
    print(f"  (overlap with sales filter):          {len(eligible_in_catalog & grace_in_catalog):>7}")
    print(f"Combined eligible:                      {len(combined):>7}")
    print(f"Excluded:                               {len(excluded):>7}")
    print(f"")
    legacy = args.legacy_budget
    if total:
        ratio = len(combined) / total
        print(f"Eligible / total catalog:               {ratio:>6.1%}")
    if legacy:
        # Savings = how much smaller the candidate pool is vs the legacy
        # cap.  If combined >= legacy, no savings (filter doesn't help vs
        # that cap).  If combined < legacy, we'll naturally scrape fewer.
        if len(combined) < legacy:
            savings = 1.0 - (len(combined) / legacy)
            print(f"Pool size vs legacy {legacy}: -{savings:>5.1%} "
                  f"(scrape {len(combined)} instead of {legacy})")
        else:
            print(f"Pool size vs legacy {legacy}: pool still larger; "
                  f"no per-run savings (still picks top {legacy} by velocity)")

    # Top 20 EXCLUDED by price (these are the ones we'd want to double-check)
    excl_rows = sorted(
        ((catalog_by_pid[pid][1], catalog_by_pid[pid][0], pid) for pid in excluded),
        key=lambda t: t[0], reverse=True,
    )[:20]
    print("\nTop 20 EXCLUDED cards by market price (sanity-check the cut):")
    for price, name, pid in excl_rows:
        print(f"  ${price:>7.2f}  pid={pid:<8}  {name[:80]}")

    # Top 20 NEWLY ELIGIBLE — products that are only in `combined` via the
    # grace window (not the sales filter).  These are the cards we now
    # focus on that we might have otherwise let stagnate.
    newly = grace_in_catalog - eligible_in_catalog
    newly_rows = sorted(
        ((catalog_by_pid[pid][1], catalog_by_pid[pid][0], pid) for pid in newly),
        key=lambda t: t[0], reverse=True,
    )[:20]
    print("\nTop 20 NEWLY-ELIGIBLE-BY-GRACE cards by market price:")
    for price, name, pid in newly_rows:
        print(f"  ${price:>7.2f}  pid={pid:<8}  {name[:80]}")


if __name__ == "__main__":
    main()
