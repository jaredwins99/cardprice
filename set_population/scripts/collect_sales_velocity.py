#!/usr/bin/env python3
"""Compute per-set TCGPlayer sales velocity from the existing sales database.

Part of the set_population sub-project. ENDOGENOUS popularity / market-liquidity
proxy: sales volume is partly driven by population, so this is used as a
VALIDATION cross-check downstream, NOT as the primary divisor.

Joins individual sales (SQLite `data/tcgplayer_sales.db`) to Pokemon sets via
    tcgplayer_sales.tcg_product_id -> dim_cards.tcg_product_id -> dim_cards.set_id
    -> dim_sets.{name,release_date}
and emits per-set aggregates to set_population/data/set_sales_velocity.json.

Read-only on both databases. stdlib + sqlite3 + psycopg2 only.
"""

import argparse
import json
import os
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2

# --- paths -------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SALES_DB = os.path.join(REPO_ROOT, "data", "tcgplayer_sales.db")
SUBPROJ_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_OUT = os.path.abspath(os.path.join(SUBPROJ_DATA, "set_sales_velocity.json"))
PG_DSN = "dbname=cardprice"  # peer auth via unix socket


def parse_iso_date(s):
    """Parse an ISO-8601 sale_date string to a date. Tolerant of the
    '+00:00' / fractional-second forms seen in the sales DB."""
    if not s:
        return None
    # Normalize 'Z' and rely on fromisoformat for the rest.
    s = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Last resort: take the leading YYYY-MM-DD.
        try:
            return datetime.fromisoformat(s[:10])
        except ValueError:
            return None


def load_product_to_set(verbose=False):
    """Return (product_id -> set_id) and (set_id -> {name, release_date,
    distinct_products_in_set}). Built from Postgres dim_cards / dim_sets.

    A handful of tcg_product_ids legitimately map to >1 set in dim_cards; we
    pick the lexicographically-first set_id deterministically (count is ~2, so
    the choice is immaterial to the aggregates)."""
    conn = psycopg2.connect(PG_DSN)
    try:
        cur = conn.cursor()
        # set metadata
        cur.execute("SELECT set_id, name, release_date FROM dim_sets;")
        set_meta = {}
        for set_id, name, release_date in cur.fetchall():
            set_meta[set_id] = {
                "set_name": name,
                "release_date": release_date.isoformat() if release_date else None,
            }
        # product -> set, taking first set_id per product deterministically
        cur.execute(
            "SELECT tcg_product_id, MIN(set_id) "
            "FROM dim_cards "
            "WHERE tcg_product_id IS NOT NULL AND set_id IS NOT NULL "
            "GROUP BY tcg_product_id;"
        )
        product_to_set = {pid: set_id for pid, set_id in cur.fetchall()}
        # distinct products per set (denominator) — count cards with a product id
        cur.execute(
            "SELECT set_id, COUNT(DISTINCT tcg_product_id) "
            "FROM dim_cards "
            "WHERE set_id IS NOT NULL AND tcg_product_id IS NOT NULL "
            "GROUP BY set_id;"
        )
        distinct_in_set = dict(cur.fetchall())
        for set_id, n in distinct_in_set.items():
            if set_id in set_meta:
                set_meta[set_id]["distinct_products_in_set"] = n
    finally:
        conn.close()
    if verbose:
        print(f"  dim_sets: {len(set_meta)} sets")
        print(f"  product->set map: {len(product_to_set)} products")
    return product_to_set, set_meta


def collect(sales_db, verbose=False):
    product_to_set, set_meta = load_product_to_set(verbose=verbose)

    conn = sqlite3.connect(f"file:{sales_db}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT tcg_product_id, sale_date, sale_price, quantity FROM tcgplayer_sales;"
    )

    # per-set accumulators
    total_sales = defaultdict(int)           # sum of quantity
    products_with_sales = defaultdict(set)   # distinct product ids with a sale
    min_date = defaultdict(lambda: None)
    max_date = defaultdict(lambda: None)
    prices = defaultdict(list)               # individual sale_price values
    gmv = defaultdict(float)                 # sum(sale_price * quantity)

    orphan_pids = set()
    global_min = None
    global_max = None

    for pid, sale_date, sale_price, qty in cur:
        qty = qty if qty is not None else 1
        dt = parse_iso_date(sale_date)
        if dt is not None:
            if global_min is None or dt < global_min:
                global_min = dt
            if global_max is None or dt > global_max:
                global_max = dt

        set_id = product_to_set.get(pid)
        if set_id is None:
            orphan_pids.add(pid)
            continue

        total_sales[set_id] += qty
        products_with_sales[set_id].add(pid)
        if sale_price is not None:
            prices[set_id].append(sale_price)
            gmv[set_id] += sale_price * qty
        if dt is not None:
            if min_date[set_id] is None or dt < min_date[set_id]:
                min_date[set_id] = dt
            if max_date[set_id] is None or dt > max_date[set_id]:
                max_date[set_id] = dt

    conn.close()

    sets_out = {}
    for set_id in sorted(set(total_sales) | set(products_with_sales)):
        meta = set_meta.get(set_id, {})
        denom = meta.get("distinct_products_in_set", 0)
        # months spanned by THIS set's sales (>= a small floor so a single-day
        # set isn't divided by ~0).
        if min_date[set_id] and max_date[set_id]:
            span_days = (max_date[set_id] - min_date[set_id]).days
            months = max(span_days / 30.4375, 1.0 / 30.4375)
        else:
            months = 1.0
        ts = total_sales[set_id]
        spm = ts / months
        spp = (ts / denom) if denom else 0.0
        cov = (len(products_with_sales[set_id]) / denom) if denom else 0.0
        plist = prices[set_id]
        med = round(statistics.median(plist), 2) if plist else 0.0

        sets_out[set_id] = {
            "set_name": meta.get("set_name"),
            "release_date": meta.get("release_date"),
            "total_sales": ts,
            "sales_per_month": round(spm, 3),
            "distinct_products_with_sales": len(products_with_sales[set_id]),
            "distinct_products_in_set": denom,
            "coverage_ratio": round(cov, 4),
            "sales_per_product": round(spp, 3),
            "median_sale_price": med,
            "total_gmv": round(gmv[set_id], 2),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sales_db_date_range": {
            "min": global_min.isoformat() if global_min else None,
            "max": global_max.isoformat() if global_max else None,
        },
        "orphan_product_ids": len(orphan_pids),
        "sets": sets_out,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sales-db", default=DEFAULT_SALES_DB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print summary but do not write JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.sales_db):
        raise SystemExit(f"sales DB not found: {args.sales_db}")

    result = collect(args.sales_db, verbose=args.verbose)

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out}")

    dr = result["sales_db_date_range"]
    print(f"\nSales date range: {dr['min']}  ->  {dr['max']}")
    print(f"Orphan product_ids (no dim_cards link): {result['orphan_product_ids']}")
    print(f"Sets with sales: {len(result['sets'])}")

    print("\nTop 20 sets by sales_per_month:")
    print(f"{'set_id':<14}{'sales/mo':>10}{'total':>9}{'cov':>7}  set_name")
    ranked = sorted(result["sets"].items(),
                    key=lambda kv: kv[1]["sales_per_month"], reverse=True)
    for set_id, s in ranked[:20]:
        print(f"{set_id:<14}{s['sales_per_month']:>10.1f}{s['total_sales']:>9}"
              f"{s['coverage_ratio']:>7.2f}  {s['set_name']}")


if __name__ == "__main__":
    main()
