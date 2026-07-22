#!/usr/bin/env python3
"""Pre-trade live-listings check for specific cards.

Visits the TCGPlayer product pages for the given cards (a handful of page
loads — negligible vs the daily rotation) and prints the cheapest live
listings per (condition, printing), alongside our model fair value and quotes.
Snapshots are also stored in tcgplayer_listings so checks accumulate history.

Usage:
  python3 pricing_model/scripts/check_listings.py ex14-2/normal ex11-110/normal
  python3 pricing_model/scripts/check_listings.py --pids 84188 87768
"""

import argparse
import os
import sys

import pandas as pd
import psycopg2
from playwright.sync_api import sync_playwright

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from cardprice.scrapers.tcgplayer_sales import (  # noqa: E402
    _get_db, _insert_listings, create_browser_context, scrape_product_sales)

OOF = os.path.join(REPO, "pricing_model", "data", "fair_value_oof.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="*", help="card_ids, e.g. ex14-2/normal")
    ap.add_argument("--pids", nargs="*", type=int, default=[])
    args = ap.parse_args()

    pg = psycopg2.connect(dbname="cardprice")
    cur = pg.cursor()
    pids = list(args.pids)
    names = {}
    for cid in args.cards:
        cur.execute("SELECT tcg_product_id, name FROM dim_cards WHERE card_id=%s", (cid,))
        row = cur.fetchone()
        if row and row[0]:
            pids.append(row[0])
            names[row[0]] = f"{row[1]} ({cid})"
    for pid in args.pids:
        cur.execute("SELECT name, card_id FROM dim_cards WHERE tcg_product_id=%s", (pid,))
        row = cur.fetchone()
        names[pid] = f"{row[0]} ({row[1]})" if row else str(pid)
    pg.close()
    if not pids:
        sys.exit("no products resolved")

    fv = None
    if os.path.exists(OOF):
        fv = pd.read_parquet(OOF)

    conn = _get_db()
    with sync_playwright() as pw:
        browser, ctx = create_browser_context(pw)
        page = ctx.new_page()
        for pid in pids:
            buf = []
            try:
                scrape_product_sales(page, pid, capture_listings=buf)
            except Exception as e:
                print(f"### {names.get(pid, pid)}: ERROR {e}")
                continue
            print(f"\n### {names.get(pid, pid)} — {len(buf)} listings captured"
                  + (f", market depth {buf[0]['total_results']}" if buf else ""))
            if buf:
                _insert_listings(conn, pid, buf)
                df = pd.DataFrame(buf)
                df["all_in"] = df["price"] + df["shipping_price"]
                df = df[df["language"].isin(["EN", ""])]
                cheapest = (df.sort_values("all_in")
                            .groupby(["printing", "condition"]).first()
                            .reset_index())
                for r in cheapest.itertuples():
                    line = (f"  {r.printing:18s} {r.condition:18s} "
                            f"${r.all_in:8.2f}  ({r.seller_name}"
                            f"{', direct' if r.direct_seller else ''})")
                    if fv is not None:
                        m = fv[(fv["pid"] == pid) & (fv["printing"] == r.printing)]
                        if len(m):
                            line += (f"   [model FV LP ${m.iloc[0]['fair_value']:.2f}, "
                                     f"label ${m.iloc[0]['label_lp']:.2f}]")
                    print(line)
        browser.close()
    conn.close()


if __name__ == "__main__":
    main()
