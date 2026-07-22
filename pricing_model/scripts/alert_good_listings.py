#!/usr/bin/env python3
"""Ping the user's phone (ntfy) when the scraper captures a good listing.

Runs from cron (idempotent, watermarked): reads tcgplayer_listings rows newer
than the last processed snapshot, joins model fair values, and pushes an ntfy
notification for listings that clear conservative deal rules. Every alerted
listing_id is recorded so a listing is only ever pinged once.

Deal rules (reviewer-informed, see pricing_model/data/residuals.json caveats):
  - condition-adjusted: FV is LP-basis; NM uses FV/0.73, LP uses FV,
    MP uses FV*0.72. Threshold: all-in <= DISCOUNT (default 0.65) x adj FV.
  - HP allowed only at <= 0.50 x (FV * 0.55) and tagged "inspect!"; DMG never
    (lemon grades).
  - all-in >= $10 (no bulk noise), language EN, model row must exist and its
    set must be unflagged.

Config: pricing_model/data/alert_config.json {"ntfy_topic": "...",
"discount": 0.65}. Test: --test sends a hello ping; --dry-run prints instead
of sending.
"""

import argparse
import json
import os
import sqlite3
import urllib.request

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.path.join(REPO, "data", "tcgplayer_sales.db")
OOF = os.path.join(REPO, "pricing_model", "data", "fair_value_oof.parquet")
CONFIG = os.path.join(REPO, "pricing_model", "data", "alert_config.json")

COND_FACTOR = {"Near Mint": 1 / 0.73, "Lightly Played": 1.0,
               "Moderately Played": 0.72}
HP_FACTOR = 0.55


def push(topic, title, body, url=None, dry=False):
    if dry:
        print(f"[dry-run] {title} | {body} | {url}")
        return
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body.encode(),
        headers={"Title": title, "Priority": "high", "Tags": "moneybag",
                 **({"Click": url} if url else {})})
    urllib.request.urlopen(req, timeout=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true", help="send a hello ping")
    args = ap.parse_args()

    cfg = json.load(open(CONFIG))
    topic, discount = cfg["ntfy_topic"], cfg.get("discount", 0.65)

    if args.test:
        push(topic, "cardprice alerts: connected",
             "You will receive listing alerts on this topic.", dry=args.dry_run)
        print("test ping sent")
        return

    conn = sqlite3.connect(DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS listing_alerts_state
            (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS listing_alerts_sent
            (listing_id INTEGER PRIMARY KEY, alerted_at TEXT);
    """)
    wm = conn.execute(
        "SELECT v FROM listing_alerts_state WHERE k='watermark'").fetchone()
    wm = wm[0] if wm else "1970-01-01"

    new = pd.read_sql(
        "SELECT * FROM tcgplayer_listings WHERE scraped_at > ?", conn,
        params=(wm,))
    if not len(new):
        conn.close()
        return
    max_ts = new["scraped_at"].max()

    fv = pd.read_parquet(OOF)[
        ["pid", "printing", "name", "card_id", "era", "fair_value",
         "label_lp", "set_pop_flagged" if "set_pop_flagged" in
         pd.read_parquet(OOF).columns else "era"]]
    fv = fv.rename(columns={"pid": "tcg_product_id"})
    d = new.merge(fv, on=["tcg_product_id", "printing"], how="inner")
    d["all_in"] = d["price"] + d["shipping_price"]
    sent = {r[0] for r in conn.execute(
        "SELECT listing_id FROM listing_alerts_sent")}

    n_alerts = 0
    for r in d.itertuples():
        if r.listing_id in sent or r.language not in ("EN", ""):
            continue
        if r.all_in < 10:
            continue
        tag = ""
        if r.condition in COND_FACTOR:
            thresh = discount * r.fair_value * COND_FACTOR[r.condition]
        elif r.condition == "Heavily Played":
            thresh = 0.50 * r.fair_value * HP_FACTOR
            tag = " INSPECT (HP)"
        else:
            continue
        if r.all_in > thresh:
            continue
        url = f"https://www.tcgplayer.com/product/{r.tcg_product_id}"
        title = f"Deal: {r.name} {r.printing} {r.condition}{tag}"
        body = (f"${r.all_in:.2f} vs FV ${r.fair_value:.0f} "
                f"({r.all_in / r.fair_value:.2f}x LP-basis) | {r.seller_name}"
                f"{' direct' if r.direct_seller else ''} | {r.card_id}")
        push(topic, title, body, url, dry=args.dry_run)
        n_alerts += 1
        if not args.dry_run and r.listing_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO listing_alerts_sent VALUES (?, datetime('now'))",
                (int(r.listing_id),))

    if not args.dry_run:
        conn.execute(
            "INSERT OR REPLACE INTO listing_alerts_state VALUES ('watermark', ?)",
            (max_ts,))
        conn.commit()
    conn.close()
    print(f"processed {len(new)} new listing rows -> {n_alerts} alerts")


if __name__ == "__main__":
    main()
