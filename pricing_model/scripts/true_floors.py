#!/usr/bin/env python3
"""Fetch the TRUE listing floor per (product, printing, condition).

Why this exists: JustTCG quotes (the model's label) are usually excellent —
median quote/actual-sale = 0.999 across 10k+ rows with real trades — but ~5%
sit >25% below the market, and that tail is exactly where "mispriced card"
screens land. Verified live, those cheap quotes repeatedly turn out to be
ghosts: Aquapolis Exeggutor RH quoted $25.25 (frozen across three fetches
since May) with the cheapest real LP listing at $44.39 and an LP sale at
$43.79; Platinum Electabuzz quoted $11.76 with a live LP floor of $16.00.

The browser-intercepted payload can't settle this because it returns the ~10
cheapest listings ACROSS printings, so a premium printing's floor is hidden
behind bulk Normals. But mp-search-api accepts a direct filtered POST — no
Playwright, one request per (product, printing) — returning that printing's
whole book.

Usage:
  python3 pricing_model/scripts/true_floors.py --top 60      # screen residuals
  python3 pricing_model/scripts/true_floors.py --cards ecard2-13/normal ...
Writes data/true_floor_screen.json
"""

import argparse
import json
import os
import random
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
API = "https://mp-search-api.tcgplayer.com/v1/product/{pid}/listings"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
COND_ORDER = ["Near Mint", "Lightly Played", "Moderately Played",
              "Heavily Played", "Damaged"]
CUSTOM_ID_MAX = 10_000_000


def fetch_book(pid, printing, size=50, timeout=25):
    """Full listing book for one (product, printing). Returns list of dicts."""
    body = {"filters": {"term": {"printing": [printing]}, "range": {},
                        "exclude": {}},
            "from": 0, "size": size,
            "sort": {"field": "price+shipping", "order": "asc"},
            "context": {"shippingCountry": "US"}, "aggregations": []}
    req = urllib.request.Request(
        API.format(pid=pid), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Origin": "https://www.tcgplayer.com",
                 "Referer": "https://www.tcgplayer.com/"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:
        return None, f"{type(e).__name__}"
    blocks = d.get("results") or [{}]
    out = []
    for l in blocks[0].get("results", []):
        cd = l.get("customData") or {}
        out.append({
            "listing_id": l.get("listingId"),
            "condition": l.get("condition"),
            "price": float(l.get("price") or 0),
            "ship": float(l.get("sellerShippingPrice")
                          or l.get("shippingPrice") or 0),
            "seller": l.get("sellerName"),
            "qty": l.get("quantity"),
            "custom_title": cd.get("title") if isinstance(cd, dict) else None,
            "is_custom": (l.get("listingId") or 0) < CUSTOM_ID_MAX,
        })
    return out, None


def floors(book):
    """Cheapest all-in listing per condition, excluding custom/disclosure
    listings (foreign prints etc. — see tcgplayer_listing_caveats memory)."""
    res = {}
    for l in book:
        if l["is_custom"] or l["custom_title"]:
            continue
        allin = l["price"] + l["ship"]
        c = l["condition"]
        if c not in res or allin < res[c]["all_in"]:
            res[c] = {"all_in": round(allin, 2), "seller": l["seller"],
                      "listing_id": l["listing_id"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--cards", nargs="*", default=[])
    ap.add_argument("--min-price", type=float, default=8.0)
    args = ap.parse_args()

    o = pd.read_parquet(os.path.join(DATA, "fair_value_oof.parquet"))
    if args.cards:
        sel = o[o["card_id"].isin(args.cards)].copy()
    else:
        sel = o[(o["label_lp"] >= args.min_price)
                & (o["jt_age_days"] <= 30)].nsmallest(args.top, "residual_log")

    rows = []
    for i, r in enumerate(sel.itertuples(), 1):
        book, err = fetch_book(int(r.pid), r.printing)
        time.sleep(random.uniform(0.6, 1.2))
        if err or book is None:
            print(f"[{i}/{len(sel)}] {r.card_id} {r.printing}: FETCH {err}")
            continue
        f = floors(book)
        lp_floor = f.get("Lightly Played", {}).get("all_in")
        nm_floor = f.get("Near Mint", {}).get("all_in")
        # best comparable to an LP-basis fair value
        ref = lp_floor if lp_floor else (nm_floor * 0.73 if nm_floor else None)
        rows.append({
            "card_id": r.card_id, "name": r.name, "era": r.era,
            "printing": r.printing,
            "quote_lp": round(float(r.label_lp), 2),
            "model_fv": round(float(r.fair_value), 2),
            "true_lp_floor": lp_floor, "true_nm_floor": nm_floor,
            "book_depth": len(book),
            "quote_vs_true": round(float(r.label_lp) / ref, 3) if ref else None,
            "model_vs_true": round(float(r.fair_value) / ref, 3) if ref else None,
            "floors": {c: f[c] for c in COND_ORDER if c in f},
        })
        q = rows[-1]["quote_vs_true"]
        print(f"[{i}/{len(sel)}] {r.card_id:22s} {r.printing[:16]:16s} "
              f"quote ${r.label_lp:8.2f}  LPfloor {str(lp_floor):>8}  "
              f"NMfloor {str(nm_floor):>8}  FV ${r.fair_value:8.2f}  "
              f"quote/true {q if q else 'n/a'}")

    out = {"generated_at": pd.Timestamp.now().isoformat(), "rows": rows}
    path = os.path.join(DATA, "true_floor_screen.json")
    json.dump(out, open(path, "w"), indent=1)

    df = pd.DataFrame([r for r in rows if r["quote_vs_true"]])
    if len(df):
        print(f"\n=== {len(df)} verified ===")
        print(f"quote is a GHOST (>=20% below the true floor): "
              f"{(df.quote_vs_true < 0.8).mean():.1%}")
        real = df[(df.quote_vs_true >= 0.8) & (df.model_vs_true >= 1.5)]
        print(f"REAL opportunities (a live floor genuinely below model FV): "
              f"{len(real)}")
        if len(real):
            print(real[["card_id", "name", "era", "printing", "true_lp_floor",
                        "true_nm_floor", "model_fv", "model_vs_true"]]
                  .to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
