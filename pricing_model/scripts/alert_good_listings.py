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
import re
import sqlite3
import unicodedata
import urllib.request

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.path.join(REPO, "data", "tcgplayer_sales.db")
OOF = os.path.join(REPO, "pricing_model", "data", "fair_value_oof.parquet")
EXPLAIN = os.path.join(REPO, "pricing_model", "data", "fv_explanations.parquet")
CONFIG = os.path.join(REPO, "pricing_model", "data", "alert_config.json")

COND_FACTOR = {"Near Mint": 1 / 0.73, "Lightly Played": 1.0,
               "Moderately Played": 0.72}
HP_FACTOR = 0.55
# Ordinal condition ranking (1 = best) for ladder-violation detection
COND_RANK = {"Near Mint": 1, "Lightly Played": 2, "Moderately Played": 3,
             "Heavily Played": 4, "Damaged": 5}


# Seller-title caveats that explain a cheap price without any mispricing.
# TCGPlayer's languageAbbreviation is unreliable (returns "EN" for a listing
# titled "*Chinese* Near Mint Holofoil Pikachu and Zekrom GX"), so the seller's
# own title is the real disclosure channel. Found live 2026-07-23.
CAVEAT_PAT = re.compile(
    r"chinese|japanese|korean|german|french|spanish|italian|portuguese|"
    r"thai|indonesian|russian|jpn|jap\b|foreign|"
    r"proxy|fake|replica|reprint|custom|orica|altered|"
    r"crease|bend|scratch|whitening|water\s*damage|played|damaged|"
    r"miscut|misprint|error|test\s*print",
    re.IGNORECASE)


def caveat_of(title):
    """Return the seller-disclosed caveat in a listing title, if any."""
    if not title:
        return None
    m = CAVEAT_PAT.search(title)
    return m.group(0) if m else None


def steal_evidence(r, own_map, steal_discount):
    """Is this listing cheap against the CARD'S OWN market? Model-independent.

    Two ways to qualify:
      (a) own-condition discount: price <= steal_discount x median recent sale
          of the SAME condition (needs >= 2 comparable sales);
      (b) ladder violation: price < the max recent sale of a strictly WORSE
          condition (needs >= 2 such sales, so one odd print can't trigger it).
    Returns (reason_str, ratio_to_own_market) or (None, None).
    """
    key = (r.tcg_product_id, r.printing, r.condition)
    same = own_map.get(key)
    if same and same[2] >= 2 and r.all_in <= steal_discount * same[0]:
        return (f"its own {r.condition} sells ${same[0]:.0f} "
                f"(n={int(same[2])}, 90d)"), r.all_in / same[0]
    my_rank = COND_RANK.get(r.condition, 99)
    for cond, rank in COND_RANK.items():
        if rank <= my_rank:
            continue
        worse = own_map.get((r.tcg_product_id, r.printing, cond))
        if worse and worse[2] >= 2 and r.all_in < worse[1]:
            return (f"LADDER BREAK: a {cond} copy sold ${worse[1]:.0f} "
                    f"(n={int(worse[2])})"), r.all_in / worse[1]
    return None, None


def _hdr_safe(s):
    """HTTP headers are latin-1 only, but card names carry δ (delta species),
    é (Pokémon), ♀/♂ (Nidoran), etc. Transliterate what we can, drop the rest.
    The BODY is UTF-8 and keeps the original text."""
    repl = {"δ": "delta", "♀": "(F)", "♂": "(M)",
            "★": "*", "☆": "*", "’": "'", "—": "-",
            "–": "-"}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "ignore").decode("latin-1")


# Two alert classes, deliberately distinguishable at a glance:
#   STEAL  — cheap vs THE CARD'S OWN evidence (its sales ladder). Near-
#            arbitrage: does not depend on the model being right. Red/urgent.
#   MODEL  — cheap vs cross-sectional model fair value. Depends on the model,
#            which the temporal test showed is a ranking signal (~7%/6mo
#            convergence), not a price target. Blue/high.
ALERT_STYLE = {
    "steal": {"tags": "rotating_light,fire", "priority": "urgent",
              "prefix": "STEAL"},
    "model": {"tags": "large_blue_circle,abacus", "priority": "high",
              "prefix": "MODEL"},
}


def push(topic, title, body, url=None, dry=False, actions=None,
         kind="model", image=None):
    style = ALERT_STYLE[kind]
    title = f"{style['prefix']}: {title}"
    if dry:
        print(f"[dry-run/{kind}] {title} | {body} | {url}")
        return
    headers = {"Title": _hdr_safe(title), "Priority": style["priority"],
               "Tags": style["tags"]}
    if url:
        headers["Click"] = url
    if actions:
        headers["Actions"] = actions
    if image:
        # ntfy renders an Attach URL inline in the app — card art in the ping
        headers["Attach"] = image
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body.encode(), headers=headers)
    urllib.request.urlopen(req, timeout=15)


def poll_decisions(conn, cfg, dry=False):
    """Pull veto/approve button taps from the decisions topic into the DB."""
    dtopic = cfg.get("decisions_topic")
    if not dtopic or dry:
        return 0
    row = conn.execute(
        "SELECT v FROM listing_alerts_state WHERE k='decisions_since'").fetchone()
    since = row[0] if row else "0"
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{dtopic}/json?poll=1&since={since}",
                timeout=15) as r:
            lines = r.read().decode().strip().splitlines()
    except Exception:
        return 0
    n = 0
    last_id = since
    for line in lines:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("event") != "message":
            continue
        last_id = msg.get("id", last_id)
        parts = (msg.get("message") or "").strip().split()
        if len(parts) == 2 and parts[0] in ("approve", "veto"):
            conn.execute(
                "UPDATE listing_alerts_decisions SET decision=?, "
                "decided_at=datetime('now') WHERE listing_id=? "
                "AND decision='pending'",
                (parts[0], int(parts[1])))
            n += 1
    conn.execute(
        "INSERT OR REPLACE INTO listing_alerts_state VALUES ('decisions_since', ?)",
        (last_id,))
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true", help="send a hello ping")
    args = ap.parse_args()

    cfg = json.load(open(CONFIG))
    topic = cfg["ntfy_topic"]
    discount = cfg.get("discount", 0.50)
    max_alerts = cfg.get("max_alerts_per_run", 5)
    MIN_PRICE = cfg.get("min_price", 20)

    if args.test:
        dtopic = cfg.get("decisions_topic")
        actions = None
        if dtopic:
            actions = (
                f"http, Approve, https://ntfy.sh/{dtopic}, method=POST, "
                f"body=approve 0; "
                f"http, Veto, https://ntfy.sh/{dtopic}, method=POST, "
                f"body=veto 0")
        push(topic, "TEST: Blastoise δ Holofoil Heavily Played",
             "$62.00 vs FV $262 (0.24x LP-basis) | MikusMarket | "
             "ex14-2/normal — tap Approve or Veto to test the loop",
             "https://www.tcgplayer.com/product/83899",
             dry=args.dry_run, actions=actions)
        print(f"test ping sent to topic {topic}"
              + (f"; decisions -> {dtopic}" if dtopic else ""))
        return

    conn = sqlite3.connect(DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS listing_alerts_state
            (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS listing_alerts_sent
            (listing_id INTEGER PRIMARY KEY, alerted_at TEXT);
        CREATE TABLE IF NOT EXISTS listing_alerts_decisions (
            listing_id INTEGER PRIMARY KEY,
            card_id TEXT, name TEXT, printing TEXT, condition TEXT,
            all_in REAL, fair_value REAL, ratio REAL,
            alerted_at TEXT,
            decision TEXT DEFAULT 'pending',   -- pending | approve | veto
            decided_at TEXT
        );
    """)
    n_dec = poll_decisions(conn, cfg, dry=args.dry_run)
    if n_dec:
        print(f"recorded {n_dec} decisions from phone")
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
        ["pid", "printing", "name", "card_id", "era", "fair_value", "label_lp"]]
    if os.path.exists(EXPLAIN):
        fv = fv.merge(pd.read_parquet(EXPLAIN), on=["pid", "printing"], how="left")
    else:
        fv["top_factors"] = None
    fv = fv.rename(columns={"pid": "tcg_product_id"})
    d = new.merge(fv, on=["tcg_product_id", "printing"], how="inner")
    d["all_in"] = d["price"] + d["shipping_price"]

    # --- the card's OWN market: recent sales by condition --------------------
    # A "steal" is defined without reference to the model: cheap against this
    # card's own recent clearing prices, or violating its own condition ladder
    # (a worse-condition copy sold for more). Blastoise delta taught this the
    # hard way — its $62 HP listing looked like 0.24x model FV but was exactly
    # the HP market price, with a perfectly coherent ladder.
    pids = tuple(int(p) for p in d["tcg_product_id"].unique()) or (0,)
    own = pd.read_sql(
        f"SELECT tcg_product_id, printing, condition, sale_price "
        f"FROM tcgplayer_sales WHERE tcg_product_id IN {pids} "
        f"AND sale_date >= date('now', '-90 days')", conn)
    own_stats = (own.groupby(["tcg_product_id", "printing", "condition"])
                 ["sale_price"].agg(["median", "max", "size"]).reset_index())
    own_map = {(r.tcg_product_id, r.printing, r.condition):
               (r.median, r.max, r.size) for r in own_stats.itertuples()}
    # card art for the ping (ntfy renders an Attach URL inline)
    import psycopg2
    pgc = psycopg2.connect(dbname="cardprice")
    IMG = {cid: url for cid, url in pd.read_sql(
        "SELECT card_id, image_small FROM dim_cards "
        "WHERE image_small IS NOT NULL", pgc).values}
    pgc.close()

    # --- seller caveat reputation -------------------------------------------
    # custom_title only exists for listings captured after 2026-07-23, so
    # historical rows cannot be checked. Instead, learn WHICH SELLERS ship
    # caveated listings and distrust their cheap prints generally. Seeded with
    # SinoCARDS, caught by the user listing Chinese cards under EN products.
    seller_flags = dict(cfg.get("caveat_sellers", {}))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tcgplayer_listings)")}
    hist = (pd.read_sql(
        "SELECT seller_name, custom_title FROM tcgplayer_listings "
        "WHERE custom_title IS NOT NULL", conn)
        if "custom_title" in cols else pd.DataFrame())
    if len(hist):
        hist["cav"] = hist["custom_title"].map(lambda t: bool(caveat_of(t)))
        agg = hist.groupby("seller_name")["cav"].agg(["mean", "sum", "size"])
        for s, row in agg.iterrows():
            if row["sum"] >= 3 and row["mean"] >= 0.30:
                seller_flags[s] = f"{row['mean']:.0%} of their listings carry caveats"
    if seller_flags:
        print(f"  {len(seller_flags)} sellers on the caveat list")

    sent = {r[0] for r in conn.execute(
        "SELECT listing_id FROM listing_alerts_sent")}

    # --- score candidates, then send only the best few -----------------------
    # Rate discipline matters more than recall: the rotation now yields ~13k
    # listing rows/day, and a loose threshold produced 273 candidates in one
    # run. An alert the user ignores is worse than no alert (see project
    # memory on notification cost), so we rank by discount depth and cap per
    # run; unsent candidates are NOT marked, so a genuinely deep discount
    # resurfaces next hour when the queue is shorter.
    steal_discount = cfg.get("steal_discount", 0.70)
    cands = []          # (sort_key, row, kind, headline, ratio)
    n_caveat = 0
    for r in d.itertuples():
        if r.listing_id in sent or r.language not in ("EN", ""):
            continue
        if r.all_in < MIN_PRICE:
            continue
        # seller-disclosed caveat (foreign print, damage, proxy...) explains a
        # cheap price with no mispricing — the Chinese-Zekrom lesson
        cav = caveat_of(getattr(r, "custom_title", None))
        if cav:
            n_caveat += 1
            continue
        if r.seller_name in seller_flags:
            n_caveat += 1
            continue
        # listings captured before custom_title existed can't be verified
        unverified = getattr(r, "custom_title", None) is None

        # (1) STEAL — model-independent, judged against the card's own market
        why, own_ratio = steal_evidence(r, own_map, steal_discount)
        if why:
            cands.append((own_ratio, r, "steal", why, own_ratio, unverified))
            continue

        # (2) MODEL — cross-sectional fair value
        if r.condition in COND_FACTOR:
            adj_fv = r.fair_value * COND_FACTOR[r.condition]
            thresh = discount * adj_fv
        elif r.condition == "Heavily Played":
            adj_fv = r.fair_value * HP_FACTOR
            thresh = 0.50 * adj_fv
        else:
            continue
        if r.all_in > thresh:
            continue
        ratio = r.all_in / adj_fv
        # steals sort ahead of model calls at equal depth (0.5 penalty)
        cands.append((ratio + 0.5, r, "model", r.top_factors or "", ratio, unverified))
    if n_caveat:
        print(f"  {n_caveat} listings skipped on seller-disclosed caveats")
    cands.sort(key=lambda c: c[0])

    # --- market-consensus filter (the model is the thing being tested) ------
    # If several INDEPENDENT sellers all price a card far under model FV, the
    # consensus is evidence the model is wrong, not that there are N bargains.
    # (Discovered live: 5 sellers at ~$30 on a card the model marked $143 —
    # a Trainer Gallery subset the model over-prices.) So: keep only the
    # cheapest listing per (product, printing, condition), and drop the card
    # entirely when >= CONSENSUS_N distinct sellers sit below the threshold.
    CONSENSUS_N = cfg.get("consensus_sellers", 3)
    by_card = {}
    for c in cands:
        by_card.setdefault((c[1].tcg_product_id, c[1].printing), []).append(c)
    kept = []
    for key, group in by_card.items():
        # steals are judged against the card's own trades, so seller consensus
        # is not evidence against them — only model calls get suppressed
        model_group = [g for g in group if g[2] == "model"]
        sellers = {g[1].seller_name for g in model_group}
        if len(sellers) >= CONSENSUS_N:
            print(f"  consensus-suppressed {group[0][1].card_id} "
                  f"{group[0][1].printing}: {len(sellers)} sellers below "
                  f"threshold -> model FV ${group[0][1].fair_value:.0f} "
                  "is likely wrong")
            group = [g for g in group if g[2] == "steal"]
            if not group:
                continue
        best = {}
        for g in group:
            k = g[1].condition
            if k not in best or g[0] < best[k][0]:
                best[k] = g
        kept.extend(best.values())
    cands = sorted(kept, key=lambda c: c[0])
    n_suppressed = max(0, len(cands) - max_alerts)
    if n_suppressed:
        print(f"{len(cands)} candidates; sending top {max_alerts}, "
              f"{n_suppressed} held for next run")

    n_alerts = 0
    for _sort, r, kind, why, ratio, unver in cands[:max_alerts]:
        url = f"https://www.tcgplayer.com/product/{r.tcg_product_id}"
        # Condition leads the title — it's what you're actually buying.
        title = f"{r.condition} · {r.name} · {r.printing}"
        price = (f"${r.all_in:.2f}"
                 + (f" (${r.price:.2f} + ${r.shipping_price:.2f} ship)"
                    if r.shipping_price else ""))
        seller = f"{r.seller_name}{' ✓direct' if r.direct_seller else ''}"
        if kind == "steal":
            # what it usually costs, in its own market
            body = "\n".join([
                f"{price}  —  {ratio:.0%} of its own market",
                f"vs {why}",
                f"{r.card_id} · {r.era} · {seller}",
            ] + (["⚠ listing text not captured — verify language/condition"]
                 if unver else []))
        else:
            body = "\n".join([
                f"{price}  —  {ratio:.0%} of model fair value",
                f"model says ${r.fair_value * COND_FACTOR.get(r.condition, HP_FACTOR):.0f} "
                f"for {r.condition} (LP-basis ${r.fair_value:.0f}) because:",
                why or "(no factor breakdown available)",
                f"{r.card_id} · {r.era} · {seller}",
            ] + (["⚠ listing text not captured — verify language/condition"]
                 if unver else []))
        actions = None
        dtopic = cfg.get("decisions_topic")
        if dtopic and r.listing_id is not None:
            lid = int(r.listing_id)
            actions = (
                f"http, Approve, https://ntfy.sh/{dtopic}, method=POST, "
                f"body=approve {lid}; "
                f"http, Veto, https://ntfy.sh/{dtopic}, method=POST, "
                f"body=veto {lid}")
        push(topic, title, body, url, dry=args.dry_run, actions=actions,
             kind=kind, image=IMG.get(r.card_id))
        n_alerts += 1
        if not args.dry_run and r.listing_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO listing_alerts_sent VALUES (?, datetime('now'))",
                (int(r.listing_id),))
            conn.execute(
                "INSERT OR IGNORE INTO listing_alerts_decisions "
                "(listing_id, card_id, name, printing, condition, all_in, "
                "fair_value, ratio, alerted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (int(r.listing_id), r.card_id, r.name, r.printing,
                 r.condition, float(r.all_in), float(r.fair_value),
                 float(r.all_in / r.fair_value)))

    if not args.dry_run:
        conn.execute(
            "INSERT OR REPLACE INTO listing_alerts_state VALUES ('watermark', ?)",
            (max_ts,))
        conn.commit()
    conn.close()
    print(f"processed {len(new)} new listing rows -> {n_alerts} alerts")


if __name__ == "__main__":
    main()
