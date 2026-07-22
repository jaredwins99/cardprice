#!/usr/bin/env python3
"""Collect PSA graded population for EVERY card in the catalog (per-variant).

Part of the pricing_model sub-project (isolated; reads main-project data, never
edits main-project code). Expands set_population's chase-only collection
(set_population/scripts/collect_psa_pop.py) to all ~20k cards.

Why this works: the GradedMetrics/PokeMetrics community PSA-pop mirror serves
one static JSON per PSA set containing ALL cards, and (contrary to the chase
collector's 2026-05 docstring) each card entry now carries:
    n  = card name
    e  = card number (int)
    t  = total graded
    f  = per-grade breakdown {grade: {g: count, h: half, q: qualifier}}
    x  = variant labels (["Reverse holofoil"], ["Holofoil"], ["1st Edition"],
         IR/SIR/Secret labels, error/store-promo exotica) or absent
    j  = last 8 weekly snapshots of total pop  (newest first)
    ja = last 8 weekly snapshots of PSA-10 pop (newest first)
So card matching is (card_number, name-verified) — near-deterministic — and we
get grading-velocity for free.

Set matching is NOT redone from scratch: psa_set_id per set_id is reused from
set_population/data/chase_graded_pop.json (163/170 solved, confidence-labeled).
Sets missing there (the newest 2025-26 sets) are retried against a fresh
sets.json with the same normalization rules. For WOTC/early sets we also look
for sibling "1st Edition" PSA set docs and attach their pops separately.

OUTPUT  pricing_model/data/card_graded_pops.json
  { generated_at, source, stats,
    set_meta: { set_id: {psa_set_id, psa_set_name, set_match_confidence,
                         first_ed_psa_set_id, psa_entries, matched_entries} },
    cards: { card_id: [ {e, psa_name, variants, printing_guess, psa_total,
                          psa10, psa9, by_grade, pop_hist, psa10_hist,
                          match, first_edition_doc} ] } }

printing_guess maps PSA variant labels toward fact_market_prices.subtype_name
vocabulary ("Normal"/"Holofoil"/"Reverse Holofoil"/"1st Edition"/
"1st Edition Holofoil"); "base" means "the card's default printing" (caller
resolves against the printings that actually exist in price data); "exotic"
marks error cards / store stamps (excluded from supply features, kept raw).

Raw set docs are cached in pricing_model/data/psa_raw/ so re-runs are free.
stdlib + requests + psycopg2 (read-only SELECT on dim_cards/dim_sets).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "psa_raw")
DEFAULT_OUT = os.path.abspath(os.path.join(DATA_DIR, "card_graded_pops.json"))
CHASE_POP = os.path.abspath(os.path.join(
    HERE, "..", "..", "set_population", "data", "chase_graded_pop.json"))

GM_BASE = "https://raw.githubusercontent.com/gradedmetrics/api/master/docs"
GM_SETS_URL = f"{GM_BASE}/sets.json"
GM_SET_URL = f"{GM_BASE}/sets/{{psa_set_id}}.json"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
MIN_INTERVAL = 3.0

# --- name/set normalization (same rules as set_population/collect_psa_pop.py) ---

REGIONAL_MARKERS = re.compile(
    r"\b(jp|jpn|japanese|japan|int'?l|international|korean|korea|chinese|china|"
    r"simplified|traditional|thai|german|french|spanish|italian|portuguese|"
    r"indonesian|carddass|kids|shikishi|zukancard|campaign)\b",
    re.IGNORECASE,
)
LANG_DASH = re.compile(r"^(?:[a-z0-9]+\s+)?([a-z]{1,2})\s*-\s*(.+)$", re.IGNORECASE)
JP_PRODUCT_DASH = re.compile(r"^s[vco]?\d*[a-z]?-", re.IGNORECASE)
NON_EN_LANG = {"de", "fr", "it", "es", "pt", "c", "f", "t", "i", "ja", "jp", "kr"}
ERA_PREFIXES = [
    "sword & shield", "sword and shield", "scarlet & violet", "scarlet and violet",
    "sun & moon", "sun and moon", "black & white", "black and white",
    "diamond & pearl", "diamond and pearl", "heartgold & soulsilver", "hgss",
    "ex ", "xy ", "swsh ", "international", "int'l", "mega evolution",
]


def norm(s):
    if not s:
        return ""
    s = s.lower().replace("&", " and ").replace("é", "e")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_era(nm):
    for p in ERA_PREFIXES:
        np = norm(p)
        if nm.startswith(np + " "):
            return nm[len(np):].strip()
    return nm


def psa_english_key(raw_name):
    if not raw_name:
        return None
    if JP_PRODUCT_DASH.match(raw_name.strip()):
        return None
    name = raw_name
    m = LANG_DASH.match(raw_name.strip())
    if m:
        lang, rest = m.group(1).lower(), m.group(2)
        if lang == "en":
            name = rest
        elif lang in NON_EN_LANG:
            return None
    if REGIONAL_MARKERS.search(name):
        return None
    return strip_era(norm(name))


# --- variant label -> printing vocabulary used by fact_market_prices ---

EXOTIC_PAT = re.compile(
    r"error|stamp|missing|inverted|incomplete|albino|rotated|dot|cheeks|"
    r"exclusive|promo|pre.?order|box$|deck$|kit$|league|together|black flame|"
    r"red heart|shadowless|no.?symbol|square cut",
    re.IGNORECASE,
)


def printing_guess(x_labels):
    """Map a PSA entry's variant list to our printing vocabulary."""
    if not x_labels:
        return "base"
    xs = [str(x).lower() for x in x_labels]
    core = [x for x in xs if not EXOTIC_PAT.search(x)]
    if len(core) < len(xs) and not core:
        return "exotic"
    joined = " ".join(core)
    first = "1st edition" in joined
    holo = "holofoil" in joined or joined == "holo"
    reverse = "reverse" in joined
    if len(core) < len(xs):
        # e.g. ["Black Dot Error", "Holofoil"] — an error print of the holo
        return "exotic"
    if reverse:
        return "Reverse Holofoil"
    if first and holo:
        return "1st Edition Holofoil"
    if first:
        return "1st Edition"
    if holo:
        return "Holofoil"
    # modern rarity descriptors are their own card numbers; the entry is still
    # that card's default printing
    if any(k in joined for k in ("illustration rare", "secret rare", "hyper rare",
                                 "gold star", "double rare", "ultra rare",
                                 "full art", "rainbow")):
        return "base"
    return "exotic"


# --- polite HTTP with disk cache ---

class Client:
    def __init__(self, min_interval=MIN_INTERVAL, verbose=False):
        self.min_interval = min_interval
        self.last = 0.0
        self.verbose = verbose
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def get_json(self, url, cache_name=None, max_retries=4):
        if cache_name:
            path = os.path.join(RAW_DIR, cache_name)
            if os.path.exists(path):
                try:
                    return json.load(open(path))
                except ValueError:
                    pass
        backoff = self.min_interval
        for _ in range(max_retries):
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.sess.get(url, timeout=30)
            except requests.RequestException as e:
                if self.verbose:
                    print(f"    net error {e}; retry in {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                self.last = time.time()
                continue
            self.last = time.time()
            if r.status_code == 200:
                try:
                    doc = r.json()
                except ValueError:
                    return None
                if cache_name:
                    os.makedirs(RAW_DIR, exist_ok=True)
                    tmp = os.path.join(RAW_DIR, cache_name + ".tmp")
                    with open(tmp, "w") as f:
                        json.dump(doc, f)
                    os.replace(tmp, os.path.join(RAW_DIR, cache_name))
                return doc
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
        return None


# --- our catalog ---

def load_catalog():
    """dim_cards + dim_sets via read-only Postgres. Returns
    {set_id: {"name":.., "year":.., "cards": [ {card_id,name,number,rarity} ]}}"""
    import psycopg2
    conn = psycopg2.connect(dbname="cardprice")
    cur = conn.cursor()
    cur.execute("""
        SELECT dc.set_id, ds.name, EXTRACT(YEAR FROM ds.release_date)::int,
               dc.card_id, dc.name, dc.card_number, dc.rarity
        FROM dim_cards dc JOIN dim_sets ds ON ds.set_id = dc.set_id
        ORDER BY dc.set_id, dc.card_id""")
    sets = {}
    for set_id, set_name, year, card_id, name, number, rarity in cur.fetchall():
        s = sets.setdefault(set_id, {"name": set_name, "year": year, "cards": []})
        s["cards"].append({"card_id": card_id, "name": name,
                           "number": number, "rarity": rarity})
    conn.close()
    return sets


def norm_number(n):
    """'053' -> '53', 'TG12' -> 'TG12', '86_A' -> '86A'. Uppercase, no
    leading zeros on the numeric run, punctuation stripped."""
    if n is None:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(n)).upper()
    m = re.match(r"^([A-Z]*)0*(\d+)([A-Z]*)$", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}{m.group(3)}"
    return s or None


# --- card matching within a matched PSA set ---

def summarize_entry(pc, doc_kind):
    f = pc.get("f") or {}

    def g(grade):
        v = f.get(grade)
        return int(v.get("g", 0) or 0) if isinstance(v, dict) else 0

    by_grade = {k: int(v.get("g", 0) or 0)
                for k, v in f.items()
                if k != "t" and isinstance(v, dict) and v.get("g")}
    return {
        "e": pc.get("e"),
        "psa_name": pc.get("n"),
        "variants": pc.get("x"),
        "printing_guess": printing_guess(pc.get("x")),
        "psa_total": int(pc.get("t", 0) or 0),
        "psa10": g("10"),
        "psa9": g("9"),
        "by_grade": by_grade,
        "pop_hist": pc.get("j"),
        "psa10_hist": pc.get("ja"),
        "first_edition_doc": doc_kind == "1st",
    }


def year_ok(our_year, pc, slack=2):
    """Reject a PSA entry whose card year is far from our set's release year.
    Guards against cross-era contamination when a PSA set doc was mis-mapped
    (e.g. the chase file maps BOTH xyp and swshp to PSA set 2q64)."""
    if not our_year:
        return True
    try:
        py = int(str(pc.get("y", ""))[:4])
    except (ValueError, TypeError):
        return True
    return abs(int(our_year) - py) <= slack


def numeric_part(num):
    m = re.search(r"\d+", str(num)) if num is not None else None
    return str(int(m.group(0))) if m else None


def match_set_cards(our_cards, psa_cards, doc_kind="main", our_year=None):
    """Match every our-card against PSA entries by (number, name) with a year
    guard. Returns (matches {card_id: [entry,...]}, n_matched_entries)."""
    by_number, by_numeric, by_name = {}, {}, {}
    for pc in psa_cards:
        key = norm_number(pc.get("e"))
        if key is not None:
            by_number.setdefault(key, []).append(pc)
        nkey = numeric_part(pc.get("e"))
        if nkey is not None:
            by_numeric.setdefault(nkey, []).append(pc)
        by_name.setdefault(norm(pc.get("n", "")), []).append(pc)

    matches = {}
    used = 0
    for c in our_cards:
        tnum = norm_number(c["number"])
        tname = norm(c["name"])
        ttoks = set(tname.split())
        picked = []

        def try_cands(cands, exact_name_only):
            out = []
            for pc in cands:
                if not year_ok(our_year, pc):
                    continue
                pname = norm(pc.get("n", ""))
                ptoks = set(pname.split())
                if pname == tname:
                    conf = "high"
                elif (not exact_name_only) and ttoks and (
                        ttoks <= ptoks or ptoks <= ttoks):
                    conf = "med"      # e.g. PSA "Dark Dragonite" vs "Dragonite"
                else:
                    continue          # number collision with different card
                e = summarize_entry(pc, doc_kind)
                e["match"] = conf
                out.append(e)
            return out

        if tnum:
            picked = try_cands(by_number.get(tnum, []), exact_name_only=False)
        if not picked and tnum:
            # number-style mismatch: our 'XY124' vs PSA e=124, or our '56' vs
            # PSA 'DP56' — exact name only, and never across DIFFERENT alpha
            # prefixes ('SM158' must not match 'XY158')
            npart = numeric_part(tnum)
            if npart:
                tpref = re.match(r"^[A-Z]+", tnum)
                cands = []
                for pc in by_numeric.get(npart, []):
                    pnum = norm_number(pc.get("e")) or ""
                    ppref = re.match(r"^[A-Z]+", pnum)
                    if tpref and ppref and tpref.group(0) != ppref.group(0):
                        continue
                    if pnum == tnum:
                        continue  # exact path already tried these
                    cands.append(pc)
                picked = try_cands(cands, exact_name_only=True)
                for e in picked:
                    e["match"] = "med"
        if not picked and tname in by_name and len(by_name[tname]) == 1:
            # unique-name fallback (PSA encodes some numbers differently).
            # Only when the PSA entry has no number or numeric parts agree —
            # else 'Pikachu XY89' would attach to swshp-SWSH020 'Pikachu'.
            pc = by_name[tname][0]
            pe = pc.get("e")
            if year_ok(our_year, pc) and (
                    pe is None or numeric_part(pe) == numeric_part(tnum)):
                e = summarize_entry(pc, doc_kind)
                e["match"] = "low"
                picked = [e]
        if picked:
            matches[c["card_id"]] = picked
            used += len(picked)
    return matches, used


# --- 1st Edition sibling sets (WOTC era) ---

def find_first_ed_set(set_name, set_index):
    key = strip_era(norm(set_name))
    for cand_key, lst in set_index.items():
        if cand_key in (f"{key} 1st edition", f"1st edition {key}"):
            return max(lst, key=lambda t: t[1])[0]
    return None


def build_set_index(psa_sets):
    idx = {}
    for sid, v in psa_sets.items():
        key = psa_english_key(v.get("n", ""))
        if not key:
            continue
        idx.setdefault(key, []).append(
            (sid, int(v.get("t", 0) or 0), str(v.get("y", "")), v.get("n", "")))
    for k in idx:
        idx[k].sort(key=lambda t: t[1], reverse=True)
    return idx


def fresh_set_match(set_name, year, set_index):
    target = strip_era(norm(set_name))
    cands = list(set_index.get(target, []))
    conf = "high" if cands else None
    if not cands:
        ttoks = set(target.split())
        for key, lst in set_index.items():
            ktoks = set(key.split())
            if ttoks and (ttoks <= ktoks or ktoks <= ttoks):
                cands.extend(lst)
        conf = "low" if cands else None
    if not cands:
        return None, None, "none"
    yr2 = str(year)[-2:] if year else None
    if yr2:
        hits = [c for c in cands if c[2] == yr2]
        if hits:
            cands = hits
            if conf == "low":
                conf = "med"
    sid, _, _, raw = max(cands, key=lambda t: t[1])
    return sid, raw, conf


# --- Black Star Promo sets: PSA splits these by YEAR (e.g. four separate
# "Swsh Black Star Promo" sets for '20-'23), so single-doc mapping misses most
# cards. Instead: pool entries from ALL "black star promo"-named PSA sets and
# match on the era-prefixed card number PSA uses there (e="SWSH247", "XY89").
# For prefix-less promo sets (basep, np: plain numeric), a doc-year window
# disambiguates WOTC ('00-'01) from Nintendo-era ('03+) docs.

PROMO_SET_YEARS = {  # our set_id -> (accept docs from year, to year)
    "basep": (1999, 2002), "np": (2003, 2007), "dpp": (2007, 2011),
    "hsp": (2010, 2012), "bwp": (2010, 2013), "xyp": (2013, 2017),
    "smp": (2016, 2019), "swshp": (2019, 2023), "svp": (2022, 2029),
}


def collect_promo_pool(client, psa_sets, verbose=False):
    """Fetch every PSA set whose name contains 'black star'/'promo black star'
    and return [(doc_year:int|None, card_entry), ...]."""
    pool = []
    for sid, v in psa_sets.items():
        raw = str(v.get("n", ""))
        rl = norm(raw)
        # HGSS/BW-era promo lines are named without "black star"
        if ("black star" not in raw.lower()
                and rl not in ("black and white promo",
                               "heartgold and soulsilver promo")):
            continue
        if psa_english_key(raw) is None:  # foreign-language promo sets
            continue
        doc = client.get_json(GM_SET_URL.format(psa_set_id=sid),
                              cache_name=f"{sid}.json")
        if not doc:
            continue
        try:
            yy = int(str(v.get("y", "")))
            doc_year = 2000 + yy if yy < 90 else 1900 + yy
        except ValueError:
            doc_year = None
        for pc in doc.get("c", []):
            pool.append((doc_year, pc))
        if verbose:
            print(f"  promo pool: {sid} '{raw}' y={doc_year} "
                  f"+{len(doc.get('c', []))} entries")
    return pool


def match_promo_set(set_id, our_cards, pool):
    """Match one of our promo sets against the pooled promo entries."""
    lo, hi = PROMO_SET_YEARS[set_id]
    cards = [pc for (yr, pc) in pool if yr is None or lo <= yr <= hi]
    # era-prefixed numbers make cross-era collisions impossible; the year
    # window above handles the numeric-only WOTC/Nintendo docs
    return match_set_cards(our_cards, cards, "main", our_year=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit-sets", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    chase = json.load(open(CHASE_POP))["sets"]
    catalog = load_catalog()
    client = Client(verbose=args.verbose)

    psa_sets = client.get_json(GM_SETS_URL, cache_name="sets.json")
    if not psa_sets:
        sys.exit("FATAL: cannot fetch GradedMetrics sets.json")
    set_index = build_set_index(psa_sets)

    plan = []
    for set_id, meta in sorted(catalog.items()):
        prior = chase.get(set_id)
        psa_id = prior.get("psa_set_id") if prior else None
        sconf = prior.get("set_match_confidence") if prior else None
        src = "chase_file"
        if not psa_id:
            psa_id, _, sconf = fresh_set_match(meta["name"], meta["year"], set_index)
            src = "fresh_match"
        first_ed = find_first_ed_set(meta["name"], set_index)
        plan.append((set_id, meta, psa_id, sconf, first_ed, src))

    if args.dry_run:
        for set_id, meta, psa_id, sconf, first_ed, src in plan:
            tag = f"{psa_id} [{sconf},{src}]" if psa_id else "NO MATCH"
            fe = f" +1stEd:{first_ed}" if first_ed else ""
            print(f"  {set_id:<12} {meta['name']:<30} -> {tag}{fe}  "
                  f"({len(meta['cards'])} cards)")
        n_unmatched = sum(1 for p in plan if not p[2])
        print(f"\n{len(plan)} sets, {n_unmatched} without a PSA set match")
        return

    if args.limit_sets:
        plan = plan[:args.limit_sets]

    out_cards, set_meta = {}, {}
    stats = {"sets_total": len(plan), "sets_matched": 0, "cards_total": 0,
             "cards_with_pop": 0, "entries_matched": 0,
             "match_high": 0, "match_med": 0, "match_low": 0}

    promo_pool = None
    for set_id, meta, psa_id, sconf, first_ed, src in plan:
        stats["cards_total"] += len(meta["cards"])
        if set_id in PROMO_SET_YEARS:
            if promo_pool is None:
                promo_pool = collect_promo_pool(client, psa_sets,
                                                verbose=args.verbose)
            matches, used = match_promo_set(set_id, meta["cards"], promo_pool)
            set_meta[set_id] = {
                "psa_set_id": None, "set_match_confidence": "promo_pool",
                "set_match_source": "promo_pool", "first_ed_psa_set_id": None,
                "psa_entries": len(promo_pool), "matched_entries": used}
            stats["sets_matched"] += 1
            stats["entries_matched"] += used
            for cid, entries in matches.items():
                out_cards[cid] = entries
                stats["cards_with_pop"] += 1
                for e in entries:
                    stats[f"match_{e['match']}"] += 1
            if args.verbose:
                print(f"  {set_id:<12} {meta['name']:<30} PROMO POOL "
                      f"matched_cards={len(matches)}")
            continue
        sm = {"psa_set_id": psa_id, "set_match_confidence": sconf or "none",
              "set_match_source": src, "first_ed_psa_set_id": first_ed,
              "psa_entries": 0, "matched_entries": 0}
        set_meta[set_id] = sm
        if not psa_id:
            continue
        doc = client.get_json(GM_SET_URL.format(psa_set_id=psa_id),
                              cache_name=f"{psa_id}.json")
        if not doc:
            sm["set_match_confidence"] = "fetch_failed"
            continue
        stats["sets_matched"] += 1
        psa_cards = doc.get("c", [])
        sm["psa_entries"] = len(psa_cards)
        matches, used = match_set_cards(meta["cards"], psa_cards, "main")

        if first_ed:
            fe_doc = client.get_json(GM_SET_URL.format(psa_set_id=first_ed),
                                     cache_name=f"{first_ed}.json")
            if fe_doc:
                fe_matches, fe_used = match_set_cards(
                    meta["cards"], fe_doc.get("c", []), "1st")
                used += fe_used
                for cid, entries in fe_matches.items():
                    matches.setdefault(cid, []).extend(entries)

        sm["matched_entries"] = used
        stats["entries_matched"] += used
        for cid, entries in matches.items():
            out_cards[cid] = entries
            stats["cards_with_pop"] += 1
            for e in entries:
                stats[f"match_{e['match']}"] += 1
        if args.verbose:
            print(f"  {set_id:<12} {meta['name']:<30} psa={psa_id} "
                  f"entries={len(psa_cards)} matched_cards={len(matches)}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "GradedMetrics/PokeMetrics community PSA pop mirror "
                  "(github.com/gradedmetrics/api docs/), per-card expansion",
        "stats": stats,
        "set_meta": set_meta,
        "cards": out_cards,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    os.replace(tmp, args.out)
    print(f"\nWrote {args.out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
