#!/usr/bin/env python3
"""Collect PSA (+ best-effort CGC) graded population for each set's chase cards.

Part of the set_population sub-project. Reads chase_cards.json (produced by
select_chase_cards.py) and, for every chase card, fetches its PSA graded
population (total + per-grade) and writes chase_graded_pop.json.

ACCESS PATH (researched 2026-05 — see module docstring tail and README)
-----------------------------------------------------------------------
We do NOT scrape psacard.com directly: psacard.com/pop returns HTTP 403
(Cloudflare) to scripted browsers, and the official PSA Public API
(api.psacard.com/publicapi) requires an OAuth2 bearer token (paid; 100
calls/day free) which we do not have.

Instead we use the **GradedMetrics / PokeMetrics community dataset** — a
public, auth-free, regularly-updated mirror of the PSA population report,
hosted as static JSON on GitHub Pages / raw.githubusercontent.com:

    https://github.com/gradedmetrics/api  (docs/ is the published data)
      docs/sets.json            -> {psa_set_id: {n:name, t:total, y:year, ...}}
      docs/sets/<psa_set_id>.json -> {"c":[ {n:name, t:total, f:{grade->{g,h}}, x:[variants], ...}, ... ]}

This is exactly the "prefer a credible community pop dataset over scraping"
path called for in the brief. CGC pop lives under docs/cgc/ in the same repo;
we attach a CGC set-level total when we can map the set, but per-card CGC
matching is left null (the CGC card files key differently and would need its
own matcher — out of scope for the numerator's PSA-primary signal).

MATCHING (the hard part — recorded per card, never silently guessed)
--------------------------------------------------------------------
1. Set match: normalize our dim_sets name and the PSA set name; PSA prefixes
   English modern sets with the era ("Sword & Shield Evolving Skies"), so we
   strip known era prefixes and compare. We REJECT obviously non-English /
   regional PSA sets (Japanese "Sv2a-", "Int'l", "Thai", "Korean", etc.) by
   keyword, and prefer the English set with the matching year and the largest
   population (the main English print is by far the most-graded).
2. Card match: within the matched PSA set, normalize card names and match the
   chase card name; prefer a card whose variant list `x` contains our printing
   (Holofoil / 1st Edition). Card-number is NOT in this dataset, so name is the
   key — we require a strong normalized-name match.
3. Confidence:
     high  -> set matched with high confidence AND exact normalized card-name hit
     med   -> set matched but card-name matched only fuzzily / via token subset
     low   -> set matched weakly (year-only or ambiguous) or multiple card hits
     none  -> no defensible set or card match (we DO NOT emit a pop number)
   We err toward "none": a wrong PSA spec silently corrupts the estimate.

OUTPUT  set_population/data/chase_graded_pop.json
   { "<set_id>": {"set_name":..,"psa_set_id":..,"psa_set_name":..,
        "set_match_confidence":..,"cgc_set_total":<int|null>,
        "chase":[ {card_id,name,psa_total,psa_by_grade,cgc_total,bgs_total,
                   match_confidence,source_url,psa_card_name,psa_variants}, ...]},
     ... }

Writes incrementally (after each card) so a mid-run network block keeps progress.
stdlib + requests only. No DB access (works purely off chase_cards.json).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

SUBPROJ_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_IN = os.path.abspath(os.path.join(SUBPROJ_DATA, "chase_cards.json"))
DEFAULT_OUT = os.path.abspath(os.path.join(SUBPROJ_DATA, "chase_graded_pop.json"))

GM_BASE = "https://raw.githubusercontent.com/gradedmetrics/api/master/docs"
GM_SETS_URL = f"{GM_BASE}/sets.json"
GM_CGC_SETS_URL = f"{GM_BASE}/cgc/sets.json"
GM_SET_URL = f"{GM_BASE}/sets/{{psa_set_id}}.json"
# Human-facing source page for a PSA set (for the source_url field).
PSA_SET_PAGE = "https://www.pokemetrics.org/sets/{psa_set_id}"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
MIN_INTERVAL = 3.0  # polite floor between network requests (seconds)

# PSA set names that are NOT the main English print — reject for English match.
# Word-level regional markers (full words / phrases).
REGIONAL_MARKERS = re.compile(
    r"\b(jp|jpn|japanese|japan|int'?l|international|korean|korea|chinese|china|"
    r"simplified|traditional|thai|german|french|spanish|italian|portuguese|"
    r"indonesian|carddass|kids|shikishi|zukancard|campaign)\b",
    re.IGNORECASE,
)

# PSA's modern catalog tags English/foreign prints with a "<code> <LANG>-<name>"
# pattern, e.g. "Mew EN-151", "Par EN-Paradox Rift" (English) vs
# "Mew FR-151", "Par de-Paradox Rift", "Sv2a-Pokemon 151" (foreign). We must
# (a) accept EN- and strip the "<code> EN-" prefix, (b) reject any other
# language code, and (c) reject the bare "Sv2a-Pokemon ..." Japanese product
# lines. Language codes seen: EN de FR It ES PT C(hinese) F T(hai) I(talian).
LANG_DASH = re.compile(r"^(?:[a-z0-9]+\s+)?([a-z]{1,2})\s*-\s*(.+)$", re.IGNORECASE)
JP_PRODUCT_DASH = re.compile(r"^s[vco]?\d*[a-z]?-", re.IGNORECASE)  # Sv2a-, Sm-, etc.
NON_EN_LANG = {"de", "fr", "it", "es", "pt", "c", "f", "t", "i", "ja", "jp", "kr"}

# Era prefixes PSA prepends to English set names; strip before comparing.
ERA_PREFIXES = [
    "sword & shield", "sword and shield", "scarlet & violet", "scarlet and violet",
    "sun & moon", "sun and moon", "black & white", "black and white",
    "diamond & pearl", "diamond and pearl", "heartgold & soulsilver", "hgss",
    "ex ", "xy ", "swsh ", "international", "int'l",
]


def norm(s):
    """Aggressive normalization for name comparison: lowercase, strip
    punctuation/era words, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ").replace("é", "e")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_era(norm_name):
    for p in ERA_PREFIXES:
        np = norm(p)
        if norm_name.startswith(np + " "):
            return norm_name[len(np):].strip()
    return norm_name


def psa_english_key(raw_name):
    """Map a raw PSA set name to a normalized English match key, or return None
    if the set is a non-English / foreign / product-tin spec we must reject.

    Handles PSA's "<code> <LANG>-<name>" tagging: keep & strip EN-, reject other
    language codes; reject bare Japanese "Sv2a-..." product lines and word-level
    regional markers."""
    if not raw_name:
        return None
    # bare Japanese product line, e.g. "Sv2a-Pokemon 151"
    if JP_PRODUCT_DASH.match(raw_name.strip()):
        return None
    name = raw_name
    m = LANG_DASH.match(raw_name.strip())
    if m:
        lang, rest = m.group(1).lower(), m.group(2)
        if lang == "en":
            name = rest            # English print — strip "<code> EN-" prefix
        elif lang in NON_EN_LANG:
            return None            # foreign-language print
        # else: not a language code (e.g. a hyphenated real name) — keep as-is
    if REGIONAL_MARKERS.search(name):
        return None
    return strip_era(norm(name))


class RateLimited:
    """Polite GET with min interval + exponential backoff on 403/429/5xx."""
    def __init__(self, min_interval=MIN_INTERVAL, verbose=False):
        self.min_interval = min_interval
        self.last = 0.0
        self.verbose = verbose
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def get_json(self, url, max_retries=4):
        backoff = self.min_interval
        for attempt in range(max_retries):
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
                    return r.json()
                except ValueError:
                    return None
            if r.status_code == 404:
                return None  # genuinely missing; no point retrying
            if r.status_code in (403, 429) or r.status_code >= 500:
                if self.verbose:
                    print(f"    HTTP {r.status_code}; backoff {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
        return None


def build_set_index(psa_sets):
    """{normalized_english_set_name: [(psa_set_id, total, year, raw_name), ...]}
    sorted by total desc. Excludes regional/non-English PSA sets."""
    idx = {}
    for sid, v in psa_sets.items():
        raw = v.get("n", "")
        key = psa_english_key(raw)
        if not key:
            continue
        idx.setdefault(key, []).append(
            (sid, int(v.get("t", 0) or 0), str(v.get("y", "")), raw))
    for k in idx:
        idx[k].sort(key=lambda t: t[1], reverse=True)
    return idx


def match_set(our_name, our_year, set_index, psa_sets):
    """Return (psa_set_id, psa_raw_name, confidence) or (None, None, 'none').

    Strategy: normalized+era-stripped exact key match first; then token-subset
    fallback. Among candidates prefer one whose PSA year matches our release
    year, then the one with the largest population (main English print)."""
    target = strip_era(norm(our_name))
    if not target:
        return None, None, "none"
    yr2 = str(our_year)[-2:] if our_year else None

    cands = list(set_index.get(target, []))
    conf = "high" if cands else None

    if not cands:
        # token-subset fallback: our tokens are a subset of a candidate key
        ttoks = set(target.split())
        if ttoks:
            for key, lst in set_index.items():
                ktoks = set(key.split())
                if ttoks and (ttoks <= ktoks or ktoks <= ttoks):
                    # require meaningful overlap (avoid 1-word coincidences)
                    if len(ttoks & ktoks) >= max(1, min(len(ttoks), len(ktoks))):
                        cands.extend(lst)
        if cands:
            conf = "low"  # fuzzy set match
            cands.sort(key=lambda t: t[1], reverse=True)

    if not cands:
        return None, None, "none"

    # prefer matching year
    if yr2:
        year_hits = [c for c in cands if c[2] == yr2]
        if year_hits:
            cands = year_hits
            if conf == "low":
                conf = "med"  # year corroborates a fuzzy name
    # largest population among remaining
    sid, total, year, raw = max(cands, key=lambda t: t[1])
    return sid, raw, conf


def grade_breakdown(card):
    """Extract {grade_label: count} from a PSA card's `f` dict.
    Grade labels: '1'..'10', halves like '9.5', 'a' (authentic). Skip 't'."""
    out = {}
    f = card.get("f") or {}
    for k, v in f.items():
        if k == "t":
            continue
        if not isinstance(v, dict):
            continue
        g = v.get("g")
        if g is None:
            # qualifier-only entries: count halves/qualifiers if present
            g = v.get("h", 0)
        if g:
            out[k] = int(g)
    return out


def match_card(chase_card, psa_set_cards):
    """Within a PSA set's card list, find the chase card.
    Returns (psa_card_dict, confidence_suffix) where suffix is 'exact'|'fuzzy'|None."""
    target = norm(chase_card["name"])
    if not target:
        return None, None
    ttoks = set(target.split())

    exact, subset = [], []
    for c in psa_set_cards:
        cn = norm(c.get("n", ""))
        if not cn:
            continue
        if cn == target:
            exact.append(c)
        else:
            ctoks = set(cn.split())
            if ttoks and ttoks <= ctoks:  # our name is a subset (e.g. PSA adds "EX")
                subset.append(c)

    want_variant = (chase_card.get("subtype") or "").lower()

    def pick(cands):
        if len(cands) == 1:
            return cands[0]
        # prefer a candidate whose variant list matches our printing
        for c in cands:
            xs = [str(x).lower() for x in (c.get("x") or [])]
            if want_variant and any(want_variant in x or x in want_variant
                                    for x in xs):
                return c
        # else the most-graded one
        return max(cands, key=lambda c: int(c.get("t", 0) or 0))

    if exact:
        return pick(exact), "exact"
    if subset:
        return pick(subset), "fuzzy"
    return None, None


def combine_conf(set_conf, card_suffix, n_card_hits=1):
    """Fold set-match confidence + card-match quality into final label."""
    if set_conf is None or card_suffix is None:
        return "none"
    if set_conf == "high" and card_suffix == "exact" and n_card_hits == 1:
        return "high"
    if set_conf == "none":
        return "none"
    if card_suffix == "exact" and set_conf in ("high", "med"):
        return "med"
    return "low"


def collect(chase_data, limit=None, dry_run=False, out_path=DEFAULT_OUT,
            verbose=False):
    client = RateLimited(verbose=verbose)

    if verbose:
        print(f"Fetching PSA set index: {GM_SETS_URL}")
    psa_sets = client.get_json(GM_SETS_URL)
    if not psa_sets:
        print("FATAL: could not fetch PSA sets.json — community dataset "
              "unreachable from this environment.", file=sys.stderr)
        return None
    cgc_sets = client.get_json(GM_CGC_SETS_URL) or {}
    if verbose:
        print(f"  PSA sets: {len(psa_sets)}  CGC sets: {len(cgc_sets)}")

    set_index = build_set_index(psa_sets)
    cgc_index = build_set_index(cgc_sets) if cgc_sets else {}

    sets = chase_data["sets"]
    # plan: list of (our_set_id, our_set_meta)
    plan = list(sets.items())
    if limit is not None:
        # limit counts CARDS, not sets — walk sets until we hit `limit` cards
        trimmed, n = [], 0
        for sid, meta in plan:
            trimmed.append((sid, meta))
            n += len(meta.get("chase", []))
            if n >= limit:
                break
        plan = trimmed

    if dry_run:
        print("\nDRY RUN — would process:")
        for sid, meta in plan:
            psa_id, psa_name, sconf = match_set(
                meta["set_name"], (meta.get("release_date") or "")[:4], set_index,
                psa_sets)
            cards = meta.get("chase", [])
            tag = f"{psa_id} '{psa_name}' [{sconf}]" if psa_id else "NO SET MATCH"
            print(f"  {sid:<10} {meta['set_name']:<28} -> {tag}  "
                  f"({len(cards)} chase cards)")
        return None

    out = {}
    # resume from existing output if present
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            out = prev.get("sets", {})
            if verbose:
                print(f"  resuming: {len(out)} sets already collected")
        except (ValueError, OSError):
            out = {}

    set_cache = {}  # psa_set_id -> card list
    stats = {"high": 0, "med": 0, "low": 0, "none": 0,
             "psa_with_pop": 0, "cards_total": 0}

    def save():
        if dry_run:
            return
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "GradedMetrics/PokeMetrics community PSA pop mirror "
                      "(github.com/gradedmetrics/api docs/)",
            "stats": stats,
            "sets": out,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        os.replace(tmp, out_path)

    for sid, meta in plan:
        if sid in out:  # already done in a prior run
            continue
        our_year = (meta.get("release_date") or "")[:4]
        psa_id, psa_name, sconf = match_set(
            meta["set_name"], our_year, set_index, psa_sets)

        cgc_total = None
        if cgc_index:
            cgc_id, _, _ = match_set(meta["set_name"], our_year, cgc_index, cgc_sets)
            if cgc_id and cgc_id in cgc_sets:
                cgc_total = int(cgc_sets[cgc_id].get("t", 0) or 0)

        psa_cards = []
        source_url = None
        if psa_id:
            source_url = PSA_SET_PAGE.format(psa_set_id=psa_id)
            if psa_id not in set_cache:
                if verbose:
                    print(f"  {sid}: fetching PSA set {psa_id} '{psa_name}'")
                setdoc = client.get_json(GM_SET_URL.format(psa_set_id=psa_id))
                set_cache[psa_id] = (setdoc or {}).get("c", []) if setdoc else []
            psa_cards = set_cache[psa_id]

        chase_out = []
        for cc in meta.get("chase", []):
            stats["cards_total"] += 1
            psa_card, suffix = (None, None)
            if psa_cards:
                psa_card, suffix = match_card(cc, psa_cards)
            mconf = combine_conf(sconf, suffix)
            psa_total = None
            by_grade = None
            psa_card_name = None
            psa_variants = None
            if psa_card and mconf != "none":
                psa_total = int(psa_card.get("t", 0) or 0)
                by_grade = grade_breakdown(psa_card)
                psa_card_name = psa_card.get("n")
                psa_variants = psa_card.get("x")
                stats["psa_with_pop"] += 1
            stats[mconf] = stats.get(mconf, 0) + 1
            chase_out.append({
                "card_id": cc["card_id"],
                "name": cc["name"],
                "psa_total": psa_total,
                "psa_by_grade": by_grade,
                "cgc_total": None,   # per-card CGC not matched (see docstring)
                "bgs_total": None,   # BGS not in this dataset
                "match_confidence": mconf,
                "source_url": source_url,
                "psa_card_name": psa_card_name,
                "psa_variants": psa_variants,
            })

        out[sid] = {
            "set_name": meta["set_name"],
            "psa_set_id": psa_id,
            "psa_set_name": psa_name,
            "set_match_confidence": sconf or "none",
            "cgc_set_total": cgc_total,
            "chase": chase_out,
        }
        save()  # incremental
        if verbose:
            confs = ",".join(c["match_confidence"] for c in chase_out)
            print(f"  {sid:<10} {meta['set_name']:<26} psa={psa_id} [{confs}]")

    save()
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=DEFAULT_IN,
                    help="chase_cards.json from select_chase_cards.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N chase cards (small test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the set-match plan; no network set fetches / writes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        raise SystemExit(f"chase cards file not found: {args.infile}\n"
                         f"Run select_chase_cards.py first.")
    chase_data = json.load(open(args.infile))

    stats = collect(chase_data, limit=args.limit, dry_run=args.dry_run,
                    out_path=args.out, verbose=args.verbose)
    if stats is None:
        return

    print(f"\nWrote {args.out}")
    print("Match-confidence breakdown (per chase card):")
    for k in ("high", "med", "low", "none"):
        print(f"  {k:<5}: {stats.get(k, 0)}")
    print(f"  cards with a PSA pop number: {stats['psa_with_pop']}"
          f" / {stats['cards_total']}")


if __name__ == "__main__":
    main()
