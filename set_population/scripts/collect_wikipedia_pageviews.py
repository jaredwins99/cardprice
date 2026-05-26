#!/usr/bin/env python3
"""Collect per-set demand signal from Wikipedia pageviews.

Part of the `set_population/` sub-project. Builds an EXOGENOUS popularity proxy
per Pokemon TCG set from English-Wikipedia pageviews, summed over a fixed recent
window so every set is compared over the same period.

Why Wikipedia and not Bulbapedia for the *quantitative* signal
--------------------------------------------------------------
Bulbapedia DOES have a per-set article (`<Set Name> (TCG)`) and we query its
MediaWiki API to *confirm* the article exists. But Bulbapedia does NOT expose
pageview data: the MediaWiki `PageViewInfo` extension is not installed there
(verified: `prop=pageviews` returns "Unrecognized value"), and Bulbapedia is not
covered by the Wikimedia Pageviews REST API (that API serves Wikimedia
Foundation projects only, i.e. *.wikipedia.org etc.). So Bulbapedia gives us a
boolean "article exists" but no counts, and the numbers come from en.wikipedia
via the Wikimedia Pageviews API.

Article resolution (per set), in order:
  1. set_article:  Try `<Name> (Pokémon)` on en.wikipedia. Accept only if the
     (redirect-resolved) target is a STANDALONE article -- i.e. not the merged
     "List of Pokémon Trading Card Game sets" page that almost every individual
     set redirects to. Most TCG sets do NOT have a standalone article.
  2. mascot_card:  Fall back to the set's flagship / mascot Pokemon species
     article (MASCOTS map below). Accept only if standalone (many species
     redirect to "List of generation N Pokémon"). Tagged proxy_type=mascot_card.
  3. none:         No usable article -> pageviews null.

Pageviews are pulled from:
  https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
      en.wikipedia/all-access/user/<ARTICLE>/monthly/<START>/<END>
summed over the window. For sets released mid-window we still divide by the
number of *calendar months in the window that the set had been released*, so
pageviews_per_month is comparable across old and new sets.

Outputs set_population/data/set_pageviews.json. Read-only on dim_sets.
stdlib + requests only.
"""

import argparse
import json
import sys
import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import requests

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

# --- paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUT_PATH = DATA_DIR / "set_pageviews.json"

# --- API config ------------------------------------------------------------
USER_AGENT = (
    "cardprice-set-population/1.0 (https://github.com/; godlikehydraa@gmail.com) "
    "research bot; per-set demand-signal collection"
)
HEADERS = {"User-Agent": USER_AGENT}
WP_API = "https://en.wikipedia.org/w/api.php"
BULBA_API = "https://bulbapedia.bulbagarden.net/w/api.php"
PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/monthly/{start}/{end}"
)
RATE_LIMIT_SEC = 1.0  # polite ~1 req/sec to the Wikimedia APIs

# Targets that, when reached via redirect, mean the set has NO standalone
# article -- the set was merged into a list/overview page. Reject these.
NON_STANDALONE_TARGETS = {
    "List of Pokémon Trading Card Game sets",
    "Pokémon Trading Card Game",
    "Pokémon Trading Card Game Live",
    "Pokémon Trading Card Game (video game)",
}
NON_STANDALONE_PREFIXES = ("List of",)

DB_DSN = "dbname=cardprice"  # peer auth as current user

# Extra en.wikipedia title candidates for sets whose dim_sets name differs from
# the article title, or that are known to have a standalone TCG-set article.
# These are still content-validated (TCG keyword check) before being accepted,
# so a stale entry just degrades to the mascot fallback. Keyed by set_id.
WP_TITLE_ALIASES = {
    "base1": ["Base Set (Pokémon)", "Base Set (Pokémon Trading Card Game)"],
}

# --- mascot fallback map ---------------------------------------------------
# Flagship / chase / mascot Pokemon per set, used ONLY when the set has no
# standalone Wikipedia article. Value is the en.wikipedia species article title
# we try; it is still validated as standalone before use, so a bad guess just
# degrades to proxy_type="none" rather than producing a garbage number.
# Species with their own standalone articles on en.wikipedia are the strong
# anchors (Charizard, Mewtwo, Pikachu, Mew, Lugia, Greninja, etc.); many other
# species redirect to "List of generation N Pokémon" and will be rejected.
MASCOTS = {
    "base1": "Charizard", "base2": "Pikachu", "base3": "Aerodactyl",
    "base4": "Charizard", "base5": "Mewtwo", "base6": "Charizard",
    "basep": "Pikachu",
    "gym1": "Pikachu", "gym2": "Charizard",
    "neo1": "Lugia", "neo2": "Pikachu", "neo3": "Lugia", "neo4": "Mewtwo",
    "si1": "Mew", "col1": "Pikachu",
    "ecard1": "Charizard", "ecard2": "Lugia", "ecard3": "Charizard",
    "bp": "Pikachu",
    "ex1": "Mewtwo", "ex2": "Pikachu", "ex3": "Rayquaza",
    "ex4": "Pikachu", "ex5": "Pikachu", "ex6": "Charizard",
    "ex7": "Mewtwo", "ex8": "Rayquaza", "ex9": "Pikachu",
    "ex10": "Lugia", "ex11": "Mewtwo", "ex12": "Mew",
    "ex13": "Pikachu", "ex14": "Pikachu", "ex15": "Charizard",
    "ex16": "Pikachu",
    "np": "Pikachu",
    "pop1": "Pikachu", "pop2": "Pikachu", "pop3": "Pikachu", "pop4": "Pikachu",
    "pop5": "Pikachu", "pop6": "Pikachu", "pop7": "Pikachu", "pop8": "Pikachu",
    "pop9": "Pikachu",
    "tk1a": "Mew", "tk1b": "Mew", "tk2a": "Pikachu", "tk2b": "Pikachu",
    "dp1": "Pikachu", "dp2": "Pikachu", "dp3": "Pikachu", "dp4": "Pikachu",
    "dp5": "Pikachu", "dp6": "Pikachu", "dp7": "Charizard", "dpp": "Pikachu",
    "pl1": "Pikachu", "pl2": "Pikachu", "pl3": "Pikachu", "pl4": "Pikachu",
    "ru1": "Pikachu",
    "hgss1": "Lugia", "hgss2": "Pikachu", "hgss3": "Pikachu",
    "hgss4": "Charizard", "hsp": "Pikachu",
    "bw1": "Pikachu", "bw2": "Pikachu", "bw3": "Pikachu", "bw4": "Mewtwo",
    "bw5": "Charizard", "bw6": "Charizard", "bw7": "Pikachu", "bw8": "Pikachu",
    "bw9": "Pikachu", "bw10": "Pikachu", "bw11": "Charizard", "bwp": "Pikachu",
    "dv1": "Rayquaza",
    "mcd11": "Pikachu", "mcd12": "Pikachu", "mcd14": "Pikachu",
    "mcd15": "Pikachu", "mcd16": "Pikachu", "mcd17": "Pikachu",
    "mcd18": "Pikachu", "mcd19": "Pikachu", "mcd21": "Pikachu",
    "mcd22": "Pikachu",
    "xy0": "Pikachu", "xy1": "Pikachu", "xy2": "Charizard", "xy3": "Pikachu",
    "xy4": "Gengar", "xy5": "Kyogre", "xy6": "Rayquaza", "xy7": "Pikachu",
    "xy8": "Pikachu", "xy9": "Greninja", "xy10": "Zygarde", "xy11": "Pikachu",
    "xy12": "Charizard", "xyp": "Pikachu",
    "g1": "Charizard", "dc1": "Pikachu", "det1": "Pikachu",
    "sm1": "Pikachu", "sm2": "Pikachu", "sm3": "Charizard", "sm4": "Pikachu",
    "sm5": "Pikachu", "sm6": "Pikachu", "sm7": "Pikachu", "sm8": "Pikachu",
    "sm9": "Pikachu", "sm10": "Pikachu", "sm11": "Mewtwo", "sm12": "Pikachu",
    "sm35": "Pikachu", "sm75": "Charizard", "sm115": "Charizard",
    "sma": "Charizard", "smp": "Pikachu",
    "swsh1": "Pikachu", "swsh2": "Pikachu", "swsh3": "Charizard",
    "swsh4": "Pikachu", "swsh5": "Pikachu", "swsh6": "Pikachu",
    "swsh7": "Rayquaza", "swsh8": "Mew", "swsh9": "Charizard",
    "swsh10": "Pikachu", "swsh11": "Giratina", "swsh12": "Lugia",
    "swsh35": "Charizard", "swsh45": "Charizard", "swsh45sv": "Charizard",
    "swsh9tg": "Charizard", "swsh10tg": "Pikachu", "swsh11tg": "Giratina",
    "swsh12tg": "Lugia", "swsh12pt5": "Charizard", "swsh12pt5gg": "Charizard",
    "swshp": "Pikachu", "cel25": "Charizard", "cel25c": "Charizard",
    "fut20": "Pikachu", "pgo": "Pikachu",
    "sv1": "Pikachu", "sv2": "Charizard", "sv3": "Charizard",
    "sv3pt5": "Mew", "sv4": "Pikachu", "sv4pt5": "Charizard",
    "sv5": "Pikachu", "sv6": "Greninja", "sv6pt5": "Pikachu",
    "sv7": "Pikachu", "sv8": "Pikachu", "sv8pt5": "Eevee",
    "sv9": "Charizard", "sv10": "Pikachu", "svp": "Pikachu", "sve": "Pikachu",
    "zsv10pt5": "Pikachu", "rsv10pt5": "Charizard",
    "me1": "Pikachu", "me2": "Charizard", "me2pt5": "Pikachu",
}


# --- helpers ---------------------------------------------------------------
def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_sets():
    """Read (set_id, name, release_date) from dim_sets, read-only."""
    rows = []
    if psycopg2 is not None:
        try:
            conn = psycopg2.connect(DB_DSN)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT set_id, name, release_date FROM dim_sets "
                    "ORDER BY release_date NULLS LAST, set_id"
                )
                for sid, name, rel in cur.fetchall():
                    rows.append((sid, name, rel.isoformat() if rel else None))
            finally:
                conn.close()
            return rows
        except Exception as e:  # fall back to psql
            log(f"[warn] psycopg2 connect failed ({e}); shelling out to psql")
    # psql fallback
    import subprocess

    out = subprocess.run(
        [
            "psql", "cardprice", "-t", "-A", "-F", "\t", "-c",
            "SELECT set_id, name, release_date FROM dim_sets "
            "ORDER BY release_date NULLS LAST, set_id",
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        sid, name = parts[0], parts[1]
        rel = parts[2] if len(parts) > 2 and parts[2] else None
        rows.append((sid, name, rel))
    return rows


def is_standalone(target_title: str) -> bool:
    """True if a resolved article title is a real standalone article (not a
    merged list/overview page)."""
    if target_title in NON_STANDALONE_TARGETS:
        return False
    for pfx in NON_STANDALONE_PREFIXES:
        if target_title.startswith(pfx):
            return False
    return True


class WikiClient:
    def __init__(self, rate=RATE_LIMIT_SEC):
        self.rate = rate
        self.sess = requests.Session()
        self.sess.headers.update(HEADERS)
        self._last = 0.0

    def _throttle(self):
        dt = time.time() - self._last
        if dt < self.rate:
            time.sleep(self.rate - dt)
        self._last = time.time()

    def resolve_title(self, title, want_intro=False):
        """Resolve a candidate en.wikipedia title following redirects.

        Returns (resolved_title | None, standalone: bool, intro: str).
        resolved_title is None if the page is missing. intro is the lowercased
        lead extract (only populated when want_intro=True).
        """
        self._throttle()
        params = {
            "action": "query", "format": "json",
            "titles": title, "redirects": 1, "prop": "info",
        }
        if want_intro:
            params.update({
                "prop": "info|extracts", "exintro": 1,
                "explaintext": 1, "exsentences": 4,
            })
        try:
            r = self.sess.get(WP_API, params=params, timeout=25)
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            for _pid, p in pages.items():
                if "missing" in p:
                    return None, False, ""
                t = p["title"]
                intro = p.get("extract", "").lower() if want_intro else ""
                return t, is_standalone(t), intro
        except Exception as e:
            log(f"[warn] WP resolve {title!r} failed: {e}")
        return None, False, ""

    def bulbapedia_exists(self, title):
        """Confirm a Bulbapedia article exists (no pageviews available there).

        Returns the resolved Bulbapedia title or None.
        """
        self._throttle()
        try:
            r = self.sess.get(
                BULBA_API,
                params={
                    "action": "query", "format": "json",
                    "titles": title, "redirects": 1,
                },
                timeout=25,
            )
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            for _pid, p in pages.items():
                if "missing" in p:
                    return None
                return p["title"]
        except Exception as e:
            log(f"[warn] Bulbapedia check {title!r} failed: {e}")
        return None

    def monthly_pageviews(self, article, start, end):
        """Sum monthly en.wikipedia pageviews for article over [start, end].

        Returns (total_views:int, n_months_with_data:int). Returns (0,0) on a
        clean 404 (article has no pageview record in window).
        """
        self._throttle()
        enc = urllib.parse.quote(article, safe="")
        url = PAGEVIEWS_API.format(article=enc, start=start, end=end)
        try:
            r = self.sess.get(url, timeout=25)
            if r.status_code == 404:
                return 0, 0
            r.raise_for_status()
            items = r.json().get("items", [])
            total = sum(int(it.get("views", 0)) for it in items)
            return total, len(items)
        except Exception as e:
            log(f"[warn] pageviews {article!r} failed: {e}")
            return None, 0


# --- window math -----------------------------------------------------------
def month_floor(d: date):
    return date(d.year, d.month, 1)


def months_between(a: date, b: date):
    """Number of calendar months from a to b inclusive (a<=b)."""
    return (b.year - a.year) * 12 + (b.month - a.month) + 1


def build_window(months: int):
    """Return (start_yyyymm, end_yyyymm, start_date, end_date) for the last
    `months` complete months ending last month (avoid the partial current
    month)."""
    today = date.today()
    # end = first day of previous month
    if today.month == 1:
        end = date(today.year - 1, 12, 1)
    else:
        end = date(today.year, today.month - 1, 1)
    # start = end shifted back (months-1) months
    sy, sm = end.year, end.month - (months - 1)
    while sm <= 0:
        sm += 12
        sy -= 1
    start = date(sy, sm, 1)
    return (
        start.strftime("%Y%m"), end.strftime("%Y%m"), start, end,
    )


# --- resolution plan -------------------------------------------------------
def resolve_set(client: WikiClient, set_id, name):
    """Resolve a set to (resolved_article, proxy_type, bulbapedia_title).

    proxy_type in {"set_article", "mascot_card", "none"}.
    """
    # Confirm Bulbapedia set article (existence only; informational).
    bulba = client.bulbapedia_exists(f"{name} (TCG)")

    # 1) standalone set article on en.wikipedia. A bare title match is NOT
    # enough: `<Name> (Pokémon)` frequently lands on a species article (Arceus),
    # the manga ("Pokémon Adventures"), or an unrelated topic ("Team Rocket").
    # Require the lead paragraph to describe a TCG expansion/set so we only tag
    # proxy_type=set_article when the article is really about the card set.
    TCG_KW = ("trading card", "tcg", "expansion", "card set",
              "set of cards", "pokémon card")
    candidates = list(WP_TITLE_ALIASES.get(set_id, []))
    candidates += [f"{name} (Pokémon)",
                   f"{name} (Pokémon Trading Card Game)",
                   f"{name} (TCG)"]
    for cand in candidates:
        title, standalone, intro = client.resolve_title(cand, want_intro=True)
        if title and standalone and any(k in intro for k in TCG_KW):
            return title, "set_article", bulba

    # 2) mascot fallback -- the set's flagship Pokemon species article.
    mascot = MASCOTS.get(set_id)
    if mascot:
        title, standalone, _ = client.resolve_title(mascot)
        if title and standalone:
            return title, "mascot_card", bulba

    # 3) nothing usable
    return None, "none", bulba


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=24,
                    help="window length in months (default 24)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve articles only (cap at 5 pageview lookups), "
                         "print the plan, do not write output")
    ap.add_argument("--rate", type=float, default=RATE_LIMIT_SEC,
                    help="seconds between API requests (default 1.0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N sets (debug)")
    args = ap.parse_args()

    start_ym, end_ym, start_d, end_d = build_window(args.months)
    start_api = start_ym + "0100"
    end_api = end_ym + "0100"
    log(f"[info] window {start_ym}..{end_ym} ({args.months} months)")

    sets = load_sets()
    if args.limit:
        sets = sets[: args.limit]
    log(f"[info] loaded {len(sets)} sets from dim_sets")

    client = WikiClient(rate=args.rate)

    # ---- DRY RUN: resolve only ----
    if args.dry_run:
        counts = {"set_article": 0, "mascot_card": 0, "none": 0}
        plan = []
        pv_lookups = 0
        for sid, name, rel in sets:
            article, ptype, bulba = resolve_set(client, sid, name)
            counts[ptype] += 1
            sample_views = None
            # cap at 5 actual pageview lookups to show the data is live
            if article and pv_lookups < 5:
                total, n = client.monthly_pageviews(article, start_api, end_api)
                sample_views = total
                pv_lookups += 1
            plan.append((sid, name, article, ptype, bulba, sample_views))

        print("=== DRY RUN: article resolution plan ===")
        print(f"window: {start_ym}..{end_ym}  ({args.months} months)")
        print(f"{'set_id':12} {'proxy':12} {'wikipedia_article':40} bulbapedia")
        print("-" * 100)
        for sid, name, article, ptype, bulba, sv in plan:
            sv_s = f"  [views~{sv}]" if sv is not None else ""
            print(f"{sid:12} {ptype:12} {str(article):40} "
                  f"{str(bulba)}{sv_s}")
        print("-" * 100)
        print(f"resolved (set_article): {counts['set_article']}")
        print(f"fallback (mascot_card): {counts['mascot_card']}")
        print(f"none:                   {counts['none']}")
        print(f"total sets:             {len(sets)}")
        print(f"(sampled {pv_lookups} live pageview lookups)")
        return

    # ---- REAL RUN ----
    out_sets = {}
    counts = {"set_article": 0, "mascot_card": 0, "none": 0}
    real_views_n = 0
    for i, (sid, name, rel) in enumerate(sets, 1):
        article, ptype, bulba = resolve_set(client, sid, name)
        counts[ptype] += 1

        pv_total = None
        pv_per_month = None
        if article:
            total, n_months = client.monthly_pageviews(article, start_api, end_api)
            if total is not None:
                pv_total = total
                # months-available normalization: a set released mid-window
                # should be divided only by the months it existed in-window.
                avail = args.months
                if rel:
                    try:
                        rel_d = month_floor(datetime.strptime(rel, "%Y-%m-%d").date())
                        if rel_d > start_d:
                            avail = months_between(rel_d, end_d)
                    except ValueError:
                        pass
                avail = max(1, min(args.months, avail))
                pv_per_month = round(total / avail, 2)
                if total > 0:
                    real_views_n += 1

        out_sets[sid] = {
            "set_name": name,
            "release_date": rel,
            "resolved_article": article,
            "proxy_type": ptype,
            "bulbapedia_article": bulba,
            "pageviews_total": pv_total,
            "pageviews_per_month": pv_per_month,
        }
        log(f"[{i}/{len(sets)}] {sid:12} {ptype:12} "
            f"{str(article):35} total={pv_total} /mo={pv_per_month}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start_ym, "end": end_ym, "months": args.months},
        "source": {
            "pageviews": "Wikimedia Pageviews REST API (en.wikipedia, "
                         "all-access, user agents)",
            "article_existence_crosscheck": "Bulbapedia MediaWiki API "
                                             "(no pageview data available there)",
            "user_agent": USER_AGENT,
        },
        "coverage": {
            "total_sets": len(sets),
            "set_article": counts["set_article"],
            "mascot_card": counts["mascot_card"],
            "none": counts["none"],
            "with_nonzero_pageviews": real_views_n,
        },
        "sets": out_sets,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    log(f"[done] wrote {OUT_PATH}")
    log(f"[done] coverage: set_article={counts['set_article']} "
        f"mascot_card={counts['mascot_card']} none={counts['none']} "
        f"(nonzero pageviews: {real_views_n})")


if __name__ == "__main__":
    main()
