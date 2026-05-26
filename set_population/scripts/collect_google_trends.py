#!/usr/bin/env python3
"""Collect per-set Google Trends interest for the set_population sub-project.

Google Trends is a comparative, query-relative signal: every
``interest_over_time`` query is normalized to 0-100 *within that query*, and a
query may hold at most 5 terms. That means values from two different queries are
NOT directly comparable -- a "100" in one batch is not the same absolute volume
as a "100" in another.

To stitch all 171 Pokemon sets onto a single common scale we use **anchor
chaining**: one stable, high-volume set ("Base Set") is included as a fixed slot
in *every* batch. Within a batch we read each term's mean interest over the
window, then rescale every term so the anchor reads a fixed value (100) across
all batches. After rescaling, a set's value is its interest *relative to the
anchor*, which is consistent batch-to-batch.

    rescaled(term) = 100 * mean_interest(term) / mean_interest(anchor)   [in that batch]

The anchor itself is forced to exactly 100. If a batch's anchor mean is 0 (the
anchor never registered any interest in that window relative to the other terms)
we cannot rescale that batch -- those sets get null.

Term building
-------------
Raw set names ("Base", "Jungle", "Fossil") collide with unrelated searches, so
we disambiguate by appending the franchise keyword:

    query_term = f"{display_name} pokemon"

where ``display_name`` strips a few noisy tokens (trailing "Set", promo cruft)
but is otherwise the dim_sets ``name``. "pokemon" (lowercase, no accent) is the
most robust disambiguator for Trends, which is accent/case-insensitive. The rule
is intentionally simple and uniform so the anchor is built the same way.

Output: set_population/data/set_trends.json (written incrementally after every
batch so a mid-run HTTP 429 never loses completed work).
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SET_POP_ROOT = HERE.parent
DATA_DIR = SET_POP_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "set_trends.json"

# The anchor: a stable, high-search-volume set present in every batch.
ANCHOR_SET_ID = "base1"  # "Base" -> "Base Set pokemon"
ANCHOR_DISPLAY = "Base Set"
ANCHOR_TERM = "Base Set pokemon"
ANCHOR_VALUE = 100.0  # the anchor is pinned to this on the common scale

WINDOW = "today 5-y"  # pytrends timeframe: last 5 years, weekly resolution
WINDOW_LABEL = "last 5 years (today 5-y)"
GEO = ""  # worldwide
MAX_TERMS_PER_QUERY = 5  # Google Trends hard limit
SETS_PER_BATCH = MAX_TERMS_PER_QUERY - 1  # 1 slot reserved for the anchor

# Rate limiting (Trends 429s aggressively).
# Empirically, Google 429s back-to-back batches at 5-15s spacing but tolerates
# ~40s. Keep a wide jittered window; the run is slow but it's a batch job.
SLEEP_MIN_S = 35.0
SLEEP_MAX_S = 50.0
BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 600.0
MAX_RETRIES = 5

DB_DSN = "cardprice"  # local peer-auth Postgres


# ---------------------------------------------------------------------------
# Set loading + term building
# ---------------------------------------------------------------------------
def load_sets():
    """Return list of (set_id, name, release_date) from dim_sets, read-only.

    Ordered by release_date so batches group temporally-adjacent sets, which
    keeps the dynamic range within a batch reasonable.
    """
    import subprocess

    sql = (
        "SELECT set_id, name, COALESCE(release_date::text, '') "
        "FROM dim_sets ORDER BY release_date NULLS LAST, set_id;"
    )
    out = subprocess.run(
        ["psql", DB_DSN, "-At", "-F", "\t", "-c", sql],
        capture_output=True, text=True, check=True,
    ).stdout
    sets = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        set_id, name = parts[0], parts[1]
        release = parts[2] if len(parts) > 2 else ""
        sets.append((set_id, name, release))
    return sets


# Tokens that add noise to a Trends query without disambiguating.
_NOISE_SUFFIXES = (" Set",)


def build_query_term(name):
    """Disambiguate a raw set name into a Trends query term.

    Rule: take the dim_sets name, normalize a couple of noisy tokens, and append
    the franchise keyword "pokemon". Uniform across all sets including the
    anchor, so the anchor term matches ANCHOR_TERM exactly.
    """
    display = name.strip()
    # "Base" is the franchise origin; canonicalize so it matches the anchor.
    if display == "Base":
        display = "Base Set"
    # Already-mentions-pokemon names: don't double up.
    lower = display.lower()
    if "pokemon" in lower or "pokémon" in lower:
        return display
    return f"{display} pokemon"


def make_batches(set_rows, anchor_term):
    """Group set rows into batches of <=4 set terms, anchor occupies slot 5.

    The anchor set is excluded from the rotating slots (it would be redundant).
    Returns list of batches; each batch is a list of (set_id, name, term) for
    the *non-anchor* members. The anchor is implicitly appended at query time.
    """
    non_anchor = [r for r in set_rows if r[0] != ANCHOR_SET_ID]
    batches = []
    for i in range(0, len(non_anchor), SETS_PER_BATCH):
        chunk = non_anchor[i:i + SETS_PER_BATCH]
        batch = [(sid, name, build_query_term(name)) for sid, name, _ in chunk]
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def init_output(set_rows):
    sets = {}
    for set_id, name, _ in set_rows:
        sets[set_id] = {
            "set_name": name,
            "query_term": build_query_term(name),
            "interest_rescaled": None,
            "raw_batch_mean": None,
        }
    # Anchor is pinned.
    if ANCHOR_SET_ID in sets:
        sets[ANCHOR_SET_ID]["interest_rescaled"] = ANCHOR_VALUE
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "anchor_term": ANCHOR_TERM,
        "anchor_set_id": ANCHOR_SET_ID,
        "window": WINDOW_LABEL,
        "geo": GEO or "worldwide",
        "sets": sets,
    }


def write_output(doc):
    doc["generated_at"] = datetime.now(timezone.utc).isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Trends querying
# ---------------------------------------------------------------------------
def make_pytrends():
    from pytrends.request import TrendReq
    # NOTE: pytrends 4.9.2 builds a urllib3 Retry with the long-removed
    # `method_whitelist` kwarg, which raises on urllib3 >= 2.x. That code path
    # is only entered when retries>0 or backoff_factor>0, so we keep both at 0
    # and do our own exponential backoff in fetch_with_backoff().
    return TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=0, backoff_factor=0)


def fetch_batch_means(pytrends, terms):
    """Query Trends for ``terms`` (anchor included) and return mean interest.

    Returns dict term -> mean over window. Raises on hard failure so the caller
    can apply backoff / mark the batch null.
    """
    pytrends.build_payload(terms, cat=0, timeframe=WINDOW, geo=GEO, gprop="")
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return {}
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    means = {}
    for term in terms:
        if term in df.columns:
            means[term] = float(df[term].mean())
    return means


def rate_limit_sleep():
    t = random.uniform(SLEEP_MIN_S, SLEEP_MAX_S)
    time.sleep(t)


def fetch_with_backoff(pytrends, terms):
    """Fetch with exponential backoff on 429 / transient errors.

    Returns (means_dict, error_str_or_None). means_dict is {} if no usable data.
    """
    attempt = 0
    while True:
        try:
            return fetch_batch_means(pytrends, terms), None
        except Exception as exc:  # noqa: BLE001 - Trends throws many error types
            msg = str(exc)
            is_429 = "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()
            attempt += 1
            if attempt > MAX_RETRIES:
                return {}, f"gave up after {MAX_RETRIES} retries: {msg}"
            if is_429:
                wait = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                wait += random.uniform(0, 5)
                print(f"  [429] backoff {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})",
                      file=sys.stderr)
            else:
                wait = min(BACKOFF_BASE_S * attempt, BACKOFF_MAX_S)
                print(f"  [err] {msg[:120]} -> retry in {wait:.0f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(limit=None, dry_run=False):
    set_rows = load_sets()
    if limit is not None:
        # Always keep the anchor in the working set even under --limit.
        anchor_rows = [r for r in set_rows if r[0] == ANCHOR_SET_ID]
        other = [r for r in set_rows if r[0] != ANCHOR_SET_ID][: max(0, limit - len(anchor_rows))]
        set_rows = anchor_rows + other

    batches = make_batches(set_rows, ANCHOR_TERM)

    print(f"Loaded {len(set_rows)} sets (incl. anchor {ANCHOR_SET_ID!r}).")
    print(f"Anchor term : {ANCHOR_TERM!r}  (pinned to {ANCHOR_VALUE})")
    print(f"Window      : {WINDOW_LABEL}")
    print(f"Batches     : {len(batches)} (<= {SETS_PER_BATCH} sets + anchor each)")
    print()

    for bi, batch in enumerate(batches):
        terms = [ANCHOR_TERM] + [t for _, _, t in batch]
        print(f"Batch {bi + 1}/{len(batches)}: anchor + {len(batch)} sets")
        for sid, name, term in batch:
            print(f"    {sid:>8}  {name:<28}  -> {term!r}")
        if dry_run:
            print(f"    query terms: {terms}")
        print()

    if dry_run:
        print("[dry-run] no API calls made.")
        return

    # ---- real run ----
    try:
        pytrends = make_pytrends()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not init pytrends: {exc}", file=sys.stderr)
        print("Writing stub output with interest: null for all sets.", file=sys.stderr)
        doc = init_output(set_rows)
        write_output(doc)
        return

    doc = init_output(set_rows)
    write_output(doc)  # write stub immediately so partial runs are safe

    any_success = False
    for bi, batch in enumerate(batches):
        terms = [ANCHOR_TERM] + [t for _, _, t in batch]
        print(f"[batch {bi + 1}/{len(batches)}] querying {len(terms)} terms ...")
        means, err = fetch_with_backoff(pytrends, terms)

        if err or not means:
            print(f"  -> no data ({err or 'empty response'}); leaving these sets null.",
                  file=sys.stderr)
            write_output(doc)
            rate_limit_sleep()
            continue

        anchor_mean = means.get(ANCHOR_TERM)
        if not anchor_mean or anchor_mean <= 0:
            print(f"  -> anchor mean is {anchor_mean!r}; cannot rescale this batch.",
                  file=sys.stderr)
            # still record raw means for diagnostics
            for sid, name, term in batch:
                if term in means:
                    doc["sets"][sid]["raw_batch_mean"] = round(means[term], 4)
            write_output(doc)
            rate_limit_sleep()
            continue

        any_success = True
        scale = ANCHOR_VALUE / anchor_mean
        for sid, name, term in batch:
            raw = means.get(term)
            if raw is None:
                continue
            doc["sets"][sid]["raw_batch_mean"] = round(raw, 4)
            doc["sets"][sid]["interest_rescaled"] = round(raw * scale, 4)
            print(f"    {sid:>8}  raw={raw:6.2f}  rescaled={raw * scale:7.2f}")

        # record anchor's own raw mean for transparency
        doc["sets"].setdefault(ANCHOR_SET_ID, {})
        if ANCHOR_SET_ID in doc["sets"]:
            doc["sets"][ANCHOR_SET_ID]["raw_batch_mean"] = round(anchor_mean, 4)

        write_output(doc)
        rate_limit_sleep()

    write_output(doc)
    n_filled = sum(1 for s in doc["sets"].values() if s["interest_rescaled"] is not None)
    print()
    print(f"Done. {n_filled}/{len(doc['sets'])} sets have a rescaled value.")
    print(f"Output: {OUTPUT_PATH}")
    if not any_success:
        print("WARNING: no batch returned usable data -- Trends likely blocked/"
              "rate-limited in this environment.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Collect per-set Google Trends interest.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the batching plan without calling the API.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N sets (anchor always included).")
    args = ap.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
