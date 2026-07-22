#!/usr/bin/env python3
"""Capstone combiner for the set_population sub-project.

Joins every collected signal into a relative (and lightly-calibrated absolute)
print-population estimate per Pokemon set:

    rel_pop(set) = mean_chase_psa(set) x pull_denominator(set)
                 / predicted_grading_rate(set)

where `predicted_grading_rate` comes from the mechanistic model fit in
`scripts/fit_grading_rate.py` and stored in `data/grading_rate_model.json`:

    log(grading_rate(set)) = alpha[era]
                           + beta_p * log(chase_value / $100)
                           + beta_y * log(years_since_release / 10y)

Replaces the older Google-Trends popularity divisor (model v1), which was
order-of-magnitude on cross-era absolutes because search popularity does not
capture the value- and age-driven grading-rate dynamics. Trends, pageviews and
sales velocity are now held out as VALIDATION signals (Spearman cross-check).

Absolute calibration (v3.1): credibility-weighted geometric-mean scale over
the usable rungs of the dated TPC checkpoint ladder, English-converted on
per-regime increments, with production-ramped windows and subset products
excluded. Outputs are ENGLISH-ONLY projected lifetime production; unreliable
rows are flagged and their absolutes suppressed.

Every magic constant is defined + documented below and in docs/methodology.md.
The script is fully re-runnable (idempotent). Read-only on dim_sets.

Outputs:
  data/set_population_estimates.json
  docs/results.md

stdlib + numpy/scipy (already present) + psycopg2 only. No new heavy deps.
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone

import psycopg2

try:
    from scipy.stats import spearmanr as _scipy_spearman
except Exception:  # pragma: no cover - fallback path
    _scipy_spearman = None

# --- paths -------------------------------------------------------------------
SUBPROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(SUBPROJ, "data")
DOCS = os.path.join(SUBPROJ, "docs")
PG_DSN = "dbname=cardprice"  # peer auth via unix socket, read-only use

CHASE_GRADED = os.path.join(DATA, "chase_graded_pop.json")
CHASE_CARDS = os.path.join(DATA, "chase_cards.json")
TRENDS = os.path.join(DATA, "set_trends.json")
SALES = os.path.join(DATA, "set_sales_velocity.json")
PAGEVIEWS = os.path.join(DATA, "set_pageviews.json")
KNOWN_RUNS = os.path.join(DATA, "known_print_runs.json")
GRADING_MODEL = os.path.join(DATA, "grading_rate_model.json")
OUT_JSON = os.path.join(DATA, "set_population_estimates.json")
OUT_MD = os.path.join(DOCS, "results.md")

from model_constants_v3 import (  # noqa: E402
    MODEL_VERSION_V3, PRODUCTION_RAMP_DAYS, PRODUCTION_RAMP_DAYS_BOOM,
    ROSTER_LAG_DAYS, SUBSET_PARENT, anchor_value_english, calibration_scale,
    english_share, load_checkpoints, production_weight, share_doc,
    usable_rungs)

MODEL_VERSION = MODEL_VERSION_V3

# =============================================================================
# DOCUMENTED MODEL CONSTANTS  (see docs/methodology.md). All ESTIMATES.
# =============================================================================

# --- 1. Numerator ------------------------------------------------------------
CONFIDENCE_KEEP = {"high", "med"}  # exclude 'low' chase->pop matches

# --- 2. Pull-rate denominator: era buckets by release_date -------------------
# (era_name, start_date_inclusive). Sorted ascending; a set's era is the last
# bucket whose start <= release_date.
ERA_BUCKETS = [
    ("WOTC", date(1999, 1, 1)),
    ("ECARD", date(2002, 9, 1)),
    ("EX", date(2003, 6, 1)),
    ("DP", date(2007, 4, 1)),
    ("HGSS", date(2010, 2, 1)),
    ("BW", date(2011, 4, 1)),
    ("XY", date(2013, 10, 1)),
    ("SM", date(2017, 2, 1)),
    ("SWSH", date(2020, 2, 1)),
    ("SV", date(2023, 3, 1)),
]

# D[era][tier] = packs-opened-per-graded-chase-copy, relative units, WOTC
# tier-3 holo = 1.0. ESTIMATES anchored to community pull-rate data (prior_art
# #3). Deliberately conservative (< raw pull-rate inverse) so we don't double
# count value-driven grading inflation that the popularity divisor removes.
#
# IMPORTANT CALIBRATION CHOICE (see methodology.md & results.md "Confidence"):
# Prior art (prior_art.md) says modern sets are "10-100x larger per set" than
# vintage and ~60% of all cards were printed since FY2020-21 -- NOT 1000x. An
# earlier draft used tier-5 SV = 200, which made modern sets soak up ~80% of the
# 75B and pushed WOTC anchors ~1000x too low. We therefore CAP the modern apex
# multiplier near ~30-40x the WOTC holo so the implied era split (WOTC billions,
# modern era dominant but not total) matches both the anchors and the official
# back-loading. The raw pull-rate inverse of a 1:200 SIR is much larger; the gap
# is intentional and reflects that vintage holos are graded at a FAR higher rate
# (a value-driven bias the popularity divisor only partly removes cross-era).
PULL_DENOM = {
    #          t0    t1    t2    t3    t4     t5
    "WOTC":  [1.0,  1.0,  1.5,  1.0,  3.0,   6.0],
    "ECARD": [1.0,  1.0,  1.5,  1.1,  3.5,   7.0],
    "EX":    [1.0,  1.0,  1.5,  1.2,  4.0,   9.0],
    "DP":    [1.0,  1.0,  1.5,  1.2,  4.5,  11.0],
    "HGSS":  [1.0,  1.0,  1.5,  1.2,  5.0,  13.0],
    "BW":    [1.0,  1.0,  1.5,  1.3,  5.5,  15.0],
    "XY":    [1.0,  1.0,  1.5,  1.3,  6.0,  18.0],
    "SM":    [1.0,  1.0,  1.5,  1.4,  7.0,  22.0],
    "SWSH":  [1.0,  1.0,  1.5,  1.5,  8.0,  28.0],
    "SV":    [1.0,  1.0,  1.5,  1.6,  9.0,  35.0],
}

# --- 3. Grading-rate divisor (v3 model) --------------------------------------
# The mechanistic grading-rate model lives in scripts/fit_grading_rate.py and
# writes data/grading_rate_model.json; this combiner reads per-set
# predicted_grading_rate from it. Trends, pageviews and sales velocity are
# VALIDATION signals only (Spearman cross-checks); none feed rel_pop.
# Sales velocity is endogenous (population -> supply -> sales).

# --- 5. Absolute calibration -------------------------------------------------
# v3: the checkpoint ladder is loaded from known_print_runs.json GLOBAL
# anchors (16 dated rungs, 12 official) and converted to ENGLISH cards via
# english_share(); windows use the production ramp. See model_constants_v3.py.
# Output absolutes are ENGLISH-ONLY projected lifetime production.
# Multiplicative uncertainty band: grading-rate (~3x) x pull-rate (~2-3x),
# collapsed to one factor. Bands are order-of-magnitude rails, not CIs.
BAND_FACTOR = 3.0


# =============================================================================
# helpers
# =============================================================================
def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.fromisoformat(s[:10]).date()
        except ValueError:
            return None


def era_of(rel_date):
    if rel_date is None:
        return None
    era = ERA_BUCKETS[0][0]
    for name, start in ERA_BUCKETS:
        if rel_date >= start:
            era = name
        else:
            break
    return era


def zscore(values):
    """Return dict-free z-scores for a list. Constant input -> all zeros."""
    if not values:
        return []
    mu = statistics.mean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def spearman(xs, ys):
    """Spearman rank correlation. Uses scipy if available, else by-hand
    (average-rank ties, Pearson on ranks)."""
    if len(xs) < 3:
        return None
    if _scipy_spearman is not None:
        rho, _ = _scipy_spearman(xs, ys)
        return float(rho)

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# =============================================================================
# data loaders
# =============================================================================
def load_release_dates():
    """set_id -> (name, release_date) from dim_sets (read-only)."""
    out = {}
    conn = psycopg2.connect(PG_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT set_id, name, release_date FROM dim_sets")
        for sid, name, rel in cur.fetchall():
            rd = rel if isinstance(rel, date) else parse_date(str(rel) if rel else None)
            out[sid] = (name, rd)
    finally:
        conn.close()
    return out


# =============================================================================
# model
# =============================================================================
def compute_numerator(graded_set):
    """Return (mean_psa, n_used) over confident, non-null chase cards."""
    pops = []
    for c in graded_set.get("chase", []):
        if c.get("match_confidence") in CONFIDENCE_KEEP and c.get("psa_total") is not None:
            pops.append(c["psa_total"])
    if not pops:
        return None, 0
    return statistics.mean(pops), len(pops)


def chase_max_tier(chase_set):
    """Max rarity tier among a set's chase cards (0-5). chase_cards.json stores
    selection.tier_used; fall back to per-card tier if present."""
    sel = chase_set.get("selection", {})
    t = sel.get("tier_used")
    if t is not None:
        return int(t)
    tiers = [c.get("tier") for c in chase_set.get("chase", []) if c.get("tier") is not None]
    return max(tiers) if tiers else 0


def pull_denominator(era, tier):
    table = PULL_DENOM.get(era)
    if table is None:
        return 1.0
    tier = max(0, min(tier, len(table) - 1))
    return table[tier]


def build_estimates(args):
    graded = load_json(CHASE_GRADED)["sets"]
    chase = load_json(CHASE_CARDS)["sets"]
    trends = load_json(TRENDS)["sets"]
    sales = load_json(SALES)["sets"]
    pageviews = load_json(PAGEVIEWS)["sets"]
    rel_dates = load_release_dates()

    # Load the fitted grading-rate model (v2 divisor).
    if not os.path.exists(GRADING_MODEL):
        raise SystemExit(
            f"Missing {GRADING_MODEL}. Run scripts/fit_grading_rate.py first.")
    grading_model = load_json(GRADING_MODEL)
    per_set_model = grading_model["per_set"]
    model_meta = grading_model["model"]

    sets = {}  # set_id -> working record

    # --- pass 1: numerator, era, tier, raw signals --------------------------
    for sid in graded:
        mean_psa, n_used = compute_numerator(graded[sid])
        cc = chase.get(sid, {})
        name = cc.get("set_name") or rel_dates.get(sid, (sid, None))[0] or sid
        rd = rel_dates.get(sid, (None, None))[1]
        if rd is None:
            rd = parse_date(cc.get("release_date"))
        era = era_of(rd)
        tier = chase_max_tier(cc) if cc else 0
        flags = []
        if n_used == 0:
            flags.append("no_chase_pop")
        if cc.get("selection", {}).get("tier_mixed"):
            flags.append("tier_mixed")
        if cc.get("selection", {}).get("fallback"):
            flags.append("tier_fallback")
        if era is None:
            flags.append("no_era")

        tr = trends.get(sid, {})
        interest = tr.get("interest_rescaled")
        pv = pageviews.get(sid, {})
        pv_pm = pv.get("pageviews_per_month")
        pv_proxy = pv.get("proxy_type")
        sv = sales.get(sid, {})

        # Grading-rate from the fitted model (None if not modelled).
        gm = per_set_model.get(sid, {})
        sets[sid] = {
            "set_name": name,
            "era": era,
            "release_date": rd.isoformat() if rd else None,
            "n_chase_used": n_used,
            # raw value used in arithmetic; rounded only at output time
            "mean_chase_psa": mean_psa,
            "chase_tier": tier,
            "pull_denominator": round(pull_denominator(era, tier), 3) if era else 1.0,
            "chase_value": gm.get("chase_value"),
            "chase_value_imputed": gm.get("chase_value_imputed", False),
            "predicted_grading_rate": gm.get("predicted_grading_rate"),
            "_interest": interest,
            "_pv_pm": pv_pm if pv_proxy == "set_article" else None,
            "interest_rescaled": interest,  # kept for output transparency
            "pageviews_per_month": pv_pm if pv_proxy == "set_article" else None,
            "sales_per_month": sv.get("sales_per_month"),
            "flags": flags,
        }
        if sets[sid]["predicted_grading_rate"] is None:
            sets[sid]["flags"].append("no_grading_rate")

    # --- pass 2: also compute the old popularity z-score for transparency
    # (we report it in the JSON so the validation Spearman can use it).
    log_trends = {sid: math.log1p(r["_interest"]) for sid, r in sets.items()
                  if r["_interest"] is not None}
    if log_trends:
        ids = list(log_trends)
        zs = zscore([log_trends[i] for i in ids])
        trend_z = dict(zip(ids, zs))
    else:
        trend_z = {}
    for sid, r in sets.items():
        r["trend_z"] = round(trend_z.get(sid), 4) if sid in trend_z else None

    # --- pass 3: relative population (divide by predicted grading rate) -----
    for sid, r in sets.items():
        if (r["n_chase_used"] == 0 or r["mean_chase_psa"] is None
                or r["predicted_grading_rate"] is None
                or r["predicted_grading_rate"] <= 0):
            r["rel_pop_score"] = None
            continue
        r["rel_pop_score"] = (
            r["mean_chase_psa"] * r["pull_denominator"]
            / r["predicted_grading_rate"]
        )

    scored = {sid: r for sid, r in sets.items() if r["rel_pop_score"] is not None}

    # normalise rel_pop so max = 100 (readability)
    if scored:
        mx = max(r["rel_pop_score"] for r in scored.values())
        for r in scored.values():
            r["rel_pop_norm100"] = round(100.0 * r["rel_pop_score"] / mx, 4)

    # --- pass 4: absolute calibration (v3.1: geomean over usable rungs) -----
    calib = None
    if not args.no_absolute and scored:
        pop_snapshot = parse_date(load_json(CHASE_GRADED).get("generated_at")) \
            or date.today()
        rungs = usable_rungs(load_checkpoints(load_json(KNOWN_RUNS)), pop_snapshot)
        # ramped window sums (subset products excluded — their production is
        # inside their parent's sealed product)
        def wsum_at(cp_date):
            return sum(r["rel_pop_score"] * production_weight(
                           parse_date(r["release_date"]), cp_date)
                       for sid, r in scored.items()
                       if r["release_date"] and sid not in SUBSET_PARENT)
        rung_terms = [(c["value_english"], wsum_at(c["date"]), c["weight"])
                      for c in rungs]
        scale = calibration_scale(rung_terms) if rung_terms else None
        calib = {
            "method": "credibility-weighted geometric-mean scale over usable rungs",
            "n_rungs_used": len(rungs),
            "latest_rung_date": rungs[-1]["date"].isoformat() if rungs else None,
            "latest_rung_value_global": rungs[-1]["value_global"] if rungs else None,
            "latest_rung_value_english": rungs[-1]["value_english"] if rungs else None,
            "roster_lag_days": ROSTER_LAG_DAYS,
            "rung_ratios": [
                {"date": c["date"].isoformat(),
                 "target_english": c["value_english"],
                 "ratio_est_over_target": (ws * scale / c["value_english"])
                 if (scale and c["value_english"]) else None}
                for c, (t, ws, w) in zip(rungs, rung_terms)],
            "production_ramp_days": [PRODUCTION_RAMP_DAYS_BOOM, PRODUCTION_RAMP_DAYS],
            "subsets_excluded_from_windows": sorted(SUBSET_PARENT),
            "scale": scale,
            "band_factor": BAND_FACTOR,
            "language_scope": "english_only, projected lifetime production",
        }
        if scale:
            for r in scored.values():
                mid = r["rel_pop_score"] * scale
                r["abs_estimate_mid"] = mid
                r["abs_low"] = mid / BAND_FACTOR
                r["abs_high"] = mid * BAND_FACTOR

    # --- pass 4b: reliability flags (red-team audit 2026-07-22) -------------
    # The grading-rate framework assumes booster-pack economics; giveaway /
    # promo / starter products have near-zero graded pops and their rel_pop is
    # a floor artifact (mcd16's "3,332 cards" was literally ~1 graded card /
    # era rate), so their absolutes are suppressed. Subset products double-
    # claim their parent's production. Recently released sets have not
    # accumulated graded pop yet (Prismatic Evolutions ranked 105th by pop vs
    # 10th by sales). base1's chase pop is dominated by THE hobby icon and
    # beta_p=0.5 cannot absorb that premium.
    from datetime import timedelta
    pop_snapshot_f = parse_date(load_json(CHASE_GRADED).get("generated_at")) \
        or date.today()
    NON_BOOSTER_PAT = ("mcdonald", "black star promo", "pop series", "trainer kit")
    NON_BOOSTER_IDS = {"xy0", "dv1", "ru1", "si1", "bp"}
    for sid, r in sets.items():
        name_l = (r["set_name"] or "").lower()
        psa_d = (r["mean_chase_psa"] or 0) * (r["pull_denominator"] or 1)
        if sid in SUBSET_PARENT:
            r["flags"].append("subset_set")
            r["subset_of"] = SUBSET_PARENT[sid]
        if (any(p in name_l for p in NON_BOOSTER_PAT) or sid in NON_BOOSTER_IDS
                or (r["n_chase_used"] > 0 and psa_d < 1000)):
            r["flags"].append("numerator_unreliable")
        rd = parse_date(r["release_date"])
        if rd and rd > pop_snapshot_f - timedelta(days=730):
            r["flags"].append("pop_lag_underestimate")
        if sid == "base1":
            r["flags"].append("icon_premium_suspect")
    for sid, r in scored.items():
        if "numerator_unreliable" in r["flags"] or "subset_set" in r["flags"]:
            r["abs_estimate_mid"] = None
            r["abs_low"] = None
            r["abs_high"] = None

    # --- pass 5: sales velocity rank (validation) ---------------------------
    sv_sets = [(sid, r["sales_per_month"]) for sid, r in scored.items()
               if r.get("sales_per_month") is not None]
    sv_sorted = sorted(sv_sets, key=lambda x: x[1], reverse=True)
    sv_rank = {sid: i + 1 for i, (sid, _) in enumerate(sv_sorted)}
    for sid, r in scored.items():
        r["sales_velocity_rank"] = sv_rank.get(sid)

    return sets, scored, calib, model_meta


# =============================================================================
# validation
# =============================================================================
def _spearman_pair(scored, value_key):
    pairs = [(r["rel_pop_score"], r[value_key]) for r in scored.values()
             if r.get(value_key) is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    return spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)


def validate_spearman(scored):
    """Compute Spearman of rel_pop_score vs each held-out signal."""
    rho_sales, n_sales = _spearman_pair(scored, "sales_per_month")
    rho_trends, n_trends = _spearman_pair(scored, "interest_rescaled")
    rho_pv, n_pv = _spearman_pair(scored, "pageviews_per_month")

    # Outliers: rel_pop_rank vs sales_rank disagreement (sales is the most populated).
    sids = [sid for sid, r in scored.items() if r.get("sales_per_month") is not None]
    rp = {sid: scored[sid]["rel_pop_score"] for sid in sids}
    sv = {sid: scored[sid]["sales_per_month"] for sid in sids}
    rp_rank = {sid: i + 1 for i, sid in enumerate(sorted(sids, key=lambda s: rp[s], reverse=True))}
    sv_rank = {sid: i + 1 for i, sid in enumerate(sorted(sids, key=lambda s: sv[s], reverse=True))}
    outliers = sorted(sids, key=lambda s: abs(rp_rank[s] - sv_rank[s]), reverse=True)[:10]
    out_rows = [{"set_id": s, "set_name": scored[s]["set_name"],
                 "rel_pop_rank": rp_rank[s], "sales_rank": sv_rank[s],
                 "rank_gap": rp_rank[s] - sv_rank[s]} for s in outliers]
    return {
        "spearman_relpop_vs_sales": rho_sales,
        "spearman_relpop_vs_trends": rho_trends,
        "spearman_relpop_vs_pageviews": rho_pv,
        "n_sales": n_sales, "n_trends": n_trends, "n_pageviews": n_pv,
        "spearman_note": ("rel_pop_vs_sales should be positive but <1 "
                          "(more printed -> more supply -> more sales, "
                          "but popularity/age/price break the tie). "
                          "rel_pop_vs_trends similar (popular sets get printed more)."),
        "biggest_rank_outliers_vs_sales": out_rows,
    }


def anchor_sensitivity(scored):
    """Compare abs_estimate_mid vs per-set total_print_run anchors.

    v3: anchors tagged cards_all_languages are converted to English at the
    set's release-date share before the ratio (estimates are English-only).
    Variant-subset anchors (1st_edition/shadowless) are kept in the table but
    tagged excluded_from_headline — the estimate covers ALL variants of the
    set, so the ratio is structurally inflated (same exclusion the fit uses)."""
    runs = load_json(KNOWN_RUNS)["anchors"]
    rows = []
    for a in runs:
        sid = a["set_id"]
        if a["estimate_type"] != "total_print_run" or a.get("value_mid") is None:
            continue
        rec = scored.get(sid)
        if rec is None or rec.get("abs_estimate_mid") is None:
            continue  # GLOBAL / *_ERA_AVG / WOTC_* pseudo-ids never match scored
        rd = parse_date(rec.get("release_date"))
        anchor_en, converted = anchor_value_english(a, rd)
        est = rec["abs_estimate_mid"]
        rows.append({
            "set_id": sid,
            "set_name": a["set_name"],
            "variant": a.get("print_variant"),
            "credibility": a.get("source_credibility"),
            "anchor_mid_raw": a["value_mid"],
            "anchor_mid_english": anchor_en,
            "unit_converted_from_all_languages": converted,
            "estimate_mid": est,
            "ratio_est_over_anchor": est / anchor_en,
            "excluded_from_headline": a.get("print_variant") in ("1st_edition", "shadowless"),
        })
    return rows


# =============================================================================
# output
# =============================================================================
def fmt_int(x):
    return f"{int(round(x)):,}" if x is not None else "n/a"


def write_json(sets, scored, calib, validation, sens, grading_model_meta, args):
    out_sets = {}
    for sid, r in sets.items():
        rec = {
            "set_name": r["set_name"],
            "era": r["era"],
            "release_date": r["release_date"],
            "n_chase_used": r["n_chase_used"],
            "mean_chase_psa": round(r["mean_chase_psa"], 2) if r["mean_chase_psa"] is not None else None,
            "chase_tier": r["chase_tier"],
            "pull_denominator": r["pull_denominator"],
            "chase_value": r.get("chase_value"),
            "predicted_grading_rate": r.get("predicted_grading_rate"),
            "rel_pop_score": round(r["rel_pop_score"], 4) if r.get("rel_pop_score") is not None else None,
            "rel_pop_norm100": r.get("rel_pop_norm100"),
            "abs_estimate_mid": round(r["abs_estimate_mid"]) if r.get("abs_estimate_mid") is not None else None,
            "abs_low": round(r["abs_low"]) if r.get("abs_low") is not None else None,
            "abs_high": round(r["abs_high"]) if r.get("abs_high") is not None else None,
            "sales_velocity_rank": r.get("sales_velocity_rank"),
            # validation signals (no longer divisor inputs)
            "interest_rescaled": r.get("interest_rescaled"),
            "trend_z": r.get("trend_z"),
            "pageviews_per_month": r.get("pageviews_per_month"),
            "sales_per_month": r.get("sales_per_month"),
            "flags": r["flags"],
            "subset_of": r.get("subset_of"),
        }
        out_sets[sid] = rec

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "language_scope": "english_only",
        "english_share": share_doc(),
        "notes": (
            "v3 (2026-07-21): ALL absolutes are ENGLISH-ONLY projected lifetime "
            "production. rel_pop = mean_chase_psa * pull_denominator / "
            "predicted_grading_rate; grading rate from data/grading_rate_model.json "
            "fit against a 16-rung dated TPC checkpoint ladder (12 official rungs, "
            "converted global->English via a documented english_share layer: 0.40 "
            "pre-2020 / 0.35 after, +/-0.10), unit-corrected community per-set "
            "anchors (all-languages WAGs halved-ish at release-date share), and an "
            "independent SEC-revenue-derived WOTC 1999-2001 English window. "
            "Checkpoint windows use a production ramp (12mo boom-era / 24mo "
            "modern). Bands +/-{}x. Within-era RELATIVES remain the strongest "
            "output; absolutes are now unit-consistent but inherit the english_share "
            "assumption linearly (see results.md sensitivity).".format(BAND_FACTOR)
        ),
        "grading_rate_model": {
            "equation": grading_model_meta.get("equation"),
            "beta_p": grading_model_meta.get("beta_p"),
            "beta_y": grading_model_meta.get("beta_y"),
            "alpha_by_era": grading_model_meta.get("alpha_by_era"),
            "anchor_exclusions": grading_model_meta.get("anchor_exclusions"),
            "fit_weights": grading_model_meta.get("fit_weights"),
        },
        "constants": {
            "confidence_keep": sorted(CONFIDENCE_KEEP),
            "band_factor": BAND_FACTOR,
            "production_ramp_days": {"boom_era_pre2003": PRODUCTION_RAMP_DAYS_BOOM,
                                     "default": PRODUCTION_RAMP_DAYS},
            "era_buckets": [[n, d.isoformat()] for n, d in ERA_BUCKETS],
            "pull_denominator_table": PULL_DENOM,
            "pull_denominator_units": "packs-opened-per-graded-chase-copy, relative, WOTC tier-3 holo = 1.0 (ESTIMATES)",
        },
        "calibration": calib,
        "validation": validation,
        "anchor_sensitivity": sens,
        "stats": {
            "n_sets": len(sets),
            "n_scored": len(scored),
            "n_unscored": len(sets) - len(scored),
        },
        "sets": out_sets,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def write_results_md(doc, scored, calib, validation, sens):
    ranked = sorted(
        [(sid, r) for sid, r in doc["sets"].items() if r["rel_pop_score"] is not None],
        key=lambda x: x[1]["rel_pop_score"], reverse=True)

    L = []
    L.append("# Results: Set Print-Population Estimates\n")
    L.append(f"_Generated {doc['generated_at']} — model {doc['model_version']}._\n")
    L.append(
        "> **Model v3 (english-only + dated checkpoint ladder).** ALL absolute "
        "numbers are ENGLISH-ONLY projected lifetime production. rel_pop = "
        "mean_chase_psa × pull_D / predicted_grading_rate; the grading-rate fit "
        "targets a dated TPC cumulative-checkpoint ladder (6 rungs "
        "archive-verified, the current 85B live-page, the rest official-claim or "
        "transcription-tier and down-weighted accordingly) converted "
        "global→English via a documented `english_share` layer applied to "
        "per-regime INCREMENTS (0.40 pre-2020 / 0.35 after, ±0.10), "
        "unit-corrected community anchors (every per-set anchor is an "
        "all-languages community guess — provenance traced 2026-07-21), and an "
        "SEC-revenue-derived WOTC 1999–2001 English window (a consistency check "
        "that shares the assumption layer — see caveats). Checkpoint windows "
        "use a production ramp (12 mo boom-era / 24 mo after); subset products "
        "are excluded from windows. Trends / pageviews / sales velocity remain "
        "held-out VALIDATION signals (pageviews has n=1 usable pair and is not "
        "reported). See `docs/anchor_research_2026-07-21.md` for the evidence "
        "base.\n")
    st = doc["stats"]
    L.append(f"- Sets scored: **{st['n_scored']}** / {st['n_sets']} "
             f"({st['n_unscored']} unscored, no confident chase pop or grading rate).")
    if calib and calib.get("scale"):
        L.append(f"- Absolute calibration: credibility-weighted geometric-mean scale "
                 f"over **{calib['n_rungs_used']} usable ladder rungs** (latest: "
                 f"{fmt_int(calib['latest_rung_value_global'])} global @ "
                 f"{calib['latest_rung_date']} → "
                 f"{fmt_int(calib['latest_rung_value_english'])} English, "
                 f"increment-correct share). Rungs after (pop snapshot − "
                 f"{calib['roster_lag_days']} d) are excluded: the scored roster "
                 f"lacks 2025-26 sets and recent sets' graded pops lag. "
                 f"scale={calib['scale']:.3f}; no single rung is pinned exactly — "
                 "per-rung residuals are in the ladder table below.")

    # Grading-rate model summary
    gm = doc.get("grading_rate_model", {})
    if gm:
        L.append("\n## Grading-rate model (v3 divisor)\n")
        L.append(f"`{gm.get('equation')}`")
        L.append(f"\n- `beta_p = {gm.get('beta_p')}` (pinned prior — grading rate "
                 "rises with chase value; sqrt scaling).")
        L.append(f"- `beta_y = {gm.get('beta_y')}` (pinned; age absorbed into era intercept).")
        L.append("- Per-era intercept `alpha[era]` (log grading rate at chase_value=$100):\n")
        L.append("| era | alpha | baseline rate | median predicted rate (across sets) |")
        L.append("|-----|------:|--------------:|-----------------------------:|")
        # Pull per-era median from grading_rate_model.json
        try:
            gmodel = load_json(GRADING_MODEL)
            per_era = gmodel.get("per_era", {})
        except Exception:
            per_era = {}
        for era in ("WOTC","ECARD","EX","DP","HGSS","BW","XY","SM","SWSH","SV"):
            alpha = gm.get("alpha_by_era", {}).get(era)
            row = per_era.get(era, {})
            if alpha is None:
                continue
            base = math.exp(alpha)
            med = row.get("median_predicted_grading_rate")
            med_s = f"{med:.2e}" if med is not None else "—"
            L.append(f"| {era} | {alpha:.3f} | {base:.2e} | {med_s} |")
        L.append("")

    def table(rows, title):
        L.append(f"## {title}\n")
        L.append("| # | set_id | name | era | n_chase | mean PSA | pull_D | chase $ | grading rate | rel_pop(norm100) | abs_mid | abs_low–high | flags |")
        L.append("|---|--------|------|-----|--------:|---------:|------:|--------:|-------------:|-----------------:|--------:|-------------|-------|")
        for i, (sid, r) in enumerate(rows, 1):
            ab = (f"{fmt_int(r['abs_low'])}–{fmt_int(r['abs_high'])}"
                  if r.get("abs_low") is not None else "n/a")
            gr = r.get("predicted_grading_rate")
            gr_s = f"{gr:.2e}" if gr is not None else "—"
            cv = r.get("chase_value")
            cv_s = f"${cv:.0f}" if cv is not None else "—"
            L.append("| {i} | {sid} | {nm} | {era} | {n} | {psa} | {pd} | {cv} | {gr} | {rn} | {am} | {ab} | {fl} |".format(
                i=i, sid=sid, nm=r["set_name"], era=r["era"], n=r["n_chase_used"],
                psa=fmt_int(r["mean_chase_psa"]), pd=r["pull_denominator"],
                cv=cv_s, gr=gr_s,
                rn=r.get("rel_pop_norm100"),
                am=fmt_int(r.get("abs_estimate_mid")), ab=ab,
                fl=",".join(r["flags"]) or "—"))
        L.append("")

    table(ranked[:20], "Top 20 sets by estimated print population")
    table(list(reversed(ranked[-20:])), "Bottom 20 sets by estimated print population")

    # sensitivity table
    L.append("## Anchor sensitivity (estimate / published anchor, ENGLISH units)\n")
    L.append("Per-set `total_print_run` anchors from `known_print_runs.json`, converted "
             "to English cards at the set's release-date share when tagged "
             "`cards_all_languages` (all of them are — the 2026-07-21 provenance audit "
             "traced every one to all-languages community guesses). The grading-rate "
             "model is fit *jointly* against these (weight 0.5 for hobbyist-guess) and "
             "the checkpoint ladder, so this is a fit-quality measure, not held-out. "
             "Variant-subset rows (1st Ed/Shadowless) are excluded from the headline — "
             "the estimate covers all variants of a set.\n")
    if sens:
        head = [s for s in sens if not s.get("excluded_from_headline")]
        within2 = sum(1 for s in head if 0.5 <= s["ratio_est_over_anchor"] <= 2.0)
        within3 = sum(1 for s in head if (1 / 3.0) <= s["ratio_est_over_anchor"] <= 3.0)
        L.append(f"**{within2}/{len(head)} anchors within 2×, {within3}/{len(head)} within 3×** "
                 f"(v2 counted 8/18 within 2× against UN-converted all-language anchors — "
                 "not comparable; v1 had 1/10).\n")
        L.append("| set | variant | credibility | anchor raw | anchor EN | estimate_mid | est/anchor | headline |")
        L.append("|-----|---------|-------------|-----------:|----------:|-------------:|-----------:|----------|")
        for s in sorted(sens, key=lambda x: x["ratio_est_over_anchor"], reverse=True):
            L.append("| {sid} ({nm}) | {v} | {c} | {ar} | {a} | {e} | {r:.2f}× | {h} |".format(
                sid=s["set_id"], nm=s["set_name"], v=s["variant"], c=s["credibility"],
                ar=fmt_int(s["anchor_mid_raw"]), a=fmt_int(s["anchor_mid_english"]),
                e=fmt_int(s["estimate_mid"]),
                r=s["ratio_est_over_anchor"],
                h="excluded" if s.get("excluded_from_headline") else "yes"))
    else:
        L.append("_No per-set anchors matched scored sets._")
    L.append("")

    # checkpoint ladder + revenue window (from grading_rate_model.json)
    try:
        gmodel = load_json(GRADING_MODEL)
        L.append("## Checkpoint-ladder fit (English targets)\n")
        L.append("| date | global | share | EN target | predicted | ratio | credibility |")
        L.append("|------|-------:|------:|----------:|----------:|------:|-------------|")
        for r in gmodel.get("tpc_fit", []):
            L.append(f"| {r['checkpoint_date']} | {r['checkpoint_value_global']/1e9:.1f}B "
                     f"| {r['english_share']:.2f} | {r['checkpoint_value_english']/1e9:.1f}B "
                     f"| {r['predicted_windowed_sum']/1e9:.1f}B "
                     f"| {r['ratio_pred_over_cp']:.2f} | {r['credibility']} |")
        for w in gmodel.get("english_window_fit", []):
            L.append(f"\nRevenue-derived window **{w['anchor_id']}** "
                     f"({w['window'][0]}→{w['window'][1]}): predicted "
                     f"**{w['predicted_windowed_sum']/1e9:.1f}B** vs target "
                     f"{w['target_english_mid']/1e9:.1f}B "
                     f"[{(w['target_english_low'] or 0)/1e9:.1f}–"
                     f"{(w['target_english_high'] or 0)/1e9:.1f}] — "
                     f"**{'WITHIN' if w['within_band'] else 'OUTSIDE'} band** "
                     f"(ratio {w['ratio_pred_over_target']:.2f}). Honesty note: "
                     "this is a CONSISTENCY CHECK, not independent corroboration — "
                     "the window is a term in the same joint fit, its prediction is "
                     "the same quantity the end-2001 rung constrains, and the two "
                     "target derivations (TPC totals × share vs SEC dollars ÷ "
                     "wholesale price × EN-of-West share) use disjoint primary "
                     "documents but SHARE the community-assumption share layer, "
                     "whose 0.40 value was itself chosen partly for this "
                     "agreement.")
        L.append("")
    except Exception:
        pass

    # Per-era summary
    try:
        gmodel = load_json(GRADING_MODEL)
        per_era = gmodel.get("per_era", {})
    except Exception:
        per_era = {}
    if per_era:
        L.append("## Per-era summary\n")
        L.append("| era | n sets | median rate | median print run | sum print run |")
        L.append("|-----|------:|-----------:|----------------:|--------------:|")
        for era in ("WOTC","ECARD","EX","DP","HGSS","BW","XY","SM","SWSH","SV"):
            row = per_era.get(era, {})
            n = row.get("n", 0)
            if n == 0: continue
            mr = row.get("median_predicted_grading_rate")
            mp = row.get("median_predicted_print_run")
            sp = row.get("sum_predicted_print_run")
            L.append(f"| {era} | {n} | {mr:.2e} | {fmt_int(mp)} | {fmt_int(sp)} |")
        L.append("")

    # Validation: Spearman against multiple signals
    L.append("## Validation: rel_pop vs held-out signals (Spearman)\n")
    rho_s = validation.get("spearman_relpop_vs_sales")
    rho_t = validation.get("spearman_relpop_vs_trends")
    rho_p = validation.get("spearman_relpop_vs_pageviews")
    L.append("| signal | ρ | n | interpretation |")
    L.append("|--------|--:|--:|----------------|")
    if rho_s is not None:
        L.append(f"| sales velocity (TCGPlayer) | {rho_s:.3f} | {validation.get('n_sales')} | "
                 "endogenous: bigger print -> more supply -> more sales (expect +) |")
    if rho_t is not None:
        L.append(f"| Google Trends interest | {rho_t:.3f} | {validation.get('n_trends')} | "
                 "demand-side: popular sets get printed more (expect +, weaker than sales) |")
    if rho_p is not None:
        L.append(f"| Wikipedia pageviews | {rho_p:.3f} | {validation.get('n_pageviews')} | "
                 "mostly shared-mascot articles; expect weak/noisy |")
    L.append("\nExpectation: **positive but < 1** — more printed ⇒ more supply/demand traffic, "
             "but popularity, age and price break the tie. A value near 0 or negative would "
             "flag a bug; ~1.0 would mean we just re-derived demand.\n")
    outliers = validation.get("biggest_rank_outliers_vs_sales", [])
    if outliers:
        L.append("Biggest rank disagreements vs sales (plausibly real, not bugs):\n")
        L.append("| set_id | name | rel_pop_rank | sales_rank | gap |")
        L.append("|--------|------|-------------:|-----------:|----:|")
        for o in outliers:
            L.append(f"| {o['set_id']} | {o['set_name']} | {o['rel_pop_rank']} | "
                     f"{o['sales_rank']} | {o['rank_gap']:+d} |")
        L.append("\n_Positive gap = ranks much higher in population than in sales (printed big "
                 "but trades slowly — old/cheap bulk). Negative = trades hot for its print size "
                 "(small but in demand)._")
    L.append("")

    # Confidence
    L.append("## Confidence & caveats (frank)\n")
    L.append(
        "### What changed in v3 (2026-07-21/22 deep-research + adversarial audit)\n"
        "1. **Units fixed.** v2 compared English graded pops against ALL-LANGUAGES anchors "
        "and global checkpoints — an implicit english_share of 1.0, smearing Japanese-only "
        "volume across English sets. v3 outputs are English-only: the share (0.40 pre-2020 "
        "/ 0.35 after, ±0.10) is applied to per-regime production INCREMENTS (applying it "
        "to cumulative totals made the EN ladder non-monotonic — caught in audit), and "
        "every per-set community anchor was provenance-traced and converted.\n"
        "2. **5 checkpoints → dated ladder.** Rungs by evidence tier: 6 archive-verified "
        "(21.5B@2015, 25.7B@2018, 27.2B@2019, 30.4B@2020, 34.1B@2021, 64.8B@2024 — the "
        "last correcting v2's 64.9B), the 85B@2026 live page, other official-claim rungs, "
        "and transcription-tier rungs (13B@2005 via forum-relayed press releases, 12B@2001, "
        "14B@2006, 20B@2013, 43.2B@2022) at half weight. Rungs newer than the scored "
        "roster supports are EXCLUDED from the fit (currently capped at Mar-2024) so "
        "recent unscored production cannot be redistributed onto older sets.\n"
        "3. **SEC-revenue window.** Evidence tiers, precisely: $568M/2000 is derived from "
        "an audited-period Hasbro 10-K405 disclosure (the 15% sentence itself sits outside "
        "the auditor's opinion); ~$500M/1999 is an ICv2 trade-press derivation across "
        "filings (well-sourced-estimate); 2001 has only an official ≤$286M ceiling with "
        "~$100M inferred. ÷ wholesale pack price (40–55% of the $3.29 MSRP, WOTC's own "
        "archived store) × 11 cards/pack × 0.65 EN-of-West → 4.2–7.4B EN window, fit "
        "jointly (a consistency check — see the ladder section's honesty note).\n"
        "4. **Production ramp + subset exclusion.** Windows ramp over 12 mo (boom era, "
        "motivated by the documented 2001 overproduction writeoffs AND tuned partly to "
        "reduce the Mar-2005 rung overshoot — both true) / 24 mo (modern); subset "
        "products (Trainer Galleries, cel25c) no longer double-claim parent production. "
        "Absolutes are projected LIFETIME production.\n"
        "5. **Reliability flags** (`numerator_unreliable`, `pop_lag_underestimate`, "
        "`subset_set`, `icon_premium_suspect`) mark rows where the framework's "
        "assumptions fail; absolutes are SUPPRESSED for non-booster/subset rows rather "
        "than published as floor artifacts.\n")
    L.append(
        "### Why we PIN beta_p instead of fitting it\n"
        "All 17 per-set anchors are hobbyist guesses (weight 0.5). Re-tested "
        "2026-07-22 with the expanded 6-era anchor set: a free fit gives "
        "beta_p≈0.04 (no longer sign-flipped as with the v2 anchor set, but a "
        "slope learned from guess-tier data; ≈0 would claim chase value doesn't "
        "drive grading propensity within an era, contradicting observable PSA "
        "submission behavior). We keep the 0.5 physical prior and only fit "
        "per-era intercepts. The run is deterministic.\n")
    tension = ""
    try:
        gmodel_t = load_json(GRADING_MODEL)
        ratios = [(r["checkpoint_date"], r["ratio_pred_over_cp"])
                  for r in gmodel_t.get("tpc_fit", [])]
        if ratios:
            worst = max(ratios, key=lambda x: abs(math.log(x[1])))
            tension = (f"Worst rung: {worst[1]:.2f}× @ {worst[0]}; full residual "
                       "profile in the ladder table above.")
    except Exception:
        pass
    L.append(
        "### Known residual tensions (documented, not hidden)\n"
        "- **The ladder has a residual SLOPE misfit**: the fit over-predicts "
        "crash/mid-era rungs and under-predicts the steep post-2021 English "
        "increments (FY23 alone implies ~4B EN of new production at share 0.35). "
        "Under the geomean calibration the residuals are spread rather than "
        "hidden in a pinned rung. " + tension + " Unresolved candidate causes: "
        "english_share >0.35 post-2020, modern reprint tails longer than 24 mo, "
        "transcription-tier crash-era rungs being loose, or genuinely larger "
        "modern per-set runs than community English estimates (~1B/set).\n"
        "- **base1 is flagged `icon_premium_suspect`** (est ≈2.5× its halved WAG "
        "anchor). The Charizard-icon grading premium exceeds what beta_p=0.5 "
        "corrects; base1's within-era relative is likely inflated ~2×.\n"
        "- **The 12B end-2001 rung and the WOTC per-set WAGs are NOT independent** "
        "— the WAGs were constructed to sum to that checkpoint, so jointly "
        "fitting both partly double-counts one source (both are low-weight).\n"
        "- **All cumulative figures are 'over X' floors** used here as point "
        "targets; true values sit above each rung by an unknown margin.\n"
        "- **BW-era estimates are very low** (median ~7M EN/set), violating the "
        "single-poster ordinal 'BW > HGSS' — but matching the sealed-box market "
        "(BW boxes price above many WOTC boxes). We report, not force, the "
        "ordinal.\n"
        "- **Era bucketing is date-mechanical**: bw11 Legendary Treasures lands "
        "in era XY, swshp in era SM. Cosmetic for flagged rows, but visible in "
        "tables.\n")
    L.append(
        "### What's still order-of-magnitude\n"
        "- **english_share is THE load-bearing assumption** — no official language split "
        "exists; absolutes scale linearly in it (±0.10 ⇒ ±25–29%). Evidence: Japan market "
        "≈ US market; TPCi/TPC production split; language count 11→16.\n"
        "- **Pull-rate `D` table** unchanged — still the second-biggest lever, still "
        "estimates.\n"
        "- **Unmodelled biases:** WOTC 1st-Ed/Shadowless/Unlimited pop-merging; attrition; "
        "crack-and-resubmit inflation; precon dilution (~30% of EX-era prints were theme "
        "decks per community estimate); grading-rate drift.\n"
        "- **Bands ±3×** are order-of-magnitude rails, not confidence intervals.\n")
    L.append(
        "### Honest read\n"
        "Within-era relatives among UNFLAGGED mainline booster sets remain the "
        "strongest output (flagged rows — subsets, promos, icon-premium, "
        "pop-lagged — are exactly where relatives break too). Absolutes are now "
        "unit-consistent and ladder-dense: for mid-band mainline sets of "
        "~2003–2022 the ±3× band is a fair claim; for WOTC (base1 especially), "
        "2023+ mega-sets, and anything flagged, treat the numbers as directional "
        "only. Everything scales linearly in english_share. The EX–XY dead zone "
        "is now constrained by real checkpoints instead of smoothness alone; no "
        "NEW conclusive per-set evidence for those eras was found by the "
        "2026-07-21 research pass — the existing community per-set numbers "
        "remain unsourced guesses (and two of them, ex7 and xy12, are used here "
        "at guess weight).\n")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L))


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-absolute", action="store_true",
                    help="Skip absolute calibration (emit relatives only).")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout summary.")
    args = ap.parse_args()

    sets, scored, calib, grading_model_meta = build_estimates(args)
    validation = validate_spearman(scored)
    sens = anchor_sensitivity(scored) if not args.no_absolute else []

    doc = write_json(sets, scored, calib, validation, sens, grading_model_meta, args)
    write_results_md(doc, scored, calib, validation, sens)

    if not args.quiet:
        ranked = sorted([(sid, r) for sid, r in doc["sets"].items()
                         if r["rel_pop_score"] is not None],
                        key=lambda x: x[1]["rel_pop_score"], reverse=True)
        print(f"Scored {len(scored)}/{len(sets)} sets (model {MODEL_VERSION}).")
        if calib and calib.get("scale"):
            print(f"Calibrated: geomean over {calib['n_rungs_used']} rungs, latest "
                  f"{fmt_int(calib['latest_rung_value_global'])} global @ "
                  f"{calib['latest_rung_date']} -> "
                  f"{fmt_int(calib['latest_rung_value_english'])} EN "
                  f"(scale={calib['scale']:.4f}).")
        print(f"Spearman rel_pop vs sales:   {validation['spearman_relpop_vs_sales']}")
        print(f"Spearman rel_pop vs trends:  {validation['spearman_relpop_vs_trends']}")
        print(f"Spearman rel_pop vs pviews:  {validation['spearman_relpop_vs_pageviews']}")
        if sens:
            head = [s for s in sens if not s.get('excluded_from_headline')]
            w2 = sum(1 for s in head if 0.5 <= s['ratio_est_over_anchor'] <= 2.0)
            w3 = sum(1 for s in head if (1/3.0) <= s['ratio_est_over_anchor'] <= 3.0)
            print(f"Anchors within 2x: {w2}/{len(head)}, within 3x: {w3}/{len(head)} "
                  f"({len(sens)-len(head)} variant-subset rows excluded)")
        print("Top 10:")
        for sid, r in ranked[:10]:
            print(f"  {sid:10s} {r['set_name'][:28]:28s} rel100={r.get('rel_pop_norm100'):>7} "
                  f"abs_mid={fmt_int(r.get('abs_estimate_mid'))}")
        print("Bottom 10:")
        for sid, r in ranked[-10:]:
            print(f"  {sid:10s} {r['set_name'][:28]:28s} rel100={r.get('rel_pop_norm100'):>7} "
                  f"abs_mid={fmt_int(r.get('abs_estimate_mid'))}")
        print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
