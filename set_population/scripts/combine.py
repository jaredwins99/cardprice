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

Absolute calibration pins the windowed sum of rel_pop to an official TPC
cumulative checkpoint (the same anchor the grading-rate model already fits to,
so the post-fit scale should be ~1.0).

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

MODEL_VERSION = "v2-grading-rate"

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

# --- 3. Grading-rate divisor (model v2) --------------------------------------
# Replaces the v1 popularity divisor. The mechanistic grading-rate model lives
# in scripts/fit_grading_rate.py and writes data/grading_rate_model.json. This
# combiner reads the per-set predicted_grading_rate from that file.
#
# Trends, pageviews and sales velocity are now VALIDATION signals only (each
# gets a Spearman cross-check vs rel_pop). None of them feed into rel_pop.
W_TRENDS_VALIDATION = 1.0    # only used for the validation Spearman, NOT divisor
W_PAGEVIEWS_VALIDATION = 0.0 # near-useless (mostly shared mascot articles); kept @0
# sales velocity is endogenous (population -> supply -> sales); validation only.

# --- 5. Absolute calibration -------------------------------------------------
# Official TPC "Pokemon in Figures" cumulative checkpoints (value, date).
# From known_print_runs.json GLOBAL anchors + prior_art.md. Used to scale the
# windowed sum of rel_pop. value in cards.
TPC_CHECKPOINTS = [
    (23_600_000_000, date(2017, 3, 31)),
    (43_200_000_000, date(2022, 3, 31)),
    (52_900_000_000, date(2023, 3, 31)),
    (64_900_000_000, date(2024, 3, 31)),
    (75_000_000_000, date(2025, 3, 31)),
]
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
            "mean_chase_psa": round(mean_psa, 2) if mean_psa is not None else None,
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

    # --- pass 3: relative population (NEW v2: divide by predicted grading rate) -
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

    # --- pass 4: absolute calibration ---------------------------------------
    calib = None
    if not args.no_absolute and scored:
        today = date.today()
        usable = [(v, d) for v, d in TPC_CHECKPOINTS if d <= today]
        cp_value, cp_date = max(usable, key=lambda x: x[1]) if usable else max(
            TPC_CHECKPOINTS, key=lambda x: x[1])
        # window: sets released before checkpoint date with a rel_pop
        windowed = [r for r in scored.values()
                    if r["release_date"] and parse_date(r["release_date"]) <= cp_date]
        wsum = sum(r["rel_pop_score"] for r in windowed)
        scale = cp_value / wsum if wsum else None
        calib = {
            "checkpoint_value": cp_value,
            "checkpoint_date": cp_date.isoformat(),
            "n_sets_in_window": len(windowed),
            "windowed_rel_pop_sum": wsum,
            "scale": scale,
            "band_factor": BAND_FACTOR,
        }
        if scale:
            for r in scored.values():
                mid = r["rel_pop_score"] * scale
                r["abs_estimate_mid"] = mid
                r["abs_low"] = mid / BAND_FACTOR
                r["abs_high"] = mid * BAND_FACTOR

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
    """Compare abs_estimate_mid vs per-set total_print_run anchors."""
    runs = load_json(KNOWN_RUNS)["anchors"]
    rows = []
    for a in runs:
        sid = a["set_id"]
        if sid in ("GLOBAL", "MODERN_AVG") or a["estimate_type"] != "total_print_run":
            continue
        if a.get("value_mid") is None:
            continue
        est = scored.get(sid, {}).get("abs_estimate_mid")
        if est is None:
            continue
        ratio = est / a["value_mid"]
        rows.append({
            "set_id": sid,
            "set_name": a["set_name"],
            "variant": a.get("print_variant"),
            "credibility": a.get("source_credibility"),
            "anchor_mid": a["value_mid"],
            "estimate_mid": est,
            "ratio_est_over_anchor": ratio,
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
            "mean_chase_psa": r["mean_chase_psa"],
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
        }
        out_sets[sid] = rec

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "notes": (
            "rel_pop = mean_chase_psa * pull_denominator / predicted_grading_rate. "
            "Grading rate from mechanistic model (data/grading_rate_model.json): "
            "log(rate) = alpha[era] + beta_p * log(chase_value/$100) + beta_y * log(years/10y). "
            "Replaces v1 Google-Trends popularity divisor. Trends/pageviews/sales-velocity "
            "are held out for validation (Spearman cross-checks). "
            "Absolutes pinned to TPC cumulative checkpoint (windowed sum); bands +/-{}x. "
            "Defensible on RELATIVES, order-of-magnitude on ABSOLUTES "
            "(see results.md anchor-sensitivity).".format(BAND_FACTOR)
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
        "> **Model v2 (grading-rate divisor).** rel_pop = mean_chase_psa × pull_D / "
        "predicted_grading_rate. Grading rate is a mechanistic function of era + chase "
        "value (see `methodology.md` and `data/grading_rate_model.json`). Trends / "
        "pageviews / sales velocity are now VALIDATION signals, not divisor inputs.\n")
    st = doc["stats"]
    L.append(f"- Sets scored: **{st['n_scored']}** / {st['n_sets']} "
             f"({st['n_unscored']} unscored, no confident chase pop or grading rate).")
    if calib and calib.get("scale"):
        L.append(f"- Absolute calibration anchor: TPC **{fmt_int(calib['checkpoint_value'])}** "
                 f"cards cumulative @ {calib['checkpoint_date']} "
                 f"(windowed over {calib['n_sets_in_window']} pre-checkpoint sets; "
                 f"post-fit scale={calib['scale']:.3f}).")

    # Grading-rate model summary
    gm = doc.get("grading_rate_model", {})
    if gm:
        L.append("\n## Grading-rate model (v2 divisor)\n")
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
    L.append("## Anchor sensitivity (estimate / published anchor)\n")
    L.append("Per-set `total_print_run` anchors from `known_print_runs.json`. "
             "The grading-rate model is fit *jointly* against these (with low weight for "
             "`hobbyist-guess` credibility) and the official TPC cumulative checkpoints. "
             "This sensitivity is a fit-quality measure, not held-out.\n")
    if sens:
        within2 = sum(1 for s in sens if 0.5 <= s["ratio_est_over_anchor"] <= 2.0)
        within3 = sum(1 for s in sens if (1 / 3.0) <= s["ratio_est_over_anchor"] <= 3.0)
        L.append(f"**{within2}/{len(sens)} anchors within 2×, {within3}/{len(sens)} within 3× "
                 "(model v1 had 1/10 within 2×, 1/10 within 3×).**\n")
        L.append("| set | variant | credibility | anchor_mid | estimate_mid | est/anchor |")
        L.append("|-----|---------|-------------|-----------:|-------------:|-----------:|")
        for s in sorted(sens, key=lambda x: x["ratio_est_over_anchor"], reverse=True):
            L.append("| {sid} ({nm}) | {v} | {c} | {a} | {e} | {r:.2f}× |".format(
                sid=s["set_id"], nm=s["set_name"], v=s["variant"], c=s["credibility"],
                a=fmt_int(s["anchor_mid"]), e=fmt_int(s["estimate_mid"]),
                r=s["ratio_est_over_anchor"]))
    else:
        L.append("_No per-set anchors matched scored sets._")
    L.append("")

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
        "### What changed in v2\n"
        "v1 used `exp(z(log(Google Trends)))` as the divisor. That captures search-popularity "
        "but is *blind* to grading-rate dynamics: a $3,000 vintage card gets graded ~10–20× "
        "harder than a $100 modern SIR, regardless of who's searching. v1 anchors landed "
        "1/10 within 2× (vintage anchors ~30–1000× off). v2 swaps in a mechanistic "
        "`grading_rate = exp(alpha[era] + 0.5·log(chase_value/$100))` fit jointly to "
        "per-set anchors AND the official TPC cumulative checkpoints.\n")
    L.append(
        "### Why we PIN beta_p instead of fitting it\n"
        "Of the 15 anchors in `known_print_runs.json`, only 8 are usable per-set print-run "
        "estimates after excluding variant subsets and the `neo3` cumulative checkpoint. "
        "**Seven of those 8 are WOTC and one is EX.** Fitting log_price/log_yrs slopes on that "
        "distribution either over-fits or returns negative coefficients (more value → less "
        "grading), which is physically wrong. We pin `beta_p=0.5` (square-root scaling, "
        "consistent with PSA's value-driven grading-rate dynamics) and only fit per-era "
        "intercepts. **The model is reproducible from the JSON; the scripts/fit_grading_rate.py "
        "run is deterministic.**\n")
    L.append(
        "### What's still order-of-magnitude\n"
        "- **Single-era anchors drive their era's intercept.** WOTC has 7 anchors → WOTC alpha "
        "is data-driven. EX has 1 → EX alpha is essentially that one anchor. ECARD, DP, HGSS, "
        "BW, XY, SM, SWSH, SV have ZERO per-set anchors; their alphas are determined by TPC "
        "checkpoint windowed sums + cross-era smoothness. Per-set absolutes in those eras are "
        "order-of-magnitude.\n"
        "- **Pull-rate `D` table unchanged from v1** — still the second-biggest lever and still "
        "all estimates. See `methodology.md` for the table.\n"
        "- **Unmodelled biases unchanged:** WOTC 1st-Ed/Shadowless/Unlimited graded separately "
        "but pop-merged; JP vs EN separate prints; attrition; crack-and-resubmit pop inflation; "
        "precon-deck dilution; grading-rate drift over time.\n"
        "- **Bands ±3×** are honest order-of-magnitude rails, not confidence intervals.\n")
    L.append(
        "### Honest read\n"
        "Within-era relatives remain the strongest output (the numerator and pull_D are "
        "unchanged; the divisor change mostly affects cross-era). Cross-era relatives and "
        "all absolutes are still order-of-magnitude — better than v1 but not tight. To tighten "
        "the absolutes meaningfully we need more per-set anchors, especially in DP/HGSS/BW/XY/"
        "SM/SWSH/SV (currently zero per-set anchors in those eras).\n")

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
        print(f"Scored {len(scored)}/{len(sets)} sets (model v2: grading-rate divisor).")
        if calib and calib.get("scale"):
            print(f"Calibrated to TPC {fmt_int(calib['checkpoint_value'])} @ "
                  f"{calib['checkpoint_date']} over {calib['n_sets_in_window']} sets "
                  f"(scale={calib['scale']:.4f}).")
        print(f"Spearman rel_pop vs sales:   {validation['spearman_relpop_vs_sales']}")
        print(f"Spearman rel_pop vs trends:  {validation['spearman_relpop_vs_trends']}")
        print(f"Spearman rel_pop vs pviews:  {validation['spearman_relpop_vs_pageviews']}")
        if sens:
            w2 = sum(1 for s in sens if 0.5 <= s['ratio_est_over_anchor'] <= 2.0)
            w3 = sum(1 for s in sens if (1/3.0) <= s['ratio_est_over_anchor'] <= 3.0)
            print(f"Anchors within 2x: {w2}/{len(sens)}, within 3x: {w3}/{len(sens)}")
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
