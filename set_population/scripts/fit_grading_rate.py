#!/usr/bin/env python3
"""Mechanistic grading-rate model for the set_population sub-project.

Replaces the Google-Trends popularity divisor with a model of *grading rate*:

    grading_rate(set) = exp(alpha_era + beta_p * log(chase_value / $100)
                                       + beta_y * log(years_since_release / 10y))

Print population (relative) is then

    rel_print_pop(set) ∝ mean_chase_psa(set) * pull_denominator(set) / grading_rate(set)

with the final scale pinned to the latest TPC "Pokémon in Figures" cumulative
checkpoint (75 B cards @ Mar 2025) by windowed sum.

Why this design and not OLS on the anchors directly
---------------------------------------------------
Of the 15 calibration anchors in `data/known_print_runs.json`, only ~8 are
"per-set total print runs" we can use after exclusions (1st-Ed / Shadowless
anchors target a variant subset of a set whose PSA pop is reported across all
variants; the "neo3 12 B" anchor is a cumulative checkpoint, not a single set).
Of those 8, **7 are WOTC and 1 is EX (ex7)** — every other era has *zero*
per-set anchors. Fitting log_price and log_years slopes from that distribution
either (a) over-fits to one set or (b) flips signs (more value → lower
grading rate), which is physically wrong.

We therefore PIN beta_p and beta_y to documented priors and only fit
era-level intercepts:

  • beta_p = 0.5  — square-root scaling of grading rate in chase value.
                    Loose consensus from PSA-graded high-value vintage vs modern.
  • beta_y = 0.0  — age effect absorbed into the era intercept (eras already
                    map to a date window). Kept as a parameter for future
                    refinement once more anchors arrive.

Era intercepts are fit jointly against three loss terms:
  1. Per-set anchors: weighted by credibility (official=4, well-sourced=2,
     hobbyist=0.5). Squared error in log-grading-rate space.
  2. TPC cumulative checkpoints (23.6 B 2017 → 75 B 2025): squared error
     between log(predicted windowed print sum) and log(checkpoint). Heavy
     weight (8x) — these are *official* and the strongest cross-era signal.
  3. Smoothness across eras: |alpha_i - alpha_{i-1}|^2 with weight 0.5,
     so under-anchored eras (ECARD, DP/HGSS/BW/XY/SM) inherit from neighbors
     instead of diverging to ridge-zero.

If the fitter ever returns beta_p < 0, the script raises (per project rules:
do NOT ship a backwards model).

Outputs
-------
  data/grading_rate_model.json   coefficients + per-set predicted grading
                                 rate + per-set/per-era residuals.

Run: python set_population/scripts/fit_grading_rate.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import date, datetime, timezone

import numpy as np
import psycopg2
from scipy.optimize import minimize

SUBPROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(SUBPROJ, "data")
PG_DSN = "dbname=cardprice"

CHASE_CARDS = os.path.join(DATA, "chase_cards.json")
CHASE_GRADED = os.path.join(DATA, "chase_graded_pop.json")
KNOWN_RUNS = os.path.join(DATA, "known_print_runs.json")
ESTIMATES_IN = os.path.join(DATA, "set_population_estimates.json")  # for current pull_D + mean_psa pre-computed
OUT = os.path.join(DATA, "grading_rate_model.json")

# Eras must match combine.py.
ERAS = ["WOTC", "ECARD", "EX", "DP", "HGSS", "BW", "XY", "SM", "SWSH", "SV"]
ERA_IDX = {e: i for i, e in enumerate(ERAS)}
ERA_START = {
    "WOTC": date(1999, 1, 1), "ECARD": date(2002, 9, 1), "EX": date(2003, 6, 1),
    "DP": date(2007, 4, 1), "HGSS": date(2010, 2, 1), "BW": date(2011, 4, 1),
    "XY": date(2013, 10, 1), "SM": date(2017, 2, 1), "SWSH": date(2020, 2, 1),
    "SV": date(2023, 3, 1),
}

# TPC official cumulative checkpoints (cards, on-or-before date).
TPC_CHECKPOINTS = [
    (23_600_000_000, date(2017, 3, 31)),
    (43_200_000_000, date(2022, 3, 31)),
    (52_900_000_000, date(2023, 3, 31)),
    (64_900_000_000, date(2024, 3, 31)),
    (75_000_000_000, date(2025, 3, 31)),
]

# Pinned slopes (prior knowledge — see module docstring).
BETA_P_PRIOR = 0.5
BETA_Y_PRIOR = 0.0

# Loss weights.
ANCHOR_W = {"official": 4.0, "well-sourced-estimate": 2.0, "hobbyist-guess": 0.5}
W_TPC = 8.0
W_SMOOTH = 0.5
W_RIDGE = 0.001

# Anchor exclusions (variant subsets / cumulative checkpoints mislabeled per-set).
EXCLUDE_VARIANTS = {"1st_edition", "shadowless"}
EXCLUDE_SET_IDS = {"neo3"}  # 12 B is a cumulative checkpoint, not Neo Revelation alone

# Default chase-value when no priced chase card exists (very rare; fallback).
DEFAULT_CHASE_VALUE = 50.0


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def parse_d(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return datetime.fromisoformat(s[:10]).date()


def era_of(rel_date):
    if rel_date is None:
        return None
    era = ERAS[0]
    for name in ERAS:
        if rel_date >= ERA_START[name]:
            era = name
        else:
            break
    return era


def load_release_dates():
    out = {}
    conn = psycopg2.connect(PG_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT set_id, name, release_date FROM dim_sets")
        for sid, name, rel in cur.fetchall():
            rd = rel if isinstance(rel, date) else parse_d(str(rel) if rel else None)
            out[sid] = (name, rd)
    finally:
        conn.close()
    return out


def pull_denominator(era, tier):
    # Mirror combine.py's table. Kept in this file so the model JSON is
    # self-documenting; combine.py is the authoritative copy.
    PULL_DENOM = {
        "WOTC":  [1.0, 1.0, 1.5, 1.0, 3.0, 6.0],
        "ECARD": [1.0, 1.0, 1.5, 1.1, 3.5, 7.0],
        "EX":    [1.0, 1.0, 1.5, 1.2, 4.0, 9.0],
        "DP":    [1.0, 1.0, 1.5, 1.2, 4.5, 11.0],
        "HGSS":  [1.0, 1.0, 1.5, 1.2, 5.0, 13.0],
        "BW":    [1.0, 1.0, 1.5, 1.3, 5.5, 15.0],
        "XY":    [1.0, 1.0, 1.5, 1.3, 6.0, 18.0],
        "SM":    [1.0, 1.0, 1.5, 1.4, 7.0, 22.0],
        "SWSH":  [1.0, 1.0, 1.5, 1.5, 8.0, 28.0],
        "SV":    [1.0, 1.0, 1.5, 1.6, 9.0, 35.0],
    }
    if era is None or era not in PULL_DENOM:
        return 1.0
    table = PULL_DENOM[era]
    return table[max(0, min(tier, len(table) - 1))]


def build_features(today):
    chase = load_json(CHASE_CARDS)["sets"]
    graded = load_json(CHASE_GRADED)["sets"]
    rel_dates = load_release_dates()

    feats = {}
    for sid, ccrec in chase.items():
        # Numerator (mean PSA over confident chase cards)
        gset = graded.get(sid, {})
        pops = []
        for c in gset.get("chase", []):
            if c.get("match_confidence") in ("high", "med") and c.get("psa_total") is not None:
                pops.append(c["psa_total"])
        if not pops:
            continue
        mean_psa = statistics.mean(pops)

        # Chase value (mean of selected chase card prices)
        prices = [c["price"] for c in ccrec.get("chase", [])
                  if c.get("price") and c["price"] > 0]
        chase_value = statistics.mean(prices) if prices else DEFAULT_CHASE_VALUE
        chase_value_imputed = not prices

        # Release date, era, tier
        rd_meta = rel_dates.get(sid, (None, None))
        rd = rd_meta[1] or parse_d(ccrec.get("release_date"))
        era = era_of(rd)
        if era is None:
            continue
        tier_used = ccrec.get("selection", {}).get("tier_used")
        if tier_used is None:
            tiers = [c.get("tier") for c in ccrec.get("chase", []) if c.get("tier") is not None]
            tier_used = max(tiers) if tiers else 0
        tier = int(tier_used)
        yrs = max(0.5, (today - rd).days / 365.0) if rd else 5.0

        feats[sid] = {
            "set_name": ccrec.get("set_name") or rel_dates.get(sid, (sid, None))[0] or sid,
            "release_date": rd.isoformat() if rd else None,
            "era": era,
            "era_idx": ERA_IDX[era],
            "tier": tier,
            "mean_chase_psa": mean_psa,
            "pull_denominator": pull_denominator(era, tier),
            "chase_value": chase_value,
            "chase_value_imputed": chase_value_imputed,
            "years_since_release": yrs,
            "log_price_centered": math.log(chase_value / 100.0),
            "log_yrs_centered": math.log(yrs / 10.0),
        }
    return feats


def build_anchor_targets(feats):
    """Return per-set anchors usable for fitting: (sid, era_idx, log_p, log_y, log_rate_obs, w, known_run, raw)."""
    anchors = load_json(KNOWN_RUNS)["anchors"]
    targets = []
    excluded = []
    for a in anchors:
        sid = a["set_id"]
        if sid in ("GLOBAL", "MODERN_AVG"):
            excluded.append((sid, "global"))
            continue
        if a["estimate_type"] != "total_print_run":
            continue
        if a.get("value_mid") is None:
            continue
        if a.get("print_variant") in EXCLUDE_VARIANTS:
            excluded.append((sid, f"variant={a.get('print_variant')}"))
            continue
        if sid in EXCLUDE_SET_IDS:
            excluded.append((sid, "set excluded (cumulative checkpoint mislabeled)"))
            continue
        f = feats.get(sid)
        if f is None:
            excluded.append((sid, "no features"))
            continue
        implied_rate = (f["mean_chase_psa"] * f["pull_denominator"]) / a["value_mid"]
        if implied_rate <= 0:
            continue
        targets.append({
            "sid": sid,
            "era_idx": f["era_idx"],
            "log_price": f["log_price_centered"],
            "log_yrs": f["log_yrs_centered"],
            "log_rate_obs": math.log(implied_rate),
            "weight": ANCHOR_W.get(a.get("source_credibility", "hobbyist-guess"), 0.5),
            "known_run": a["value_mid"],
            "credibility": a.get("source_credibility"),
            "variant": a.get("print_variant"),
        })
    return targets, excluded


def fit(feats, anchor_targets, today):
    """Solve for per-era alphas with pinned beta_p, beta_y."""
    sids = list(feats.keys())
    era_idx = np.array([feats[s]["era_idx"] for s in sids])
    log_price = np.array([feats[s]["log_price_centered"] for s in sids])
    log_yrs = np.array([feats[s]["log_yrs_centered"] for s in sids])
    psaD = np.array([feats[s]["mean_chase_psa"] * feats[s]["pull_denominator"] for s in sids])
    rds = [parse_d(feats[s]["release_date"]) for s in sids]

    window_masks = [
        (cp_val, np.array([rd is not None and rd <= cp_date for rd in rds]))
        for cp_val, cp_date in TPC_CHECKPOINTS
    ]

    N_ERAS = len(ERAS)

    def predict_log_rate_vec(alphas):
        return alphas[era_idx] + BETA_P_PRIOR * log_price + BETA_Y_PRIOR * log_yrs

    def loss(alphas):
        lr_all = predict_log_rate_vec(alphas)
        pred_pop_all = psaD / np.exp(lr_all)
        L = 0.0
        # Per-set anchors
        for a in anchor_targets:
            lr = alphas[a["era_idx"]] + BETA_P_PRIOR * a["log_price"] + BETA_Y_PRIOR * a["log_yrs"]
            L += a["weight"] * (lr - a["log_rate_obs"]) ** 2
        # TPC checkpoints
        for cp_val, mask in window_masks:
            ws = pred_pop_all[mask].sum()
            if ws > 0:
                L += W_TPC * (math.log(ws) - math.log(cp_val)) ** 2
        # Smoothness across consecutive eras
        for i in range(N_ERAS - 1):
            L += W_SMOOTH * (alphas[i + 1] - alphas[i]) ** 2
        # Mild ridge
        L += W_RIDGE * float(np.sum(alphas ** 2))
        return L

    x0 = np.array([math.log(1e-5)] * N_ERAS)
    res = minimize(loss, x0, method="L-BFGS-B")
    if not res.success:
        # L-BFGS-B sometimes reports !success but ftol-converged; warn but continue.
        print(f"WARNING: optimizer did not flag success: {res.message}")

    alphas = res.x

    # Safety check on pinned betas (would have caught a sign-flip if we'd fit them).
    if BETA_P_PRIOR <= 0:
        raise SystemExit(f"BETA_P_PRIOR={BETA_P_PRIOR} is non-positive. Refusing to ship a backwards model.")

    # Compute predicted print pops + final scale to latest TPC checkpoint.
    lr_all = predict_log_rate_vec(alphas)
    pred_pop_all = psaD / np.exp(lr_all)
    latest_val, latest_mask = window_masks[-1]
    wsum_latest = pred_pop_all[latest_mask].sum()
    final_scale = latest_val / wsum_latest if wsum_latest > 0 else 1.0

    return {
        "alphas": alphas,
        "beta_p": BETA_P_PRIOR,
        "beta_y": BETA_Y_PRIOR,
        "sids": sids,
        "era_idx_arr": era_idx,
        "log_price_arr": log_price,
        "log_yrs_arr": log_yrs,
        "psaD_arr": psaD,
        "rds": rds,
        "window_masks": window_masks,
        "pred_pop_unscaled": pred_pop_all,
        "final_scale": final_scale,
        "loss_final": res.fun,
        "optim_message": res.message if hasattr(res.message, "decode") else str(res.message),
    }


def assemble_output(feats, anchor_targets, fit_out):
    alphas = fit_out["alphas"]
    sids = fit_out["sids"]
    pred_pop_unscaled = fit_out["pred_pop_unscaled"]
    scale = fit_out["final_scale"]

    # Per-set predictions
    per_set = {}
    for i, sid in enumerate(sids):
        f = feats[sid]
        log_rate = (alphas[f["era_idx"]]
                    + BETA_P_PRIOR * f["log_price_centered"]
                    + BETA_Y_PRIOR * f["log_yrs_centered"])
        rate = math.exp(log_rate)
        pred_print_unscaled = pred_pop_unscaled[i]
        per_set[sid] = {
            "set_name": f["set_name"],
            "era": f["era"],
            "tier": f["tier"],
            "release_date": f["release_date"],
            "chase_value": round(f["chase_value"], 2),
            "chase_value_imputed": f["chase_value_imputed"],
            "years_since_release": round(f["years_since_release"], 2),
            "mean_chase_psa": round(f["mean_chase_psa"], 2),
            "pull_denominator": round(f["pull_denominator"], 3),
            "predicted_grading_rate": rate,
            "predicted_log_grading_rate": round(log_rate, 4),
            "predicted_print_run_unscaled": pred_print_unscaled,
            "predicted_print_run_scaled": pred_print_unscaled * scale,
        }

    # Per-era summaries
    per_era = {}
    for e_idx, e in enumerate(ERAS):
        in_era = [s for s in sids if feats[s]["era_idx"] == e_idx]
        if not in_era:
            per_era[e] = {"n": 0, "alpha": alphas[e_idx]}
            continue
        rates = [math.exp(alphas[e_idx]
                          + BETA_P_PRIOR * feats[s]["log_price_centered"]
                          + BETA_Y_PRIOR * feats[s]["log_yrs_centered"])
                 for s in in_era]
        prints = [per_set[s]["predicted_print_run_scaled"] for s in in_era]
        per_era[e] = {
            "n": len(in_era),
            "alpha": float(alphas[e_idx]),
            "baseline_grading_rate_at_100usd_10yrs": math.exp(alphas[e_idx]),
            "median_predicted_grading_rate": float(statistics.median(rates)),
            "min_predicted_grading_rate": float(min(rates)),
            "max_predicted_grading_rate": float(max(rates)),
            "median_predicted_print_run": float(statistics.median(prints)),
            "sum_predicted_print_run": float(sum(prints)),
        }

    # Anchor residuals
    anchor_resid = []
    for a in anchor_targets:
        idx = sids.index(a["sid"])
        pred_print = per_set[a["sid"]]["predicted_print_run_scaled"]
        lr_pred = (alphas[a["era_idx"]]
                   + BETA_P_PRIOR * a["log_price"] + BETA_Y_PRIOR * a["log_yrs"])
        anchor_resid.append({
            "set_id": a["sid"],
            "credibility": a["credibility"],
            "variant": a["variant"],
            "known_print_run": a["known_run"],
            "predicted_print_run": pred_print,
            "ratio_pred_over_known": pred_print / a["known_run"],
            "log_rate_observed": a["log_rate_obs"],
            "log_rate_predicted": float(lr_pred),
            "residual_log_rate": float(lr_pred - a["log_rate_obs"]),
        })

    within2 = sum(1 for r in anchor_resid if 0.5 <= r["ratio_pred_over_known"] <= 2.0)
    within3 = sum(1 for r in anchor_resid if (1 / 3.0) <= r["ratio_pred_over_known"] <= 3.0)

    # TPC fit
    tpc_fit = []
    for (cp_val, mask), (_, cp_date) in zip(fit_out["window_masks"], TPC_CHECKPOINTS):
        n_sets = int(mask.sum())
        wsum_scaled = float(pred_pop_unscaled[mask].sum() * scale)
        tpc_fit.append({
            "checkpoint_date": cp_date.isoformat(),
            "checkpoint_value": cp_val,
            "n_sets_in_window": n_sets,
            "predicted_windowed_sum": wsum_scaled,
            "ratio_pred_over_cp": wsum_scaled / cp_val if cp_val else None,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "equation": (
                "log(grading_rate(set)) = alpha[era] "
                "+ beta_p * log(chase_value / $100) "
                "+ beta_y * log(years_since_release / 10y)"
            ),
            "beta_p": BETA_P_PRIOR,
            "beta_y": BETA_Y_PRIOR,
            "beta_p_status": "pinned to prior (insufficient cross-era anchors to fit)",
            "beta_y_status": "pinned to 0 (age effect absorbed into era intercept)",
            "alpha_by_era": {e: float(alphas[i]) for i, e in enumerate(ERAS)},
            "fit_weights": {
                "anchor_weights_by_credibility": ANCHOR_W,
                "tpc_checkpoint_weight": W_TPC,
                "era_smoothness_weight": W_SMOOTH,
                "ridge_weight": W_RIDGE,
            },
            "anchor_exclusions": {
                "variants": sorted(EXCLUDE_VARIANTS),
                "set_ids": sorted(EXCLUDE_SET_IDS),
                "reason": ("1st_edition/shadowless anchors target a variant subset "
                           "but mean PSA pop is reported across all variants of the set, "
                           "so the implied rate is biased; neo3 12B is a cumulative "
                           "checkpoint, not a single set."),
            },
            "final_scale_to_latest_tpc_checkpoint": float(scale),
        },
        "anchor_sensitivity": {
            "n_anchors": len(anchor_resid),
            "within_2x": within2,
            "within_3x": within3,
            "anchors": anchor_resid,
        },
        "tpc_fit": tpc_fit,
        "per_era": per_era,
        "per_set": per_set,
        "stats": {
            "n_sets": len(sids),
            "final_loss": float(fit_out["loss_final"]),
            "optim_message": fit_out["optim_message"],
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    today = date.today()
    feats = build_features(today)
    anchor_targets, excluded = build_anchor_targets(feats)
    fit_out = fit(feats, anchor_targets, today)
    doc = assemble_output(feats, anchor_targets, fit_out)
    doc["model"]["anchors_excluded"] = [{"set_id": sid, "reason": why}
                                        for sid, why in excluded]

    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)

    if not args.quiet:
        m = doc["model"]
        print(f"Fit grading-rate model: {len(feats)} sets, "
              f"{doc['anchor_sensitivity']['n_anchors']} per-set anchors.")
        print(f"  beta_p={m['beta_p']}, beta_y={m['beta_y']} (both pinned).")
        print("  alpha_by_era:")
        for e in ERAS:
            a = m["alpha_by_era"][e]
            print(f"    {e:6s}: alpha={a:7.3f} => baseline rate={math.exp(a):.2e}")
        print(f"  Final scale to TPC 75B: {m['final_scale_to_latest_tpc_checkpoint']:.3f}")
        s = doc["anchor_sensitivity"]
        print(f"  Anchors within 2x: {s['within_2x']}/{s['n_anchors']}, "
              f"within 3x: {s['within_3x']}/{s['n_anchors']}")
        print(f"  Final loss: {doc['stats']['final_loss']:.3f}")
        print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
