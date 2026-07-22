#!/usr/bin/env python3
"""Mechanistic grading-rate model for the set_population sub-project.

Replaces the Google-Trends popularity divisor with a model of *grading rate*:

    grading_rate(set) = exp(alpha_era + beta_p * log(chase_value / $100)
                                       + beta_y * log(years_since_release / 10y))

Print population (relative) is then

    rel_print_pop(set) ∝ mean_chase_psa(set) * pull_denominator(set) / grading_rate(set)

with the final scale set by a credibility-weighted geometric mean over the
usable rungs of the TPC "Pokémon in Figures" cumulative ladder. v3
(2026-07-21/22): all calibration targets are ENGLISH cards — the dated
checkpoint ladder is loaded from known_print_runs.json GLOBAL anchors,
converted via the documented english_share() layer applied to per-regime
INCREMENTS, and capped at (pop snapshot − 730 d) so unscored/pop-lagged
recent sets cannot push production onto older sets; all-languages per-set
anchors are converted at the set's release-date share; the SEC-revenue
WOTC 1999-2001 English window enters as a consistency-check loss term; and
subset products are excluded from checkpoint windows.

Why this design and not OLS on the anchors directly
---------------------------------------------------
The usable per-set anchors (17 as of v3: 8 WOTC + 1 EX + 1 XY + 1 SM + 5
SWSH + 2 SV, after excluding 1st-Ed/Shadowless variant subsets and pseudo-id
rows) are ALL hobbyist-guess credibility — every one traces to community
round numbers with no documented derivation (see anchor_research doc).
Fitting log_price and log_years slopes against guesses either (a) over-fits
or (b) flips signs (more value → lower grading rate), which is physically
wrong.

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

# v3: the cumulative-checkpoint ladder is loaded from known_print_runs.json
# GLOBAL anchors (16 dated rungs, 12 official/archive-verified) and converted
# to ENGLISH cards via the documented english_share() layer. See
# model_constants_v3.py and docs/anchor_research_2026-07-21.md.
from model_constants_v3 import (  # noqa: E402
    MODEL_VERSION_V3, SUBSET_PARENT, anchor_value_english,
    calibration_scale, english_share, load_checkpoints,
    load_english_window_anchors, production_weight, share_doc, usable_rungs)

# Pinned slopes (prior knowledge — see module docstring).
BETA_P_PRIOR = 0.5
BETA_Y_PRIOR = 0.0

# Loss weights. (Checkpoint weights are per-rung by credibility — see
# model_constants_v3.CHECKPOINT_W: official=8.0, well-sourced-estimate=4.0.)
ANCHOR_W = {"official": 4.0, "well-sourced-estimate": 2.0, "hobbyist-guess": 0.5}
W_SMOOTH = 0.5
W_RIDGE = 0.001

# Anchor exclusions (variant subsets / cumulative checkpoints mislabeled per-set).
EXCLUDE_VARIANTS = {"1st_edition", "shadowless"}
# The 12B end-2001 checkpoint was reclassified from set_id neo3 to GLOBAL on
# 2026-07-21; the exclusion stays as a guard against stale data files.
EXCLUDE_SET_IDS = {"neo3"}

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
        if (sid in ("GLOBAL", "MODERN_AVG") or sid.endswith("_ERA_AVG")
                or sid.startswith("WOTC_")):
            excluded.append((sid, "pseudo-id (global / era-average / window)"))
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
        # v3: anchors tagged cards_all_languages are converted to ENGLISH
        # cards at the set's release-date share before implying a rate
        # (the numerator mean_chase_psa is English-catalog PSA pop).
        rd = parse_d(f["release_date"])
        value_en, converted = anchor_value_english(a, rd)
        implied_rate = (f["mean_chase_psa"] * f["pull_denominator"]) / value_en
        if implied_rate <= 0:
            continue
        targets.append({
            "sid": sid,
            "era_idx": f["era_idx"],
            "log_price": f["log_price_centered"],
            "log_yrs": f["log_yrs_centered"],
            "log_rate_obs": math.log(implied_rate),
            "weight": ANCHOR_W.get(a.get("source_credibility", "hobbyist-guess"), 0.5),
            "known_run": value_en,
            "known_run_raw": a["value_mid"],
            "unit_converted_from_all_languages": converted,
            "credibility": a.get("source_credibility"),
            "variant": a.get("print_variant"),
        })
    return targets, excluded


def fit(feats, anchor_targets, checkpoints, en_windows, today):
    """Solve for per-era alphas with pinned beta_p, beta_y.

    v3: all targets are ENGLISH cards — checkpoints arrive pre-converted
    (value_english), per-set anchors were converted in build_anchor_targets,
    and en_windows (e.g. the SEC-revenue-derived WOTC 1999-2001 total) are
    natively English. Checkpoint loss weight is per-rung credibility."""
    sids = list(feats.keys())
    era_idx = np.array([feats[s]["era_idx"] for s in sids])
    log_price = np.array([feats[s]["log_price_centered"] for s in sids])
    log_yrs = np.array([feats[s]["log_yrs_centered"] for s in sids])
    psaD = np.array([feats[s]["mean_chase_psa"] * feats[s]["pull_denominator"] for s in sids])
    rds = [parse_d(feats[s]["release_date"]) for s in sids]

    # v3: soft window weights — a set released d days before the checkpoint
    # contributes min(1, d/ramp) of its lifetime run (ramp: 365d boom-era,
    # 730d after). Subset products (Trainer Gallery, cel25c, ...) are zeroed:
    # their production is inside their parent's and must not double-claim
    # window volume.
    subset_zero = np.array([0.0 if s in SUBSET_PARENT else 1.0 for s in sids])
    window_masks = [
        (cp, np.array([production_weight(rd, cp["date"]) for rd in rds]) * subset_zero)
        for cp in checkpoints
    ]
    en_window_masks = [
        (w, np.array([production_weight(rd, w["window_end"])
                      if rd is not None and rd >= w["window_start"] else 0.0
                      for rd in rds]) * subset_zero)
        for w in en_windows
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
        # Cumulative checkpoints (English-converted), credibility-weighted,
        # production-ramp-weighted windows
        for cp, wts in window_masks:
            ws = float(pred_pop_all @ wts)
            if ws > 0:
                L += cp["weight"] * (math.log(ws) - math.log(cp["value_english"])) ** 2
        # English window anchors (revenue-derived, independent of the ladder)
        for w, wts in en_window_masks:
            ws = float(pred_pop_all @ wts)
            if ws > 0:
                L += w["weight"] * (math.log(ws) - math.log(w["value_mid"])) ** 2
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

    # Compute predicted print pops + calibration scale (weighted geomean
    # across all usable rungs — see model_constants_v3.calibration_scale).
    lr_all = predict_log_rate_vec(alphas)
    pred_pop_all = psaD / np.exp(lr_all)
    final_scale = calibration_scale([
        (cp["value_english"], float(pred_pop_all @ wts), cp["weight"])
        for cp, wts in window_masks
    ])

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
        "en_window_masks": en_window_masks,
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
        pred_print = per_set[a["sid"]]["predicted_print_run_scaled"]
        lr_pred = (alphas[a["era_idx"]]
                   + BETA_P_PRIOR * a["log_price"] + BETA_Y_PRIOR * a["log_yrs"])
        anchor_resid.append({
            "set_id": a["sid"],
            "credibility": a["credibility"],
            "variant": a["variant"],
            "known_print_run_english": a["known_run"],
            "known_print_run_raw": a["known_run_raw"],
            "unit_converted_from_all_languages": a["unit_converted_from_all_languages"],
            "predicted_print_run": pred_print,
            "ratio_pred_over_known": pred_print / a["known_run"],
            "log_rate_observed": a["log_rate_obs"],
            "log_rate_predicted": float(lr_pred),
            "residual_log_rate": float(lr_pred - a["log_rate_obs"]),
        })

    within2 = sum(1 for r in anchor_resid if 0.5 <= r["ratio_pred_over_known"] <= 2.0)
    within3 = sum(1 for r in anchor_resid if (1 / 3.0) <= r["ratio_pred_over_known"] <= 3.0)

    # Checkpoint fit (v3: English targets; production-ramp-weighted windows)
    tpc_fit = []
    for cp, wts in fit_out["window_masks"]:
        wsum_scaled = float((pred_pop_unscaled @ wts) * scale)
        tpc_fit.append({
            "checkpoint_date": cp["date"].isoformat(),
            "checkpoint_value_global": cp["value_global"],
            "english_share": english_share(cp["date"]),
            "checkpoint_value_english": cp["value_english"],
            "credibility": cp["credibility"],
            "loss_weight": cp["weight"],
            "n_sets_in_window": int((wts > 0).sum()),
            "effective_sets_in_window": round(float(wts.sum()), 1),
            "predicted_windowed_sum": wsum_scaled,
            "ratio_pred_over_cp": wsum_scaled / cp["value_english"] if cp["value_english"] else None,
        })

    # English window anchor fit (revenue-derived)
    window_fit = []
    for w, wts in fit_out["en_window_masks"]:
        wsum_scaled = float((pred_pop_unscaled @ wts) * scale)
        window_fit.append({
            "anchor_id": w["set_id"],
            "window": [w["window_start"].isoformat(), w["window_end"].isoformat()],
            "target_english_mid": w["value_mid"],
            "target_english_low": w["value_low"],
            "target_english_high": w["value_high"],
            "n_sets_in_window": int((wts > 0).sum()),
            "effective_sets_in_window": round(float(wts.sum()), 1),
            "predicted_windowed_sum": wsum_scaled,
            "ratio_pred_over_target": wsum_scaled / w["value_mid"] if w["value_mid"] else None,
            "within_band": (w["value_low"] is not None and w["value_high"] is not None
                            and w["value_low"] <= wsum_scaled <= w["value_high"]),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION_V3,
        "language_scope": "english_only",
        "english_share": share_doc(),
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
                "checkpoint_weight_by_credibility": {"official": 8.0,
                                                     "well-sourced-estimate": 4.0},
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
            "calibration_scale_geomean_over_usable_rungs": float(scale),
        },
        "anchor_sensitivity": {
            "n_anchors": len(anchor_resid),
            "within_2x": within2,
            "within_3x": within3,
            "anchors": anchor_resid,
        },
        "tpc_fit": tpc_fit,
        "english_window_fit": window_fit,
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
    known_runs = load_json(KNOWN_RUNS)
    pop_snapshot = parse_d(load_json(CHASE_GRADED).get("generated_at")) or today
    checkpoints = usable_rungs(load_checkpoints(known_runs), pop_snapshot)
    en_windows = load_english_window_anchors(known_runs)
    if not checkpoints:
        raise SystemExit("No usable dated GLOBAL checkpoints in known_print_runs.json — "
                         "cannot calibrate. (as_of_date fields missing?)")
    fit_out = fit(feats, anchor_targets, checkpoints, en_windows, today)
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
        last_cp = doc["tpc_fit"][-1]
        print(f"  Calibration scale (geomean over {len(doc['tpc_fit'])} usable "
              f"rungs, latest {last_cp['checkpoint_value_global']/1e9:.0f}B @ "
              f"{last_cp['checkpoint_date']}): "
              f"{m['calibration_scale_geomean_over_usable_rungs']:.3f}")
        for wf in doc.get("english_window_fit", []):
            print(f"  Revenue window {wf['anchor_id']}: pred "
                  f"{wf['predicted_windowed_sum']/1e9:.1f}B vs target "
                  f"{wf['target_english_mid']/1e9:.1f}B "
                  f"[{(wf['target_english_low'] or 0)/1e9:.1f}-"
                  f"{(wf['target_english_high'] or 0)/1e9:.1f}] "
                  f"ratio {wf['ratio_pred_over_target']:.2f} "
                  f"within_band={wf['within_band']}")
        s = doc["anchor_sensitivity"]
        print(f"  Anchors within 2x: {s['within_2x']}/{s['n_anchors']}, "
              f"within 3x: {s['within_3x']}/{s['n_anchors']}")
        print(f"  Final loss: {doc['stats']['final_loss']:.3f}")
        print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
