#!/usr/bin/env python3
"""Train the fair-value model on log(LP-adjusted price) and emit residuals.

LightGBM at (tcg_product_id, printing) grain, grouped 5-fold CV BY SET_ID
(random-row splits leak set identity — project lesson). All predictions are
out-of-fold: every card's fair value comes from a model that never saw its
set. Metrics: MALE (mean |log e|), %-within-2x, by era. Outputs:
  data/fair_value_oof.parquet  — per-row OOF prediction + residual
  data/residuals.json          — ranked under/overpriced (top tails)
  data/fair_value_metrics.json — metrics + feature importances
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FEATURES = os.path.join(DATA, "features.parquet")

CATS = ["printing", "era", "rarity", "supertype", "stage", "artist"]
NUMS = ["hp", "card_position", "set_age_days", "total_cards", "set_abs_en",
        "set_rel_pop", "set_sales_pm", "set_pop_flagged", "dex", "generation",
        "is_legendary", "is_mythical", "bst", "likeability", "likeability_n",
        "psa_pop", "psa10", "pop_vel8w", "normal_anchor", "species_loo_med",
        "artist_loo_med"]

PARAMS = dict(objective="regression", metric="l1", num_leaves=63,
              learning_rate=0.05, n_estimators=1200, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1)


def main():
    df = pd.read_parquet(FEATURES)
    df = df[df["label_lp"] > 0].reset_index(drop=True)
    y = np.log(df["label_lp"].values)

    X = df[CATS + NUMS].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    for c in ["normal_anchor", "species_loo_med", "artist_loo_med"]:
        X[c] = np.log(X[c].astype(float))
    for c in ["set_abs_en", "psa_pop"]:
        X[c] = np.log1p(X[c].astype(float))
    for c in NUMS:
        if X[c].dtype == object or str(X[c].dtype) == "bool":
            X[c] = X[c].astype(float)

    groups = df["set_id"].values
    oof = np.full(len(df), np.nan)
    importances = np.zeros(X.shape[1])
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr], y[tr],
              eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va])
        importances += m.booster_.feature_importance("gain")
        print(f"fold {fold}: n_va={len(va)} "
              f"MALE={np.mean(np.abs(oof[va]-y[va])):.3f}")

    df["log_pred"] = oof
    df["fair_value"] = np.exp(oof)
    df["residual_log"] = y - oof          # <0 => actual BELOW fair (underpriced)
    df["ratio_actual_over_fair"] = np.exp(df["residual_log"])
    df["residual_pctl"] = df["residual_log"].rank(pct=True).round(4)

    male = float(np.mean(np.abs(df["residual_log"])))
    w2 = float(np.mean(np.abs(df["residual_log"]) <= np.log(2)))
    by_era = df.groupby("era").apply(
        lambda g: pd.Series({
            "n": len(g),
            "MALE": float(np.mean(np.abs(g["residual_log"]))),
            "within_2x": float(np.mean(np.abs(g["residual_log"]) <= np.log(2))),
            "med_label": float(g["label_lp"].median()),
        })).round(3)
    print(f"\nOVERALL: n={len(df)} MALE={male:.3f} within2x={w2:.1%}")
    print(by_era.to_string())

    imp = (pd.Series(importances, index=X.columns)
           .sort_values(ascending=False) / importances.sum())
    print("\ntop features (gain share):")
    print(imp.head(15).round(4).to_string())

    df.to_parquet(os.path.join(DATA, "fair_value_oof.parquet"), index=False)

    def rows(sub):
        return [{
            "card_id": r.card_id, "name": r.name, "set_id": r.set_id,
            "era": r.era, "printing": r.printing,
            "actual_lp": round(float(r.label_lp), 2),
            "fair_value_lp": round(float(r.fair_value), 2),
            "ratio": round(float(r.ratio_actual_over_fair), 3),
            "label_source": r.label_source,
            "jt_age_days": None if pd.isna(r.jt_age_days) else int(r.jt_age_days),
            "visits": None if pd.isna(r.visits) else int(r.visits),
            "psa_pop": None if pd.isna(r.psa_pop) else int(r.psa_pop),
            "normal_anchor": None if pd.isna(r.normal_anchor) else round(float(r.normal_anchor), 2),
        } for r in sub.itertuples()]

    liquid = df[(df["label_lp"] >= 3)]
    under = liquid.nsmallest(150, "residual_log")
    over = liquid.nlargest(150, "residual_log")
    out = {
        "generated_at": pd.Timestamp("2026-07-22").isoformat(),
        "model": "lightgbm fair-value v1, OOF grouped by set",
        "metrics": {"n": len(df), "MALE": round(male, 4),
                    "within_2x": round(w2, 4),
                    "by_era": json.loads(by_era.to_json(orient="index"))},
        "feature_importance_gain": json.loads(imp.round(5).to_json()),
        "underpriced_top150": rows(under),
        "overpriced_top150": rows(over),
        "caveats": [
            "residual<1 means actual LP price sits below model fair value; "
            "verify JustTCG quote age (jt_age_days) and scrape visits before "
            "acting - stale quotes masquerade as mispricing",
            "model includes sibling-Normal anchor + LOO species/artist "
            "aggregates: residuals are 'vs comparable cards AND own demand "
            "anchors', i.e. relative value, not absolute macro calls",
        ],
    }
    with open(os.path.join(DATA, "residuals.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(DATA, "fair_value_metrics.json"), "w") as f:
        json.dump(out["metrics"] | {"feature_importance": out["feature_importance_gain"]}, f, indent=1)
    print(f"\nwrote residuals.json ({len(under)} under / {len(over)} over)")


if __name__ == "__main__":
    main()
