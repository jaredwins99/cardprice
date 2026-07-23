#!/usr/bin/env python3
"""Precompute per-row factor contributions for the fair-value model.

Alerts must be able to say WHY the model prices a card where it does, in the
ping itself. This re-runs the exact OOF fold structure of train_fair_value.py
and records each row's top feature contributions (LightGBM pred_contrib, in
log units => price multipliers), writing data/fv_explanations.parquet:

    pid, printing, top_factors  ("species $44 x2.4 | PSA pop 1363 x2.2 | ...")

Run after train_fair_value.py; cheap enough to re-run whenever the model does.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

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

# Human-readable factor labels; value formatter per feature
LABEL = {
    "species_loo_med": ("species", lambda v: f"${v:.0f} avg"),
    "artist_loo_med": ("artist", lambda v: f"${v:.0f} avg"),
    "normal_anchor": ("its Normal", lambda v: f"${v:.0f}"),
    "psa_pop": ("PSA pop", lambda v: f"{v:.0f}"),
    "psa10": ("PSA10s", lambda v: f"{v:.0f}"),
    "pop_vel8w": ("grading trend", lambda v: f"+{v:.0f}/8wk"),
    "set_age_days": ("age", lambda v: f"{v / 365:.0f}yr"),
    "era": ("era", lambda v: str(v)),
    "rarity": ("rarity", lambda v: str(v)),
    "printing": ("printing", lambda v: str(v)),
    "set_sales_pm": ("set velocity", lambda v: f"{v:.0f}/mo"),
    "set_abs_en": ("set supply", lambda v: f"{np.expm1(v) / 1e6:.0f}M"),
    "card_position": ("set position", lambda v: f"{v:.2f}"),
    "likeability": ("likeability", lambda v: f"{v:.0f}"),
    "dex": ("dex", lambda v: f"#{v:.0f}"),
    "total_cards": ("set size", lambda v: f"{v:.0f}"),
    "hp": ("HP", lambda v: f"{v:.0f}"),
    "stage": ("stage", lambda v: str(v)),
    "is_legendary": ("legendary", lambda v: "yes" if v else "no"),
}


def main():
    df = pd.read_parquet(os.path.join(DATA, "features.parquet"))
    df = df[df["label_lp"] > 0].reset_index(drop=True)
    y = np.log(df["label_lp"].values)
    X = df[CATS + NUMS].copy()
    raw = df[CATS + NUMS].copy()          # untransformed, for display
    for c in CATS:
        X[c] = X[c].astype("category")
    for c in ["normal_anchor", "species_loo_med", "artist_loo_med"]:
        X[c] = np.log(X[c].astype(float))
    for c in ["set_abs_en", "psa_pop"]:
        X[c] = np.log1p(X[c].astype(float))
    for c in NUMS:
        if X[c].dtype == object or str(X[c].dtype) == "bool":
            X[c] = X[c].astype(float)

    names = list(X.columns)
    tops = [None] * len(df)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X, y, df["set_id"].values)):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr], y[tr])
        contrib = m.predict(X.iloc[va], pred_contrib=True)[:, :-1]  # drop base
        for j, i in enumerate(va):
            c = contrib[j]
            order = np.argsort(-np.abs(c))[:3]
            parts = []
            for k in order:
                f = names[k]
                if abs(c[k]) < 0.05 or f not in LABEL:
                    continue
                lab, fmt = LABEL[f]
                v = raw.iloc[i][f]
                try:
                    vs = fmt(v) if pd.notna(v) else "n/a"
                except (TypeError, ValueError):
                    vs = str(v)
                parts.append(f"{lab} {vs} x{np.exp(c[k]):.1f}")
            tops[i] = " | ".join(parts)
        print(f"fold {fold} explained {len(va)} rows")

    out = df[["pid", "printing"]].copy()
    out["top_factors"] = tops
    path = os.path.join(DATA, "fv_explanations.parquet")
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {len(out)} rows")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
