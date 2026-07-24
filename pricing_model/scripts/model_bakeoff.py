#!/usr/bin/env python3
"""Model bakeoff: 14 models, same folds, held-out-CARD out-of-sample.

The evaluation the project actually needed. Every model predicts log(LP price)
and is scored on OUT-OF-SAMPLE CARDS — 5-fold CV GROUPED BY SET, so a card's
prediction comes from a model that never saw it or any set-mate. This is the
cross-sectional ("one time index") comparison: given the fundamentals of cards
we've never priced, how close does each method get?

Fairness rules so the comparison is apples-to-apples:
  * ONE numeric design matrix for every model (numerics + one-hot of low-card
    categoricals + fold-pure target-encoded artist/species medians). No model
    gets extra information.
  * The two leakage-prone aggregates (species median, artist median) are
    recomputed INSIDE each training fold only — never from the held-out cards.
  * Identical fold assignment across all models.

Baselines included on purpose: a global-median null, and a comps model
(median of the era x rarity x printing cell) — the thing every price site
actually ships. A fancy model only earns its keep by beating the comps.

Metrics: MALE (mean |log error| = typical multiplicative miss), median ALE,
%-within-2x / within-1.5x, RMSE(log), R2(log), and Spearman rho (ranking
quality — the model's real job). Writes data/bakeoff_results.json + a per-era
breakdown for the top models.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

LOWCARD = ["printing", "era", "rarity", "supertype", "stage"]
NUMS = ["hp", "card_position", "set_age_days", "total_cards", "set_abs_en",
        "set_rel_pop", "set_sales_pm", "set_pop_flagged", "dex", "generation",
        "is_legendary", "is_mythical", "bst", "psa_pop", "psa10", "pop_vel8w",
        "normal_anchor"]
LOG_NUMS = ["set_abs_en", "psa_pop", "normal_anchor"]


def build_design(train, test):
    """Fold-pure numeric design matrices for train/test. Returns (Xtr, Xte,
    colnames). Target-encodes artist/species from TRAIN labels only."""
    parts_tr, parts_te, names = [], [], []

    # numerics (median-impute from train; log-transform skewed)
    for c in NUMS:
        tr = train[c].astype(float).copy()
        te = test[c].astype(float).copy()
        if c in LOG_NUMS:
            tr, te = np.log1p(tr.clip(lower=0)), np.log1p(te.clip(lower=0))
        med = tr.median()
        parts_tr.append(tr.fillna(med).values.reshape(-1, 1))
        parts_te.append(te.fillna(med).values.reshape(-1, 1))
        names.append(c)

    # one-hot low-card categoricals (levels from train)
    for c in LOWCARD:
        levels = [x for x in train[c].dropna().unique()]
        for lv in levels:
            parts_tr.append((train[c] == lv).astype(float).values.reshape(-1, 1))
            parts_te.append((test[c] == lv).astype(float).values.reshape(-1, 1))
            names.append(f"{c}={lv}")

    # fold-pure target encoding for high-card artist / species
    y = np.log(train["label_lp"].values)
    gmean = y.mean()
    for key, nm in [("artist", "artist_te"), ("dex", "species_te")]:
        enc = train.assign(_y=y).groupby(key)["_y"].agg(["mean", "size"])
        # shrink toward global mean for thin groups
        k = 5.0
        enc["sm"] = (enc["mean"] * enc["size"] + gmean * k) / (enc["size"] + k)
        m = enc["sm"].to_dict()
        parts_tr.append(train[key].map(m).fillna(gmean).values.reshape(-1, 1))
        parts_te.append(test[key].map(m).fillna(gmean).values.reshape(-1, 1))
        names.append(nm)

    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    # hard guard: any residual NaN/inf (e.g. a column all-NaN in this fold so
    # its median was NaN) -> column mean from train, else 0
    colmean = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)
    for X in (Xtr, Xte):
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(colmean, np.where(bad)[1])
    return Xtr, Xte, names


def make_models():
    from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV,
                                       ElasticNetCV)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import (RandomForestRegressor,
                                  GradientBoostingRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.neural_network import MLPRegressor
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    alphas = np.logspace(-3, 2, 20)
    return {
        "OLS": (LinearRegression(), True),
        "Ridge": (RidgeCV(alphas=alphas), True),
        "Lasso": (LassoCV(alphas=alphas, max_iter=5000), True),
        "ElasticNet": (ElasticNetCV(alphas=alphas, l1_ratio=[.2, .5, .8],
                                    max_iter=5000), True),
        "kNN(k=10)": (KNeighborsRegressor(n_neighbors=10, weights="distance"),
                      True),
        "RandomForest": (RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=0), False),
        "GBM(sklearn)": (GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=0), False),
        "HistGBM": (HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0), False),
        "LightGBM": (lgb.LGBMRegressor(
            objective="regression", metric="l1", num_leaves=63,
            learning_rate=0.05, n_estimators=1200, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, verbose=-1, n_jobs=-1), False),
        "XGBoost": (xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, verbosity=0), False),
        "CatBoost": (CatBoostRegressor(
            iterations=1200, depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
            loss_function="MAE", verbose=0, random_seed=0), False),
        "MLP": (MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
            early_stopping=True, random_state=0), True),
    }


def metrics(y, p):
    e = np.abs(y - p)
    ss = np.sum((y - y.mean()) ** 2)
    return {
        "MALE": round(float(e.mean()), 4),
        "median_ALE": round(float(np.median(e)), 4),
        "within_2x": round(float((e <= np.log(2)).mean()), 4),
        "within_1.5x": round(float((e <= np.log(1.5)).mean()), 4),
        "RMSE_log": round(float(np.sqrt(((y - p) ** 2).mean())), 4),
        "R2_log": round(float(1 - np.sum((y - p) ** 2) / ss), 4),
        "spearman": round(float(spearmanr(y, p).correlation), 4),
    }


def main():
    df = pd.read_parquet(os.path.join(DATA, "features.parquet"))
    df = df[df["label_lp"] > 0].reset_index(drop=True)
    y = np.log(df["label_lp"].values)
    groups = df["set_id"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, y, groups))

    models = make_models()
    oof = {name: np.full(len(df), np.nan) for name in models}
    oof["NullMedian"] = np.full(len(df), np.nan)
    oof["CellMedian(comps)"] = np.full(len(df), np.nan)
    timings = {name: 0.0 for name in list(models) + ["NullMedian",
                                                     "CellMedian(comps)"]}

    for fi, (tr, te) in enumerate(folds):
        train, test = df.iloc[tr], df.iloc[te]
        ytr = y[tr]
        # --- baselines (need no design matrix) ---
        oof["NullMedian"][te] = np.median(ytr)
        cell = train.assign(_y=ytr).groupby(
            ["era", "rarity", "printing"])["_y"].median()
        gm = np.median(ytr)
        key = list(zip(test["era"], test["rarity"], test["printing"]))
        oof["CellMedian(comps)"][te] = [cell.get(k, gm) for k in key]

        # --- design matrix, shared ---
        Xtr, Xte, _ = build_design(train, test)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        for name, (model, needs_scale) in models.items():
            t0 = time.time()
            A, B = (Xtr_s, Xte_s) if needs_scale else (Xtr, Xte)
            model.fit(A, ytr)
            oof[name][te] = model.predict(B)
            timings[name] += time.time() - t0
        print(f"fold {fi + 1}/5 done ({len(te)} test cards)")

    # --- score everything ---
    rows = []
    for name, pred in oof.items():
        m = metrics(y, pred)
        m["model"] = name
        m["fit_sec"] = round(timings.get(name, 0), 1)
        rows.append(m)
    res = pd.DataFrame(rows).sort_values("MALE").reset_index(drop=True)

    order = ["model", "MALE", "median_ALE", "within_2x", "within_1.5x",
             "RMSE_log", "R2_log", "spearman", "fit_sec"]
    print("\n=== BAKEOFF (held-out-card OOS, 5-fold grouped by set) ===")
    print(res[order].to_string(index=False))

    # per-era MALE for the top 5 models
    top = res["model"].head(5).tolist()
    era_tbl = {}
    for name in top:
        e = np.abs(y - oof[name])
        era_tbl[name] = {era: round(float(e[df["era"] == era].mean()), 3)
                         for era in sorted(df["era"].dropna().unique())}
    print("\n=== per-era MALE, top 5 ===")
    print(pd.DataFrame(era_tbl).to_string())

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_cards": int(len(df)), "cv": "5-fold GroupKFold by set_id",
        "target": "log(LP-adjusted price)",
        "note": ("held-out-CARD OOS; fold-pure target encoding; one shared "
                 "numeric design matrix for all models"),
        "ranking": res[order].to_dict("records"),
        "per_era_MALE_top5": era_tbl,
    }
    json.dump(out, open(os.path.join(DATA, "bakeoff_results.json"), "w"),
              indent=1)
    print(f"\nwrote {os.path.join(DATA, 'bakeoff_results.json')}")


if __name__ == "__main__":
    main()
