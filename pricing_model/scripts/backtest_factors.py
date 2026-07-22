#!/usr/bin/env python3
"""Backtest: cross-sectional factors on the card panel, 2024-05..2026-04.

Layer 1 (science, paper): monthly cutoffs T. At each T, signals use ONLY
information available at T:
  VALUE  = OOF residual from a reduced fair-value model retrained at T
           (features: structural card/set attrs + as-of-T normal-anchor and
           LOO species/artist aggregates; NO pops/likeability/justtcg — those
           are 2026 snapshots and would be lookahead).
  CMOM   = card trailing 3-month log return at T.
  GMOM   = era trailing 3-month log return minus cross-era mean (excess).
  SIZE   = log price at T.
Forward return = 3-month log NM return T -> T+3m. Quintile long-short spreads
per factor per month; double-sort value x group-momentum. Monthly cutoffs
overlap the 3m forward window — report both all-months stats and
non-overlapping (every 3rd month) stats; the latter is the honest t-stat.

Layer 2 (implementable, long-only): quarterly cohorts. Portfolio = cheapest
value quintile within eras of positive trailing excess momentum, top 30 by
residual, equal-weight, held 6 months, 15% round-trip cost charged per
position. Benchmark = equal-weight universe (no costs — conservative for us).

Universe at T: (tcg_product_id, subtype_name) with NM market price >= $1 at
T, at T-3m (for momentum), and a price at T+3m. Prices from fact_market_prices
(daily, complete). NM aggregate prices — value/momentum are cross-sectional so
the condition basis cancels.
"""

import json
import os
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SETPOP = os.path.join(HERE, "..", "..", "set_population", "data",
                      "set_population_estimates.json")

MONTHS = pd.date_range("2024-02-01", "2026-07-01", freq="MS")
CUTOFFS = [m for m in MONTHS if m >= pd.Timestamp("2024-05-01")
           and m + pd.DateOffset(months=3) <= pd.Timestamp("2026-07-01")]

CATS = ["printing", "era", "rarity", "stage", "artist"]
NUMS = ["set_age_days", "total_cards", "card_position", "set_rel_pop",
        "is_legendary", "dex", "normal_anchor", "species_loo", "artist_loo"]
PARAMS = dict(objective="regression", metric="l1", num_leaves=31,
              learning_rate=0.06, n_estimators=600, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1)


def load_panel():
    pg = psycopg2.connect(dbname="cardprice")
    dates = ",".join(f"'{m.date()}'" for m in MONTHS)
    px = pd.read_sql(f"""
        SELECT tcg_product_id AS pid, subtype_name AS printing,
               price_date, market_price AS px
        FROM fact_market_prices
        WHERE price_date IN ({dates}) AND market_price IS NOT NULL
          AND tcg_product_id IS NOT NULL""", pg)
    cards = pd.read_sql("""
        SELECT dc.tcg_product_id AS pid, dc.card_id, dc.set_id, dc.rarity,
               dc.supertype, dc.card_number, dc.artist, dc.subtypes,
               dp.pokedex_num AS dex, ds.release_date, ds.total_cards
        FROM dim_cards dc
        JOIN dim_sets ds ON ds.set_id = dc.set_id
        LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
        WHERE dc.tcg_product_id IS NOT NULL""", pg)
    feats = pd.read_sql("SELECT pokedex_num AS dex, is_legendary FROM dim_pokemon_features", pg)
    pg.close()
    px["price_date"] = pd.to_datetime(px["price_date"])
    panel = px.pivot_table(index=["pid", "printing"], columns="price_date",
                           values="px", aggfunc="first")
    cards = cards.merge(feats, on="dex", how="left")

    def stage(sub):
        s = " ".join(sub) if isinstance(sub, list) else str(sub or "")
        for k in ("Stage 2", "Stage 1", "Basic"):
            if k in s:
                return k
        return None
    cards["stage"] = cards["subtypes"].map(stage)
    num = cards["card_number"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    cards["card_position"] = np.where(cards["total_cards"] > 0,
                                      num / cards["total_cards"], np.nan)
    sp = json.load(open(SETPOP))["sets"]
    cards["era"] = cards["set_id"].map(lambda s: sp.get(s, {}).get("era"))
    cards["set_rel_pop"] = cards["set_id"].map(lambda s: sp.get(s, {}).get("rel_pop_norm100"))
    cards = cards.drop(columns=["subtypes", "card_number"])
    cards = cards.drop_duplicates(subset="pid", keep="first")
    return panel, cards


def value_residual_at(T, cards, px_T):
    """OOF residual of reduced model at cutoff T. px_T: Series (pid,printing)->px."""
    df = px_T.rename("px").reset_index().merge(cards, on="pid", how="inner")
    df = df[df["px"] >= 1.0].copy()
    df["set_age_days"] = (T - pd.to_datetime(df["release_date"])).dt.days
    normals = df[df["printing"] == "Normal"][["pid", "px"]].rename(columns={"px": "normal_anchor"})
    df = df.merge(normals, on="pid", how="left")
    df.loc[df["printing"] == "Normal", "normal_anchor"] = np.nan
    for key, col in [("dex", "species_loo"), ("artist", "artist_loo")]:
        med = df.groupby(key)["px"].transform("median")
        n = df.groupby(key)["px"].transform("size")
        df[col] = np.where(n >= 5, med, np.nan)  # plain median; n>=5 ~ LOO-safe
    X = df[CATS + NUMS].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    for c in ["normal_anchor", "species_loo", "artist_loo"]:
        X[c] = np.log(X[c].astype(float))
    for c in NUMS:
        if X[c].dtype == object or str(X[c].dtype) == "bool":
            X[c] = X[c].astype(float)
    y = np.log(df["px"].values)
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, y, df["set_id"].values):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    df["value"] = y - oof  # negative = cheap vs model
    return df.set_index(["pid", "printing"])[["value", "era", "set_id"]]


def spread_stats(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {}
    nov = s.iloc[::3]  # non-overlapping subsample
    return {
        "mean_monthly": round(float(s.mean()), 4),
        "t_all_overlapping": round(float(s.mean() / (s.std() / np.sqrt(len(s)))), 2),
        "n_months": int(len(s)),
        "mean_nonoverlap": round(float(nov.mean()), 4),
        "t_nonoverlap": round(float(nov.mean() / (nov.std() / np.sqrt(len(nov)))), 2)
        if len(nov) > 2 else None,
        "n_nonoverlap": int(len(nov)),
        "pct_positive": round(float((s > 0).mean()), 3),
    }


def main():
    panel, cards = load_panel()
    factor_rows = {"value": [], "cmom": [], "gmom": [], "size": []}
    ds_rows = []          # double-sort cells
    port_rows = []        # implementable cohorts

    SKIP = int(os.environ.get("SKIP_MONTHS", "0"))  # skip-month robustness
    for T in CUTOFFS:
        T3 = T - pd.DateOffset(months=3)
        E = T + pd.DateOffset(months=SKIP)          # entry date
        F3 = E + pd.DateOffset(months=3)
        if any(x not in panel.columns for x in (T3, T, E, F3)):
            continue
        px_T, px_T3, px_E, px_F3 = panel[T], panel[T3], panel[E], panel[F3]
        ok = (px_T >= 1.0) & px_T3.notna() & (px_T3 > 0) & px_F3.notna() \
            & (px_F3 > 0) & px_E.notna() & (px_E > 0)
        val = value_residual_at(T, cards, px_T[ok])
        d = val.copy()
        d["fwd"] = np.log(px_F3[ok] / px_E[ok]).reindex(d.index)
        d["cmom"] = np.log(px_T[ok] / px_T3[ok]).reindex(d.index)
        d["size"] = np.log(px_T[ok]).reindex(d.index)
        era_mom = d.groupby("era")["cmom"].mean()
        d["gmom"] = d["era"].map(era_mom - era_mom.mean())
        d = d.dropna(subset=["fwd", "value", "cmom", "size", "gmom"])
        mkt = d["fwd"].mean()

        for f, asc in [("value", True), ("cmom", False), ("gmom", False), ("size", True)]:
            qs = pd.qcut(d[f].rank(method="first"), 5, labels=False)
            long = d.loc[qs == (0 if asc else 4), "fwd"].mean()
            short = d.loc[qs == (4 if asc else 0), "fwd"].mean()
            factor_rows[f].append({"T": str(T.date()), "spread": long - short,
                                   "long": long, "short": short, "mkt": mkt,
                                   "n": len(d)})
        # double-sort: value quintiles x gmom sign
        vq = pd.qcut(d["value"].rank(method="first"), 5, labels=False)
        for gpos in (True, False):
            sel = d[(vq == 0) & ((d["gmom"] > 0) == gpos)]
            ds_rows.append({"T": str(T.date()), "gmom_positive": gpos,
                            "cheap_q_fwd": sel["fwd"].mean(), "n": len(sel),
                            "mkt": mkt})
        # implementable quarterly cohort
        if T.month in (2, 5, 8, 11) and T + pd.DateOffset(months=6) <= pd.Timestamp("2026-07-01"):
            F6 = T + pd.DateOffset(months=6)
            fwd6 = np.log(panel[F6] / px_T).reindex(d.index)
            pos = d[(d["gmom"] > 0) & (vq == 0)].copy()
            pos["fwd6"] = fwd6
            pos = pos.dropna(subset=["fwd6"]).nsmallest(30, "value")
            if len(pos) >= 10:
                gross = float(np.expm1(pos["fwd6"]).mean())
                net = float((1 + gross) * 0.85 - 1)  # 15% round-trip on exit
                bench = float(np.expm1(fwd6.dropna()).mean())
                port_rows.append({"T": str(T.date()), "n": len(pos),
                                  "gross_6m": round(gross, 4),
                                  "net_6m": round(net, 4),
                                  "benchmark_6m": round(bench, 4),
                                  "excess_net": round(net - bench, 4)})
        print(f"{T.date()}: n={len(d)} mkt3m={mkt:+.3f}")

    results = {
        "factors": {f: {"stats": spread_stats([r["spread"] for r in rows]),
                        "monthly": rows} for f, rows in factor_rows.items()},
        "double_sort": ds_rows,
        "implementable": port_rows,
        "spec": {"cutoffs": [str(c.date()) for c in CUTOFFS],
                 "forward_window_months": 3,
                 "universe": "NM market >= $1 at T, priced at T-3m and T+3m",
                 "costs_implementable": "15% round-trip charged to strategy only",
                 "value_model_features": CATS + NUMS},
    }
    with open(os.path.join(DATA, "backtest_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)

    print("\n=== FACTOR SPREADS (monthly Q1-Q5 long-short, 3m fwd log) ===")
    for f, rows in factor_rows.items():
        print(f"{f:6s}: {spread_stats([r['spread'] for r in rows])}")
    ds = pd.DataFrame(ds_rows)
    print("\n=== DOUBLE SORT: cheapest-value quintile, by group-momentum sign ===")
    print(ds.groupby("gmom_positive")[["cheap_q_fwd", "mkt"]].mean().round(4).to_string())
    print("\n=== IMPLEMENTABLE (quarterly cohorts, 6m hold, 15% RT cost) ===")
    print(pd.DataFrame(port_rows).to_string(index=False))


if __name__ == "__main__":
    main()
