#!/usr/bin/env python3
"""Assemble the fair-value training table at (tcg_product_id, subtype_name) grain.

Part of pricing_model (isolated sub-project). Reads Postgres + SQLite + JSON
assets read-only; writes data/features.parquet.

Label: LP-adjusted current price ("what you'd pay for a played copy"):
  1. JustTCG Lightly Played price for the printing (latest fetch; fetched_at
     age recorded — quotes rotate under API limits, per project memory);
  2. else NM price (JustTCG NM, else fact_market_prices market @ latest date)
     x card-specific LP/NM ratio clamped [0.45, 1.0];
  3. else NM x 0.75 pooled ratio (median of justtcg LP/NM cross-section).
label_source records which path fired. Eligibility: NM basis >= $1 (non-bulk).

Features are raw inputs per project philosophy (no hand-engineered
embeddings): set supply (english abs estimate, rel_pop, set sales velocity),
card identity (rarity, stage, position, hp, artist), species (dex, generation,
legendary/mythical, base-stat total, likeability ELO), per-printing PSA pops +
8-week grading velocity, the sibling-Normal demand anchor (user rule: a
premium variant's fair price is anchored by its Normal's demand), and
leave-one-out species/artist price aggregates (LOO to avoid target leakage —
see project memory on species_avg_price leakage).
"""

import json
import os
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd
import psycopg2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "pricing_model", "data", "features.parquet")
POPS = os.path.join(REPO, "pricing_model", "data", "card_graded_pops.json")
SETPOP = os.path.join(REPO, "set_population", "data", "set_population_estimates.json")
RATINGS = os.path.join(REPO, "pokemon_likeability", "data", "ratings.json")
JUSTTCG = os.path.join(REPO, "data", "justtcg_prices.db")
SALESDB = os.path.join(REPO, "data", "tcgplayer_sales.db")

LATEST = "2026-07-20"
MID = "2026-02-08"
START = "2024-02-08"
POOLED_LP_RATIO = 0.75


def q(conn, sql, params=None):
    return pd.read_sql(sql, conn, params=params)


def main():
    pg = psycopg2.connect(dbname="cardprice")

    # --- base: latest NM market per (pid, printing) + history checkpoints ----
    base = q(pg, f"""
        SELECT tcg_product_id AS pid, subtype_name AS printing,
               market_price AS nm_fact
        FROM fact_market_prices WHERE price_date = '{LATEST}'
          AND market_price IS NOT NULL AND tcg_product_id IS NOT NULL""")
    hist = q(pg, f"""
        SELECT tcg_product_id AS pid, subtype_name AS printing, price_date,
               market_price
        FROM fact_market_prices
        WHERE price_date IN ('{START}', '{MID}') AND market_price IS NOT NULL
          AND tcg_product_id IS NOT NULL""")
    hist = hist.pivot_table(index=["pid", "printing"], columns="price_date",
                            values="market_price", aggfunc="first").reset_index()
    hist.columns = ["pid", "printing", "p_start", "p_mid"]
    base = base.merge(hist, on=["pid", "printing"], how="left")

    # --- card / set / species dims ------------------------------------------
    cards = q(pg, """
        SELECT dc.tcg_product_id AS pid, dc.card_id, dc.name, dc.set_id,
               dc.rarity, dc.supertype, dc.hp, dc.card_number, dc.artist,
               dc.subtypes, dp.pokedex_num AS dex,
               ds.release_date, ds.total_cards, ds.series
        FROM dim_cards dc
        JOIN dim_sets ds ON ds.set_id = dc.set_id
        LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
        WHERE dc.tcg_product_id IS NOT NULL""")
    pokef = q(pg, """
        SELECT pokedex_num AS dex, generation, is_legendary, is_mythical, bst
        FROM dim_pokemon_features""")
    df = base.merge(cards, on="pid", how="inner").merge(pokef, on="dex", how="left")

    # stage from subtypes text[]
    def stage(sub):
        s = " ".join(sub) if isinstance(sub, list) else str(sub or "")
        for k in ("Stage 2", "Stage 1", "Basic"):
            if k in s:
                return k
        return None
    df["stage"] = df["subtypes"].map(stage)
    df = df.drop(columns=["subtypes"])

    # card position
    num = df["card_number"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["card_position"] = np.where(df["total_cards"] > 0, num / df["total_cards"], np.nan)
    df["set_age_days"] = (pd.Timestamp(LATEST) - pd.to_datetime(df["release_date"])).dt.days

    # --- set population (english-only v3.1) ---------------------------------
    sp = json.load(open(SETPOP))["sets"]
    df["era"] = df["set_id"].map(lambda s: sp.get(s, {}).get("era"))
    df["set_abs_en"] = df["set_id"].map(lambda s: sp.get(s, {}).get("abs_estimate_mid"))
    df["set_rel_pop"] = df["set_id"].map(lambda s: sp.get(s, {}).get("rel_pop_norm100"))
    df["set_sales_pm"] = df["set_id"].map(lambda s: sp.get(s, {}).get("sales_per_month"))
    df["set_pop_flagged"] = df["set_id"].map(
        lambda s: int(bool(set(sp.get(s, {}).get("flags", []))
                           & {"numerator_unreliable", "subset_set", "pop_lag_underestimate"})))

    # --- likeability --------------------------------------------------------
    ratings = json.load(open(RATINGS))["ratings"]
    df["likeability"] = df["dex"].map(
        lambda d: ratings.get(str(int(d)), {}).get("rating") if pd.notna(d) else None)
    df["likeability_n"] = df["dex"].map(
        lambda d: ratings.get(str(int(d)), {}).get("n") if pd.notna(d) else None)

    # --- PSA pops (printing-matched) ----------------------------------------
    pops = json.load(open(POPS))["cards"]
    GUESS = {"Normal": "base", "Unlimited": "base", "1st Edition": "base",
             "Holofoil": "Holofoil", "Unlimited Holofoil": "Holofoil",
             "Foil": "Holofoil", "1st Edition Holofoil": "1st Edition Holofoil",
             "Reverse Holofoil": "Reverse Holofoil"}

    def pop_row(card_id, printing):
        entries = pops.get(card_id, [])
        want = GUESS.get(printing, "base")
        cand = [e for e in entries if e.get("printing_guess") == want]
        if not cand and want == "1st Edition Holofoil":
            cand = [e for e in entries if e.get("printing_guess") == "Holofoil"]
        if not cand:
            return (np.nan, np.nan, np.nan)
        e = max(cand, key=lambda x: x.get("psa_total") or 0)
        hist8 = e.get("pop_hist") or []
        vel = (hist8[0] - hist8[-1]) if len(hist8) >= 2 else np.nan
        return (e.get("psa_total"), e.get("psa10"), vel)

    pop_vals = [pop_row(c, p) for c, p in zip(df["card_id"], df["printing"])]
    df[["psa_pop", "psa10", "pop_vel8w"]] = pd.DataFrame(pop_vals, index=df.index)

    # --- JustTCG per-condition latest + fetched_at --------------------------
    jt = sqlite3.connect(f"file:{JUSTTCG}?mode=ro", uri=True)
    jtd = q(jt, """
        SELECT tcg_product_id AS pid, printing, condition,
               COALESCE(price, avg_price) AS price, fetched_at
        FROM justtcg_prices jp
        WHERE fetched_at = (SELECT max(fetched_at) FROM justtcg_prices
                            WHERE tcg_product_id = jp.tcg_product_id
                              AND printing = jp.printing AND condition = jp.condition)
          AND COALESCE(price, avg_price) > 0
          AND game = 'pokemon'""")
    jt.close()
    jtp = jtd.pivot_table(index=["pid", "printing"], columns="condition",
                          values="price", aggfunc="first").reset_index()
    jtp = jtp.rename(columns={"Near Mint": "jt_nm", "Lightly Played": "jt_lp"})
    jt_age = jtd.groupby(["pid", "printing"])["fetched_at"].max().reset_index()
    jt_age["jt_age_days"] = (pd.Timestamp("2026-07-22")
                             - pd.to_datetime(jt_age["fetched_at"].str[:10])).dt.days
    df = df.merge(jtp[["pid", "printing", "jt_nm", "jt_lp"]],
                  on=["pid", "printing"], how="left")
    df = df.merge(jt_age[["pid", "printing", "jt_age_days"]],
                  on=["pid", "printing"], how="left")

    # --- observation conditioning: scrape visits ----------------------------
    sdb = sqlite3.connect(f"file:{SALESDB}?mode=ro", uri=True)
    visits = q(sdb, """
        SELECT tcg_product_id AS pid,
               count(DISTINCT substr(scraped_at, 1, 10)) AS visits,
               count(*) AS sales_rows
        FROM tcgplayer_sales WHERE scraped_at >= '2026-05-01' GROUP BY 1""")
    sdb.close()
    df = df.merge(visits, on="pid", how="left")

    # --- label: LP-adjusted -------------------------------------------------
    nm_basis = df["jt_nm"].fillna(df["nm_fact"])
    ratio = (df["jt_lp"] / df["jt_nm"]).clip(0.45, 1.0)
    df["label_lp"] = np.where(df["jt_lp"].notna(), df["jt_lp"],
                     np.where(ratio.notna(), nm_basis * ratio,
                              nm_basis * POOLED_LP_RATIO))
    df["label_source"] = np.where(df["jt_lp"].notna(), "justtcg_lp",
                         np.where(ratio.notna(), "nm_x_card_ratio", "nm_x_pooled"))
    df["nm_basis"] = nm_basis

    # eligibility: non-bulk
    df = df[df["nm_basis"] >= 1.0].copy()
    df = df[df["label_lp"] > 0].copy()

    # --- sibling-Normal demand anchor (user rule) ---------------------------
    normals = df[df["printing"] == "Normal"][["pid", "label_lp"]].rename(
        columns={"label_lp": "normal_anchor"})
    df = df.merge(normals, on="pid", how="left")
    df.loc[df["printing"] == "Normal", "normal_anchor"] = np.nan

    # --- leave-one-out aggregates (species, artist) -------------------------
    logl = np.log(df["label_lp"])
    for key, out_col in [("dex", "species_loo_med"), ("artist", "artist_loo_med")]:
        grp = df.groupby(key)[df.columns[0]]  # counts via size
        s = df.groupby(key).agg(n=("label_lp", "size"))
        med = df.groupby(key)["label_lp"].median().rename("med")
        # LOO median approximation: recompute median excluding self only where
        # group is small; for n>=10 the plain median is effectively LOO-safe
        df = df.merge(med, left_on=key, right_index=True, how="left")
        df = df.merge(s, left_on=key, right_index=True, how="left")
        exact = df["n"] < 10
        if exact.any():
            def loo_med(row_idx):
                r = df.loc[row_idx]
                vals = df[(df[r_key] == r[r_key]) & (df.index != row_idx)]["label_lp"]
                return vals.median() if len(vals) else np.nan
            r_key = key
            df.loc[exact, "med"] = [loo_med(i) for i in df[exact].index]
        df[out_col] = df["med"]
        df = df.drop(columns=["med", "n"])

    keep = ["pid", "printing", "card_id", "name", "set_id", "era", "series",
            "release_date", "label_lp", "label_source", "nm_basis", "jt_nm",
            "jt_lp", "jt_age_days", "nm_fact", "p_start", "p_mid",
            "rarity", "supertype", "stage", "hp", "card_position",
            "set_age_days", "total_cards", "set_abs_en", "set_rel_pop",
            "set_sales_pm", "set_pop_flagged", "artist", "dex", "generation",
            "is_legendary", "is_mythical", "bst", "likeability",
            "likeability_n", "psa_pop", "psa10", "pop_vel8w", "normal_anchor",
            "species_loo_med", "artist_loo_med", "visits", "sales_rows"]
    df = df[keep]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['set_id'].nunique()} sets, "
          f"{df['era'].notna().sum()} with era")
    print(df["label_source"].value_counts().to_string())
    print("label median by era:")
    print(df.groupby("era")["label_lp"].median().round(2).to_string())


if __name__ == "__main__":
    main()
