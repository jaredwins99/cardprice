#!/usr/bin/env python3
"""Run the three audits the adversarial reviewer demanded before funding.

Test 1 (trade-anchored, decisive): for recent cutoffs where actual sales are
observable (2026-04-01, 2026-05-01), replace entry marks with the FIRST actual
NM sale within 14 days after cutoff and exit with the LAST NM sale >= 21 days
later. Report (a) Q1 entry premium: median(actual entry trade / mark), and
(b) trade-anchored Q1-minus-universe spread vs the mark-based one.
Pass: >=50% of mark spread survives AND Q1 entry premium < 5%.

Test 2 (freshness partition): at the 8 non-overlapping cutoffs, split Q1 by
mark freshness = days since the mark last CHANGED (daily panel scan-back).
Pass: fresh(<14d) Q1 spread >= 50% of all-Q1 spread.

Test 3 (survivorship audit): drop the T+3m mark requirement from the universe,
count disappearance by value quintile, re-run spread imputing disappeared at
last mark x 0.70. Pass: spread degrades < 25%.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import psycopg2

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "bt", os.path.join(HERE, "backtest_factors.py"))
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

SALESDB = os.path.join(HERE, "..", "..", "data", "tcgplayer_sales.db")
OUT = os.path.join(HERE, "..", "data", "backtest_audits.json")


def quintiles(v):
    return pd.qcut(v.rank(method="first"), 5, labels=False)


def main():
    panel, cards = bt.load_panel()
    results = {}

    # ---------- Test 3: survivorship ----------------------------------------
    rows = []
    for T in bt.CUTOFFS:
        T3, F3 = T - pd.DateOffset(months=3), T + pd.DateOffset(months=3)
        if any(x not in panel.columns for x in (T3, T, F3)):
            continue
        px_T, px_T3, px_F3 = panel[T], panel[T3], panel[F3]
        ok = (px_T >= 1.0) & px_T3.notna() & (px_T3 > 0)   # NO T+3m requirement
        val = bt.value_residual_at(T, cards, px_T[ok])
        d = val.copy()
        d["px_T"] = px_T[ok].reindex(d.index)
        d["px_F3"] = px_F3.reindex(d.index)
        d["gone"] = d["px_F3"].isna() | (d["px_F3"] <= 0)
        d["fwd_imp"] = np.where(d["gone"], np.log(0.70),
                                np.log(d["px_F3"] / d["px_T"]))
        q = quintiles(d["value"])
        rows.append({
            "T": str(T.date()),
            "gone_by_quintile": [round(float(d.loc[q == i, "gone"].mean()), 4)
                                 for i in range(5)],
            "spread_imputed": float(d.loc[q == 0, "fwd_imp"].mean()
                                    - d.loc[q == 4, "fwd_imp"].mean()),
        })
    imp = pd.Series([r["spread_imputed"] for r in rows])
    gone_q1 = np.mean([r["gone_by_quintile"][0] for r in rows])
    gone_all = np.mean([np.mean(r["gone_by_quintile"]) for r in rows])
    results["test3_survivorship"] = {
        "mean_spread_imputed": round(float(imp.mean()), 4),
        "baseline_mark_spread": 0.0807,
        "degradation_pct": round(100 * (1 - imp.mean() / 0.0807), 1),
        "q1_disappearance_rate": round(float(gone_q1), 4),
        "universe_disappearance_rate": round(float(gone_all), 4),
        "per_cutoff": rows,
        "pass": bool(imp.mean() >= 0.0807 * 0.75),
    }
    print("TEST 3 survivorship:", {k: v for k, v in results["test3_survivorship"].items() if k != "per_cutoff"})

    # ---------- Test 2: freshness partition ---------------------------------
    pg = psycopg2.connect(dbname="cardprice")
    non_overlap = bt.CUTOFFS[::3]
    fresh_rows = []
    for T in non_overlap:
        T3, F3 = T - pd.DateOffset(months=3), T + pd.DateOffset(months=3)
        if any(x not in panel.columns for x in (T3, T, F3)):
            continue
        px_T, px_T3, px_F3 = panel[T], panel[T3], panel[F3]
        ok = (px_T >= 1.0) & px_T3.notna() & (px_T3 > 0) & px_F3.notna() & (px_F3 > 0)
        val = bt.value_residual_at(T, cards, px_T[ok])
        d = val.copy()
        d["fwd"] = np.log(px_F3[ok] / px_T[ok]).reindex(d.index)
        q = quintiles(d["value"])
        q1 = d[q == 0]
        mkt = d["fwd"].mean()
        pids = tuple(int(p) for p, _ in q1.index)
        # days since mark last changed, per (pid, printing), as of T
        f = pd.read_sql(f"""
            WITH w AS (
              SELECT tcg_product_id AS pid, subtype_name AS printing,
                     price_date, market_price,
                     LAG(market_price) OVER (PARTITION BY tcg_product_id,
                        subtype_name ORDER BY price_date) AS prev
              FROM fact_market_prices
              WHERE price_date BETWEEN '{(T - pd.DateOffset(days=90)).date()}'
                AND '{T.date()}'
                AND tcg_product_id IN {pids})
            SELECT pid, printing, max(price_date) AS last_change
            FROM w WHERE market_price IS DISTINCT FROM prev
            GROUP BY pid, printing""", pg)
        f["last_change"] = pd.to_datetime(f["last_change"])
        f["days_stale"] = (T - f["last_change"]).dt.days
        q1r = q1.reset_index().merge(f, on=["pid", "printing"], how="left")
        q1r["days_stale"] = q1r["days_stale"].fillna(90)
        fresh = q1r[q1r["days_stale"] < 14]["fwd"]
        stale = q1r[q1r["days_stale"] >= 14]["fwd"]
        fresh_rows.append({"T": str(T.date()),
                           "n_fresh": int(len(fresh)), "n_stale": int(len(stale)),
                           "q1_fresh_minus_mkt": float(fresh.mean() - mkt) if len(fresh) else None,
                           "q1_stale_minus_mkt": float(stale.mean() - mkt) if len(stale) else None,
                           "q1_all_minus_mkt": float(q1["fwd"].mean() - mkt)})
        print(f"  T={T.date()} fresh n={len(fresh)} exc={fresh.mean()-mkt:+.4f} | "
              f"stale n={len(stale)} exc={(stale.mean()-mkt) if len(stale) else float('nan'):+.4f}")
    pg.close()
    fa = pd.DataFrame(fresh_rows)
    results["test2_freshness"] = {
        "mean_q1_fresh_excess": round(float(fa["q1_fresh_minus_mkt"].dropna().mean()), 4),
        "mean_q1_stale_excess": round(float(fa["q1_stale_minus_mkt"].dropna().mean()), 4),
        "mean_q1_all_excess": round(float(fa["q1_all_minus_mkt"].mean()), 4),
        "per_cutoff": fresh_rows,
        "pass": bool(fa["q1_fresh_minus_mkt"].dropna().mean()
                     >= 0.5 * fa["q1_all_minus_mkt"].mean()),
    }
    print("TEST 2 freshness:", {k: v for k, v in results["test2_freshness"].items() if k != "per_cutoff"})

    # ---------- Test 1: trade-anchored --------------------------------------
    con = sqlite3.connect(f"file:{SALESDB}?mode=ro", uri=True)
    sales = pd.read_sql("""
        SELECT tcg_product_id AS pid, printing, condition, sale_price,
               substr(sale_date, 1, 10) AS d
        FROM tcgplayer_sales WHERE condition = 'Near Mint'""", con)
    con.close()
    sales["d"] = pd.to_datetime(sales["d"])
    t1_rows = []
    for T in [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")]:
        T3 = T - pd.DateOffset(months=3)
        px_T, px_T3 = panel[T], panel[T3]
        ok = (px_T >= 1.0) & px_T3.notna() & (px_T3 > 0)
        val = bt.value_residual_at(T, cards, px_T[ok])
        d = val.copy()
        d["px_T"] = px_T[ok].reindex(d.index)
        q = quintiles(d["value"])
        d["q"] = q.values

        def anchor(sub):
            recs = []
            for (pid, printing), r in sub.iterrows():
                s = sales[(sales["pid"] == pid) & (sales["printing"] == printing)]
                entry = s[(s["d"] >= T) & (s["d"] <= T + pd.Timedelta(days=14))]
                if not len(entry):
                    continue
                e_px = float(entry.sort_values("d").iloc[0]["sale_price"])
                e_d = entry.sort_values("d").iloc[0]["d"]
                exit_ = s[s["d"] >= e_d + pd.Timedelta(days=21)]
                if not len(exit_):
                    continue
                x_px = float(exit_.sort_values("d").iloc[-1]["sale_price"])
                recs.append({"entry": e_px, "exit": x_px, "mark": float(r["px_T"]),
                             "ret": np.log(x_px / e_px),
                             "entry_premium": e_px / float(r["px_T"]) - 1})
            return pd.DataFrame(recs)
        a_q1 = anchor(d[d["q"] == 0])
        a_uni = anchor(d.sample(min(len(d), 1500), random_state=0))
        if len(a_q1) and len(a_uni):
            t1_rows.append({
                "T": str(T.date()),
                "q1_n_anchored": int(len(a_q1)),
                "q1_coverage": round(len(a_q1) / (q == 0).sum(), 3),
                "q1_entry_premium_median": round(float(a_q1["entry_premium"].median()), 4),
                "q1_trade_ret": round(float(a_q1["ret"].mean()), 4),
                "universe_trade_ret": round(float(a_uni["ret"].mean()), 4),
                "q1_minus_universe_trade": round(float(a_q1["ret"].mean() - a_uni["ret"].mean()), 4),
            })
            print("TEST 1", t1_rows[-1])
    results["test1_trade_anchored"] = {
        "rows": t1_rows,
        "note": ("entry = first actual NM sale within 14d of cutoff; exit = last NM "
                 "sale >=21d after entry; horizon truncated by sales window ending "
                 "2026-07-21 — shorter than the backtest's 3m for the May cutoff"),
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
