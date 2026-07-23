#!/usr/bin/env python3
"""Hedged characteristic-factor backtest: which card traits actually paid?

The question this answers is the "Nishida factor" one: if you had gone long
every Atsuko Nishida card and hedged away everything else about them, would
you have earned anything?

METHOD (matched long-short, implemented as cell-demeaning):
  For each month t, compute each card-printing's log return. Demean those
  returns WITHIN (era x rarity-bucket x printing) cells. A cell-demeaned
  return is exactly the return of a long position financed by a short in a
  matched basket — so the era, rarity and printing exposures net to zero and
  what remains is attributable to the characteristic being tested.
  Then average the demeaned returns across all cards sharing a characteristic
  value (artist = Nishida, species = Charizard, pop-decile = 1, ...).

  monthly_alpha(v, t) = mean_{i in v} [ r_i(t) - mean(r in cell(i), t) ]

REPORTED per characteristic value: mean monthly alpha, t-stat over months,
hit rate, cumulative, and n. Monthly observations overlap nothing (1-month
returns), so the t-stats are honest, but note the whole sample is a single
2024-2026 bull regime and cross-sectional cells share a common market factor.

Usage: python3 pricing_model/scripts/backtest_characteristics.py
Writes data/characteristic_factors.json
"""

import json
import os

import numpy as np
import pandas as pd

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location(
    "bt", os.path.join(HERE, "backtest_factors.py"))
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

MIN_MONTHS = 12          # a characteristic must be observable most of the sample
MIN_CARDS = 15           # and rest on a real cross-section


def rarity_bucket(r):
    r = str(r or "")
    if r in ("Common", "Uncommon"):
        return "CU"
    if "Secret" in r or "Ultra" in r or "Rainbow" in r or "Hyper" in r:
        return "ULTRA"
    if "Holo" in r or "ex" in r or "EX" in r or "GX" in r or "V" in r:
        return "HOLO"
    return "RARE"


def main():
    panel, cards = bt.load_panel()
    months = [c for c in panel.columns]
    months.sort()

    # long-format monthly log returns
    px = panel[months]
    rets = np.log(px / px.shift(axis=1))
    rets = rets.replace([np.inf, -np.inf], np.nan)
    # require a real price base to avoid bulk noise dominating
    rets = rets.where(px.shift(axis=1) >= 1.0)

    meta = cards.set_index("pid")
    idx = pd.DataFrame(index=rets.index).reset_index()
    idx["rarity_b"] = idx["pid"].map(meta["rarity"]).map(rarity_bucket)
    idx["era"] = idx["pid"].map(meta["era"])
    idx["artist"] = idx["pid"].map(meta["artist"])
    idx["dex"] = idx["pid"].map(meta["dex"])
    idx["set_id"] = idx["pid"].map(meta["set_id"])
    idx["cell"] = (idx["era"].astype(str) + "|" + idx["rarity_b"]
                   + "|" + idx["printing"].astype(str))

    long = rets.stack().rename("r").reset_index()
    long.columns = ["pid", "printing", "month", "r"]
    long = long.merge(idx, on=["pid", "printing"], how="left")
    # hedge: subtract the cell's mean return that month
    long["cell_mean"] = long.groupby(["month", "cell"])["r"].transform("mean")
    long["cell_n"] = long.groupby(["month", "cell"])["r"].transform("size")
    long = long[long["cell_n"] >= 5]           # need a real hedge basket
    long["alpha"] = long["r"] - long["cell_mean"]

    results = {}

    def evaluate(label, key_col, min_cards=MIN_CARDS, top=25):
        rows = []
        for val, g in long.groupby(key_col):
            n_cards = g[["pid", "printing"]].drop_duplicates().shape[0]
            if n_cards < min_cards:
                continue
            m = g.groupby("month")["alpha"].mean()
            if len(m) < MIN_MONTHS:
                continue
            t = m.mean() / (m.std() / np.sqrt(len(m))) if m.std() > 0 else np.nan
            rows.append({
                "value": str(val), "n_cards": int(n_cards),
                "n_months": int(len(m)),
                "mean_monthly_alpha": round(float(m.mean()), 5),
                "t_stat": round(float(t), 2),
                "hit_rate": round(float((m > 0).mean()), 3),
                "cumulative": round(float(np.expm1(m.sum())), 3),
            })
        rows.sort(key=lambda x: -x["mean_monthly_alpha"])
        results[label] = rows
        print(f"\n=== {label} (hedged within era x rarity x printing) ===")
        show = rows[:top // 2] + ([{"value": "...", "n_cards": 0, "n_months": 0,
                                    "mean_monthly_alpha": 0, "t_stat": 0,
                                    "hit_rate": 0, "cumulative": 0}]
                                  if len(rows) > top else []) + rows[-(top // 2):] \
            if len(rows) > top else rows
        print(f"{'value':<28}{'n':>5}{'mo':>4}{'alpha/mo':>10}{'t':>7}{'hit':>6}{'cum':>8}")
        for x in show:
            print(f"{x['value'][:27]:<28}{x['n_cards']:>5}{x['n_months']:>4}"
                  f"{x['mean_monthly_alpha']:>10.4f}{x['t_stat']:>7.2f}"
                  f"{x['hit_rate']:>6.2f}{x['cumulative']:>8.2f}")
        return rows

    evaluate("artist", "artist", min_cards=30)
    evaluate("era", "era", min_cards=50)
    evaluate("printing", "printing", min_cards=50)
    evaluate("rarity_bucket", "rarity_b", min_cards=50)

    # species: use dex, label with name where possible
    long["dex_s"] = long["dex"].map(lambda d: f"dex{int(d)}" if pd.notna(d) else None)
    evaluate("species", "dex_s", min_cards=25, top=20)

    # ---- the specific ask: an isolated Nishida factor -----------------------
    nish = long[long["artist"] == "Atsuko Nishida"]
    m = nish.groupby("month")["alpha"].mean()
    ctrl = long[long["artist"].notna() & (long["artist"] != "Atsuko Nishida")]
    mc = ctrl.groupby("month")["alpha"].mean()
    results["nishida_factor"] = {
        "n_card_printings": int(nish[["pid", "printing"]].drop_duplicates().shape[0]),
        "months": int(len(m)),
        "mean_monthly_alpha": round(float(m.mean()), 5),
        "t_stat": round(float(m.mean() / (m.std() / np.sqrt(len(m)))), 2),
        "hit_rate": round(float((m > 0).mean()), 3),
        "cumulative_hedged_return": round(float(np.expm1(m.sum())), 4),
        "control_all_other_artists_mean": round(float(mc.mean()), 5),
        "monthly_series": {str(k.date()): round(float(v), 4) for k, v in m.items()},
    }
    print("\n=== THE NISHIDA FACTOR (long Nishida, short matched non-Nishida) ===")
    print(json.dumps({k: v for k, v in results["nishida_factor"].items()
                      if k != "monthly_series"}, indent=1))

    with open(os.path.join(DATA, "characteristic_factors.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {os.path.join(DATA, 'characteristic_factors.json')}")


if __name__ == "__main__":
    main()
