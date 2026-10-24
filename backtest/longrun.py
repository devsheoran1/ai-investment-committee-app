"""
Long / stress stock backtest (Oct 24): 2018-01 -> today on the same rule-based sleeve as run_backtest (N0 old deployed, N1 new, N2 new + regime
allocator), reported per stress sub-period. Survivorship: the universe is TODAY's S&P 500, so every long-only number is flattered, and names that
joined the index after 2018 are only eligible once they have 252 sessions of history. Earnings history from yfinance (up to 100 quarters).
    /opt/miniconda3/bin/python3 -m backtest.longrun
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import config
from backtest import data_bt, engine as E
from backtest.run_backtest import md_table, num, pct
from strategies import regime as R

OUT = os.path.join(config.BASE_DIR, "reports", "backtest")
START = "2018-01-02"
OFFSETS = [0, 7]
PERIODS = {"2018 Q4": ("2018-10-01", "2018-12-31"), "2020 Feb-Jun": ("2020-02-01", "2020-06-30"), "2022": ("2022-01-01", "2022-12-31"),
           "2023-26": ("2023-01-01", "2026-12-31"), "Full 2018-26": ("2018-01-01", "2026-12-31")}


def prepare_long() -> E.Prep:
    px = data_bt.load_prices_long()
    uni = data_bt.universe_frame()
    stocks = uni.loc[uni.asset_type == "stock", "symbol"].tolist()
    earn = data_bt.load_earnings(stocks, budget_s=1, long=True)
    return E.prepare(px, uni, earn)


def period_metrics(nav: pd.Series, a: str, b: str) -> dict:
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    if len(s) < 5:
        return {"Return": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    r = s.pct_change().dropna()
    return {"Return": float(s.iloc[-1] / s.iloc[0] - 1), "Sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "MaxDD": float((s / s.cummax() - 1).min())}


def main() -> dict:
    t0 = time.time()
    P = prepare_long()
    OLD, NEW = {"weight_mode": "ic_legacy"}, {"weight_mode": "shrinkage"}
    W_old = E.weight_history(P, cfg=OLD)
    s_o, v_o, a_o = E.ensemble_panels(P, W_old)
    dep_old = E.as_deployed(P, s_o, a_o, blend=0.75)
    sc = {}
    for lab, reg in (("N1", False), ("N2", True)):
        W = E.weight_history(P, cfg=NEW, regime=reg)
        s_, v_, a_ = E.ensemble_panels(P, W)
        sc[lab] = (E.as_deployed(P, s_, a_, blend=1.0), v_)
    variants = {"N0 old deployed (IC weights, blend 0.75, -8% trail)": (dep_old, v_o, -0.08),
                "N1 new (shrinkage, 100% ensemble, -12% trail)": (*sc["N1"], config.TRAILING_STOP),
                "N2 new + regime allocator": (*sc["N2"], config.TRAILING_STOP)}
    navs = {}
    for lab, (score, votes, trail) in variants.items():
        navs[lab] = [E.simulate(P, score, votes, start=str((pd.Timestamp(START) + pd.Timedelta(days=o)).date()), trail_pct=trail)["nav"]
                     for o in OFFSETS]
        print(f"{lab}: done ({time.time() - t0:.0f}s)", flush=True)
    bench = {b: E.buy_hold(P.close, P.open, b, START) for b in data_bt.BENCH}
    rows = {}
    for pname, (a, b) in PERIODS.items():
        r = {}
        for lab, ns in navs.items():
            ms = [period_metrics(n, a, b) for n in ns]
            r[lab] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
            r[lab]["SharpeRange"] = (float(np.nanmin([m["Sharpe"] for m in ms])), float(np.nanmax([m["Sharpe"] for m in ms])))
        for bn, ns in bench.items():
            r[f"{bn} buy&hold"] = {**period_metrics(ns, a, b), "SharpeRange": None}
        rows[pname] = r
    states = R.regime_states(P.close[config.BENCHMARK])
    st = states[states.index >= START]
    st_cnt = (st.value_counts() / len(st)).round(3).to_dict()
    res = {"periods": rows, "state_share": st_cnt, "seconds": time.time() - t0}
    json.dump(res, open(os.path.join(OUT, "long_2018_2026.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "long_2018_2026.md"), "w").write(to_md(rows, st_cnt, P))
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return res


def to_md(rows: dict, st_cnt: dict, P: E.Prep) -> str:
    out = ["# Long stock backtest 2018-01 → 2026-10 (stress sub-periods)", "",
           "Same rule-based sleeve as `summary.md` (strategies, ≥3-votes gate, vol-managed sizing, real `plan_orders`, 5 trades/day, stops, next-open fills, zero cost), "
           f"mean of {len(OFFSETS)} staggered starts. Return / Sharpe / max drawdown are measured inside each window on the same continuously running fund.", "",
           "**Survivorship bias:** the universe is today's S&P 500 constituents. Names removed from the index since 2018 (mostly losers, plus bankruptcies and "
           "takeovers) are absent and names added after rallying are present, so every long-only number here is flattered, momentum most of all. The stress "
           "windows are the least reliable part: the stocks that crashed hardest in 2018Q4/2020/2022 and never came back are missing. "
           "Earnings history (PEAD, earnings premium) is yfinance's up-to-100-quarter record; the revisions, quality, insider and news signals are not backtestable.", ""]
    for pname, r in rows.items():
        t = pd.DataFrame({k: {"Return": pct(v["Return"]), "Sharpe": num(v["Sharpe"]) + (f" ({num(v['SharpeRange'][0])}–{num(v['SharpeRange'][1])})" if v["SharpeRange"] else ""),
                              "Max DD": pct(v["MaxDD"])} for k, v in r.items()}).T
        t.index.name = pname
        out += [f"## {pname}", "", md_table(t), ""]
    out += [f"SPY regime-state share of days since 2018: {st_cnt}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
