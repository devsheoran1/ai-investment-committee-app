"""Oct 27: neutral beta band, old vs new, through the REAL mean-variance optimiser (analytics.optimise) on the rule-based sleeve.
Every date is treated as the neutral regime so the comparison isolates the band. Same N1 scores/votes/stops/plan_orders as longrun.
    /opt/miniconda3/bin/python3 -m backtest.beta_band
Variants: A hard 0.9-1.2 (live until Oct 27) | B hard 0.5-1.2 (bound only) | C soft penalty, lower bound 0.5 (new default)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import analytics
import config
from backtest import engine as E
from backtest.longrun import PERIODS, OUT, period_metrics, prepare_long
from backtest.run_backtest import md_table, num, pct

VARIANTS = {"A hard 0.9-1.2 (old)": dict(beta_band=(0.9, 1.2), beta_soft=False),
            "B hard 0.5-1.2": dict(beta_band=(0.5, 1.2), beta_soft=False),
            "C soft, lower bound 0.5 (new)": dict(beta_band=(0.5, 1.2), beta_soft=True)}


def make_target_fn(P: E.Prep, score: pd.DataFrame, kw: dict, log: list):
    sector, at = P.sector.to_dict(), P.asset_type.to_dict()

    def fn(i, names, cand, first):
        top = list(dict.fromkeys(list(names) + list(cand.index[:30])))
        conv = {t: float(np.clip(2 * cand[t], -1, 1)) for t in top}
        px = P.close.iloc[:i + 1][[t for t in top if t in P.close.columns] + [config.BENCHMARK]]
        w = analytics.optimise(conv, px, sector, at, 0.98, "neutral", 17.0, config.TARGET_HOLDINGS_INIT if first else 15,
                               min_weight=0.026, ensure_min=first, **kw)
        if w:
            r = np.log(px).diff().dropna().tail(252)
            m = r[config.BENCHMARK]
            log.append(sum(x * np.cov(r[t], m)[0, 1] / m.var() for t, x in w.items() if t in r))
        return w
    return fn


def port_beta(nav: pd.Series, spy: pd.Series) -> float:
    r = pd.concat([nav.pct_change(), spy.reindex(nav.index).pct_change()], axis=1).dropna()
    return float(np.cov(r.iloc[:, 0], r.iloc[:, 1])[0, 1] / r.iloc[:, 1].var())


def main() -> dict:
    t0 = time.time()
    P = prepare_long()
    W = E.weight_history(P, cfg={"weight_mode": "shrinkage"})
    s_, v_, a_ = E.ensemble_panels(P, W)
    score = E.as_deployed(P, s_, a_, blend=1.0)
    runs, ex_beta = {}, {}
    for lab, kw in VARIANTS.items():
        log: list = []
        runs[lab] = E.simulate(P, score, v_, start="2018-01-02", target_fn=make_target_fn(P, score, kw, log))
        ex_beta[lab] = float(np.mean(log)) if log else float("nan")
        print(f"{lab}: done {time.time() - t0:.0f}s", flush=True)
    spy = P.close[config.BENCHMARK]
    rows = []
    for lab, r in runs.items():
        for per, (a, b) in PERIODS.items():
            m = period_metrics(r["nav"], a, b)
            rows.append({"Variant": lab, "Period": per, "Return": pct(m["Return"]), "Sharpe": num(m["Sharpe"]), "MaxDD": pct(m["MaxDD"])})
    summ = [{"Variant": lab, "Realised beta vs SPY": num(port_beta(r["nav"], spy)), "Avg ex-ante beta": num(ex_beta[lab]),
             "Trades": r["trades"], "Turnover/yr": pct(E.metrics(r["nav"], r["turnover_one_way"])["Turnover/yr"])} for lab, r in runs.items()]
    piv = pd.DataFrame(rows).pivot(index="Variant", columns="Period", values=["Sharpe", "MaxDD"])
    md = ["# Neutral beta band: hard 0.9-1.2 vs soft (lower bound 0.5)", "",
          "Rule-based sleeve through the live optimiser (neutral regime on every date), weekly, next-open fills, zero costs, today's S&P 500 "
          "(survivorship-flattered). Not an LLM backtest.", "", md_table(pd.DataFrame(summ)), ""]
    for per in PERIODS:
        sub = pd.DataFrame([r for r in rows if r["Period"] == per]).drop(columns="Period")
        md += [f"**{per}**", "", md_table(sub), ""]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "beta_band.md")
    open(path, "w").write("\n".join(md))
    print("\n".join(md))
    print(f"[{time.time() - t0:.0f}s] -> {path}")
    return {"runs": runs}


if __name__ == "__main__":
    main()
