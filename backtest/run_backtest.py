"""
Backtest + Monte Carlo runner (Oct 23). Files only: reports/backtest/summary.md, *.csv, *.png. No UI, no LLM.

    /opt/miniconda3/bin/python3 -m backtest.run_backtest            # needs `python -m backtest.data_bt` once
"""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from backtest import data_bt, engine as E
from strategies import regime as R

OUT = os.path.join(config.BASE_DIR, "reports", "backtest")
START = "2023-10-23"
NAVY, BLUE, GREY, RED, GREEN = "#14213D", "#3B82F6", "#9CA3AF", "#DC2626", "#16A34A"


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for k, r in df.iterrows():
        lines.append("| " + " | ".join([str(k)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


OFFSETS = [0, 7, 14, 21, 28, 35]          # six staggered start dates: same strategy, different first week -> the noise floor


def start_of(o: int) -> str:
    return str((pd.Timestamp(START) + pd.Timedelta(days=o)).date())


def mean_metrics(ms: list[dict]) -> dict:
    keys = ms[0].keys()
    out = {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
    out["SharpeMin"], out["SharpeMax"] = float(np.nanmin([m["Sharpe"] for m in ms])), float(np.nanmax([m["Sharpe"] for m in ms]))
    return out


def row(m: dict) -> dict:
    sh = num(m["Sharpe"]) + (f" ({num(m['SharpeMin'])}–{num(m['SharpeMax'])})" if "SharpeMin" in m else "")
    return {"CAGR": pct(m["CAGR"]), "Vol": pct(m["Vol"]), "Sharpe (range over 6 starts)": sh, "Max DD": pct(m["MaxDD"]),
            "Turnover/yr": "–" if m.get("Turnover/yr") is None else f"{m['Turnover/yr']:.1f}x"}


def main(quick: bool = False) -> dict:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    px = data_bt.load_prices()
    uni = data_bt.universe_frame()
    stocks = uni.loc[uni.asset_type == "stock", "symbol"].tolist()
    earn = data_bt.load_earnings(stocks, budget_s=1)            # cached by data_bt; no network here
    P = E.prepare(px, uni, earn)
    end = str(P.close.index[-1].date())

    OLD = {"weight_mode": "ic_legacy"}                       # Oct 23 rule: A-H and P1-P8 stay exactly the old study
    W = E.weight_history(P, cfg=OLD)
    ens, votes, alloc = E.ensemble_panels(P, W)
    dep = E.as_deployed(P, ens, alloc, blend=0.75)
    Weq = E.weight_history(P, equal=True)
    ens_eq, votes_eq, alloc_eq = E.ensemble_panels(P, Weq)

    runs: dict[str, dict] = {}

    def sim(label, score, v, **kw):
        kw.setdefault("trail_pct", -0.08)                    # old group: the Oct 23 -8% trailing stop
        res = [E.simulate(P, score, v, start=start_of(o), **kw) for o in OFFSETS]
        runs[label] = {**res[0], "ms": [E.metrics(r["nav"], r["turnover_one_way"]) for r in res]}
        return runs[label]

    sim("A. Quant as deployed (legacy + ensemble), weekly", dep, votes)
    sim("B. Ensemble only (IC-weighted)", ens, votes)
    sim("C. Ensemble only, equal-weight strategies", ens_eq, votes_eq)
    sim("D. Legacy technical score only (no ensemble)", P.legacy, None, vote_gate=False)
    sim("E. A, daily rebalance", dep, votes, freq="D")
    sim("F. A without the >=3-votes gate", dep, votes, vote_gate=False)
    sim("G. A without the vol-managed overlay", dep, votes, vol_managed=False)
    sim("H. A with 5 bp cost per side", dep, votes, cost_bps=5.0)
    # profit-taking / stop ablation (all on A)
    sim("P1. A: trim pairing ON,  trailing stop ON", dep, votes)
    sim("P2. A: trim pairing OFF, trailing stop ON", dep, votes, pair_trims=False)
    sim("P3. A: trim pairing ON,  trailing stop OFF", dep, votes, trailing=False)
    sim("P4. A: trim pairing OFF, trailing stop OFF (−10% stop only)", dep, votes, pair_trims=False, trailing=False)
    sim("P5. A: daily, trim pairing ON,  trailing ON", dep, votes, freq="D")
    sim("P6. A: daily, trim pairing OFF, trailing ON", dep, votes, freq="D", pair_trims=False)
    sim("P7. A: daily, trim pairing ON,  trailing OFF", dep, votes, freq="D", trailing=False)
    sim("P8. A: daily, trim pairing OFF, trailing OFF", dep, votes, freq="D", pair_trims=False, trailing=False)

    # ---- Oct 24 settings: shrinkage weights, Quant = 100% ensemble, -12% trailing stop; with and without the regime allocator
    NEW = {"weight_mode": "shrinkage"}
    W_new = E.weight_history(P, cfg=NEW)
    W_reg = E.weight_history(P, cfg=NEW, regime=True)
    s_new, v_new, a_new = E.ensemble_panels(P, W_new)
    s_reg, v_reg, a_reg = E.ensemble_panels(P, W_reg)
    new_dep = E.as_deployed(P, s_new, a_new, blend=1.0)
    reg_dep = E.as_deployed(P, s_reg, a_reg, blend=1.0)
    NEWKW = {"trail_pct": config.TRAILING_STOP}
    sim("N0. OLD deployed (Oct 23: IC weights, blend 0.75, -8% trail)", dep, votes)
    sim("N1. NEW (shrinkage, 100% ensemble, -12% trail), no regime allocator", new_dep, v_new, **NEWKW)
    sim("N2. NEW + regime allocator", reg_dep, v_reg, **NEWKW)
    states = R.regime_states(P.close[config.BENCHMARK])

    bench = {b: E.buy_hold(P.close, P.open, b, START) for b in data_bt.BENCH}
    head = {lab: mean_metrics(r["ms"]) for lab, r in runs.items()}
    bm = {b: mean_metrics([E.metrics(E.buy_hold(P.close, P.open, b, start_of(o))) for o in OFFSETS]) for b in data_bt.BENCH}

    # ---- drop-one ablation (on the ensemble-only variant B, so the effect is not diluted by the legacy score)
    base_B = head["B. Ensemble only (IC-weighted)"]
    drop = {}
    for n in P.names:
        Wd = E.weight_history(P, exclude=(n,))
        s_, v_, a_ = E.ensemble_panels(P, Wd)
        m = mean_metrics([E.metrics((r := E.simulate(P, s_, v_, start=start_of(o)))["nav"], r["turnover_one_way"]) for o in OFFSETS])
        drop[n] = {"Sharpe": m["Sharpe"], "dSharpe": m["Sharpe"] - base_B["Sharpe"], "CAGR": m["CAGR"], "MaxDD": m["MaxDD"]}

    # ---- standalone strategies
    stand = {n: E.standalone(P, n, start=START) for n in P.names}
    etf = E.etf_filter_standalone(P, START)
    tom = E.tom_study(P, START)

    # ---- Monte Carlo on A (and the legacy score for comparison), vs SPY
    a_ret = runs["A. Quant as deployed (legacy + ensemble), weekly"]["nav"].pct_change().dropna()
    d_ret = runs["D. Legacy technical score only (no ensemble)"]["nav"].pct_change().dropna()
    spy_ret = bench["SPY"].pct_change().dropna()
    mc = {"A. Quant as deployed": E.monte_carlo(a_ret, spy_ret), "D. Legacy only": E.monte_carlo(d_ret, spy_ret),
          "SPY alone (vs itself, for scale)": E.monte_carlo(spy_ret, spy_ret)}

    # ---- files
    tbl = pd.DataFrame({**{lab: row(m) for lab, m in head.items()}, **{f"{b} (buy & hold)": row(m) for b, m in bm.items()}}).T
    tbl.index.name = "Variant"
    tbl.to_csv(os.path.join(OUT, "variants.csv"))
    pd.DataFrame(stand).T.to_csv(os.path.join(OUT, "strategies_standalone.csv"))
    pd.DataFrame(drop).T.to_csv(os.path.join(OUT, "ablation_drop_one.csv"))
    W.loc[START:].iloc[::5].to_csv(os.path.join(OUT, "weights_weekly.csv"))
    pd.DataFrame({k: v["nav"] for k, v in runs.items()}).to_csv(os.path.join(OUT, "nav_series.csv"))
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in mc.items()},
              open(os.path.join(OUT, "montecarlo.json"), "w"), indent=1)

    wk = states[states.index >= START]
    last = pd.Series(wk.index.isocalendar().week.values, index=wk.index)
    wk = wk[last.ne(last.shift(-1))]
    wk.rename("state").to_csv(os.path.join(OUT, "regime_weekly.csv"))
    W_reg.loc[START:].iloc[::5].to_csv(os.path.join(OUT, "weights_weekly_regime.csv"))
    charts(runs, bench, stand, drop, mc, W)
    md = summary_md(P, W, tbl, head, bm, runs, drop, stand, etf, tom, mc, end, time.time() - t0)
    md = md.replace("## 1. Variants vs benchmarks", new_section(P, head, bm, runs, states, wk) + "\n## 1. Variants vs benchmarks", 1)
    open(os.path.join(OUT, "summary.md"), "w").write(md)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}/summary.md")
    return {"head": head, "bench": bm, "drop": drop, "stand": stand, "mc": mc, "runs": runs}


def new_section(P, head, bm, runs, states, wk) -> str:
    keys = [k for k in head if k.startswith("N")]
    cmp_ = pd.DataFrame({**{k: row(head[k]) for k in keys}, **{f"{b} (buy & hold)": row(bm[b]) for b in ("SPY", "RSP", "MTUM")}}).T
    cmp_.index.name = "Variant"
    # which state was each week in, as consecutive runs
    runs_md, start, prev, n = [], None, None, 0
    for d, st in wk.items():
        if st != prev:
            if prev is not None:
                runs_md.append((start, last_d, prev, n))
            start, prev, n = d, st, 0
        n += 1
        last_d = d
    runs_md.append((start, last_d, prev, n))
    counts = wk.value_counts()
    cnt = ", ".join(f"{k} {counts.get(k, 0)} wks ({counts.get(k, 0) / len(wk):.0%})" for k in R.STATES)
    wr = pd.DataFrame({"weeks": [int(counts.get(k, 0)) for k in R.STATES]}, index=R.STATES)
    for k in keys:
        nav = runs[k]["nav"]
        wret = nav.reindex(wk.index, method="ffill").pct_change().shift(-1)         # return over the week AFTER the state was read
        wr[k[:2] + " avg weekly return"] = [pct(wret[wk == st].mean(), 2) if (wk == st).any() else "n/a" for st in R.STATES]
    wr.index.name = "State at Friday close"
    rl = "\n".join(f"- {a.date()} → {b.date()}: **{st}** ({n} wk)" for a, b, st, n in runs_md)
    return f"""## 0. Old vs new settings (Oct 24)

N0 = the Oct 23 deployment (IC-weighted strategies, Quant 75% ensemble, −8% trailing stop). N1 = the new settings: shrinkage weights
(equal prior + λ·max(t, 0), never all-zero, PEAD cap 30%), Quant = 100% ensemble, −12% trailing stop. N2 = N1 plus the regime allocator
(family weights × SPY-state multipliers, renormalised). Same six staggered starts, same rule-based sleeve, next-open fills, zero cost.

{md_table(cmp_)}

**These settings were chosen after the first backtest, on the same 2023–26 window: they are in-sample, not significant, and the Sharpe
range over the six starts is the noise floor. Do not read N1/N2 > N0 as skill.** The 15-day live race cannot tell them apart either.

Regime state per week (read at Friday close; {len(wk)} weeks; {cnt}). Week-by-week CSV: `regime_weekly.csv`.

{md_table(wr)}

{rl}

"""


# ------------------------------------------------------------------------------------------------- charts
def charts(runs, bench, stand, drop, mc, W):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 130})
    A = runs["A. Quant as deployed (legacy + ensemble), weekly"]["nav"]
    D = runs["D. Legacy technical score only (no ensemble)"]["nav"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    for lab, s, c, lw in (("Quant as deployed (A)", A, NAVY, 2.0), ("Legacy score only (D)", D, BLUE, 1.2),
                          ("SPY", bench["SPY"], GREY, 1.4), ("MTUM", bench["MTUM"], GREEN, 1.0), ("RSP", bench["RSP"], RED, 1.0)):
        s = s[s.index >= A.index[0]]
        ax[0].plot(s.index, s / s.iloc[0] * 100, label=lab, color=c, lw=lw)
        ax[1].plot(s.index, (s / s.cummax() - 1) * 100, color=c, lw=lw)
    ax[0].set_ylabel("NAV (start = 100)"); ax[0].legend(frameon=False, ncol=3); ax[0].set_title("Rule-based Quant sleeve vs benchmarks, next-open fills, zero cost")
    ax[1].set_ylabel("Drawdown %")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_drawdown.png")); plt.close(fig)

    names = list(stand)
    sh = [stand[n]["Sharpe"] for n in names]
    ic_t = [stand[n]["IC_t"] for n in names]
    order = np.argsort(sh)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].barh([names[i] for i in order], [sh[i] for i in order], color=[GREEN if sh[i] > 0 else RED for i in order])
    ax[0].set_title("Standalone Sharpe (top-15 long-only, weekly)")
    order = np.argsort(ic_t)
    ax[1].barh([names[i] for i in order], [ic_t[i] for i in order], color=[BLUE if abs(ic_t[i]) >= 2 else GREY for i in order])
    ax[1].axvline(1, color=NAVY, lw=0.8, ls="--"); ax[1].axvline(-1, color=NAVY, lw=0.8, ls="--")
    ax[1].set_title("Rank-IC t-stat (5d forward, dashed = ±1, the ensemble's cut-off)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "strategies.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    Wm = W.loc[A.index[0]:].resample("ME").last()
    ax.stackplot(Wm.index, *[Wm[c] for c in Wm.columns], labels=Wm.columns)
    ax.set_title("Ensemble weights over time (walk-forward IC; the white space is unallocated weight)")
    ax.legend(fontsize=6, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12)); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "weights.png")); plt.close(fig)

    m = mc["A. Quant as deployed"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(m["_excess"] * 100, bins=80, color=BLUE, alpha=0.85)
    ax.axvline(0, color=NAVY, lw=1); ax.axvline(np.median(m["_excess"]) * 100, color=RED, lw=1, ls="--")
    ax.set_title(f"15-day excess return vs SPY, 10k block-bootstrap paths · P(beat SPY) = {m['P(beat benchmark)']:.0%}")
    ax.set_xlabel("% points")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "montecarlo.png")); plt.close(fig)


def summary_md(P, W, tbl, head, bm, runs, drop, stand, etf, tom, mc, end, secs) -> str:
    A = "A. Quant as deployed (legacy + ensemble), weekly"
    a, spy = head[A], bm["SPY"]
    r = runs[A]
    cut = tbl.loc[[k for k in tbl.index if k[:2] in ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.") or "buy & hold" in k]]
    pt = tbl.loc[[k for k in tbl.index if k.startswith("P")]]
    st = pd.DataFrame({n: {"Standalone Sharpe": num(v["Sharpe"]), "CAGR": pct(v["CAGR"]), "Mean IC": num(v["MeanIC"], 4),
                           "IC t": num(v["IC_t"]), "Turnover/yr": f"{v['Turnover/yr']:.1f}x"} for n, v in stand.items()}).T
    st.index.name = "Strategy"
    dr = pd.DataFrame({n: {"Sharpe w/o it": num(v["Sharpe"]), "Δ Sharpe": f"{v['dSharpe']:+.2f}", "CAGR": pct(v["CAGR"]),
                           "Max DD": pct(v["MaxDD"])} for n, v in drop.items()}).T
    dr.index.name = "Dropped strategy"
    mcr = pd.DataFrame({k: {"P(beat SPY)": pct(v["P(beat benchmark)"], 0), "P(DD ≤ −7%)": pct(v["P(DD <= -7%)"], 0),
                            "P(DD ≤ −10%)": pct(v["P(DD <= -10%)"], 0), "P(loss)": pct(v["P(loss)"], 0),
                            "15d return p5/p50/p95": f"{pct(v['return']['p5'])} / {pct(v['return']['p50'])} / {pct(v['return']['p95'])}",
                            "Excess p5/p50/p95": f"{pct(v['excess']['p5'])} / {pct(v['excess']['p50'])} / {pct(v['excess']['p95'])}"}
                        for k, v in mc.items()}).T
    mcr.index.name = "Path source"
    wlast = W.iloc[-1]
    etf_t = pd.DataFrame({k: {"Sharpe": num(v["Sharpe"]), "CAGR": pct(v["CAGR"]), "Max DD": pct(v["MaxDD"])} for k, v in etf.items()}).T
    etf_t.index.name = "Leveraged-ETF filter"
    beats = a["Sharpe"] > spy["Sharpe"]
    noise = max(h["SharpeMax"] - h["SharpeMin"] for h in head.values() if "SharpeMax" in h)
    a_rng = f"{num(a['SharpeMin'])} to {num(a['SharpeMax'])}"
    b_, d_ = head["B. Ensemble only (IC-weighted)"], head["D. Legacy technical score only (no ensemble)"]
    overlap = max(b_["SharpeMin"], d_["SharpeMin"]) <= min(b_["SharpeMax"], d_["SharpeMax"])
    verdict = (f"Ensemble-only (B) averaged Sharpe {num(b_['Sharpe'])}, the legacy Quant score (D) {num(d_['Sharpe'])}, the as-deployed blend (A) {num(a['Sharpe'])}. "
               + ("The start-date ranges overlap, so **the data cannot separate them**. " if overlap else "The start-date ranges do not overlap, which is the only kind of gap worth noticing. ")
               + ("A sits below B: the legacy half of the blend may be diluting the ensemble, which makes `ENSEMBLE['blend']` the one parameter worth a look "
                  "(not tuned here; tuning it on this window would be curve-fitting)." if a["Sharpe"] < b_["Sharpe"] else ""))
    return f"""# Backtest and Monte Carlo — multi-strategy Quant sleeve (generated {pd.Timestamp.today().date()})

**Window:** {P.close.index[P.close.index >= pd.Timestamp(START)][0].date()} → {end} (3 years, 2 further years of warm-up for the 252-day signals).
**What was tested:** only the rule-based parts of the fund (strategies, walk-forward IC weights, ≥3-votes entry gate, vol-managed
sizing, the real `plan_orders` with the 5-trades/day cap, the A16 drift band, trim pairing, A7/A12, the −10% stop and the −8% trailing stop).
LLM agents are excluded on purpose: a language model has read the future, so any LLM backtest leaks look-ahead.
**Fills:** next open after the signal close · **costs:** 0 (A20) except row H · **risk-free rate:** 0 · stocks only.

## Headline

Every number below is the **mean over six staggered start dates** (the same strategy started in six consecutive weeks), with the Sharpe range
shown. That range is the noise floor: the same rules gave Sharpe {a_rng} for A depending only on which Monday the fund started, and up to
{noise:.2f} wide for other variants. **Differences smaller than that are not evidence of anything.**

Quant as deployed (A) averaged **{pct(a['CAGR'])} CAGR, Sharpe {num(a['Sharpe'])}, max drawdown {pct(a['MaxDD'])}**
against SPY {pct(spy['CAGR'])} / {num(spy['Sharpe'])} / {pct(spy['MaxDD'])}. {"It beat" if beats else "It did **not** beat"} SPY on Sharpe over this window
(first start: {r['trades']} fills, {r['forced_exits']} forced exits, {r['trims']} drift trims; {r['trims_absorbed']} of them already had a new-name buy in the same plan,
{r['paired_buys']} needed a paired buy).
{verdict}
Three years of one mostly bullish regime is far too little to separate skill from luck; read the rankings below as direction, not proof.

![equity](equity_drawdown.png)

## 1. Variants vs benchmarks

{md_table(cut)}

A = the OCTOBER 23 Quant agent before the LLM sees it (`rule_score` = legacy technical score blended with the ensemble by
0.75 × the allocated ensemble weight; IC-weighted strategies, −8% trailing stop). The Oct 24 settings are in section 0. B/C isolate the ensemble. D is the old Quant score. E–H test one design choice at a time.
Turnover is one-way traded value per year ÷ average NAV; MTUM and RSP are buy-and-hold ETFs on the same open-to-close basis.

## 2. Profit-taking without cash and the trailing stop (B1 / B2 ablation, on A)

{md_table(pt)}

Counts in P1 (weekly, first start): {runs['P1. A: trim pairing ON,  trailing stop ON']['trims']} drift trims ({runs['P1. A: trim pairing ON,  trailing stop ON']['trims_absorbed']} already redeployed by the plan's own new-name buys), {runs['P1. A: trim pairing ON,  trailing stop ON']['paired_buys']} paired buys,
{runs['P1. A: trim pairing ON,  trailing stop ON']['forced_exits']} forced exits; P4 (stop only): {runs['P4. A: trim pairing OFF, trailing stop OFF (−10% stop only)']['forced_exits']} forced exits.
The trailing stop forces roughly the extra exits you see in that difference, which is where most of the turnover comes from.

## 3. Each strategy on its own

Long-only top-15 by one strategy's sector-neutral z, equal weight, weekly, next-open fills, no costs. IC is the cross-sectional
Spearman correlation of z with the next 5-day return, daily; the t-stat uses n/5 observations because 5-day windows overlap.
**Treat the standalone CAGRs as inflated:** the universe is today's S&P 500, which contains names admitted *because* they rallied
(survivorship / index-inclusion bias hits momentum hardest). The fully constrained fund simulation in section 1 earns far less.

{md_table(st)}

![strategies](strategies.png)

Not backtestable (no point-in-time history): `eps_revisions`, `quality_gp` (gross-margin/ROA proxy), `insider_buying`, `news_surprise`
(only today's snapshot exists; FinBERT/news history does not). `turn_of_month` is a tilt, tested as a market-timing statistic:
SPY averaged {tom['mean_window_bp']:.1f} bp/day inside the {tom['window_days']}-day window versus {tom['mean_other_bp']:.1f} bp outside (t = {tom['t']:.2f}).
`etf_trend_200dma` is tested on the 14 leveraged long ETFs, equal weight, in/out of the 200dma:

{md_table(etf_t)}

## 4. Ablation: drop one strategy (ensemble-only variant B, base Sharpe {num(head['B. Ensemble only (IC-weighted)']['Sharpe'])})

{md_table(dr)}

With 11 strategies and a 3-year window most Δ Sharpe values are inside the noise of the test; a drop that *raises* Sharpe is a
reason to look, not to delete. Ensemble weights on the last day: {", ".join(f"{k} {v:.0%}" for k, v in wlast.items() if v > 0) or "none cleared t ≥ 1"}.

![weights](weights.png)

## 5. Monte Carlo — 15 trading days, 10,000 block-bootstrap paths

Blocks of 5 days drawn jointly from the fund's and SPY's daily returns (their correlation is kept); 3 blocks per path.
Drawdown is measured inside the 15-day path from its own running peak.

{md_table(mcr)}

![mc](montecarlo.png)

P(beat SPY) near 50% means the sleeve is indistinguishable from the index over a 15-day race, which is the honest prior for the
class competition: with this horizon, the P&L ranking is mostly noise.

## 6. Known biases and limits (state these on the slide)

- **Survivorship:** the universe is today's S&P 500 constituents; names that were dropped from the index (usually losers) are absent, so every long-only number is flattered.
- **Frozen fundamentals / no fundamentals at all:** quality, estimate revisions and the Research agent cannot be replayed point-in-time, so they are not in the backtest. Sector/industry labels are today's GICS.
- **No historical news:** `news_surprise`, the Sentiment agent and the politician-trades context are not backtested; the C4 news veto is not exercised.
- **Earnings data:** EPS surprises come from yfinance's history (reported values, some restatement risk); `earnings_premium` uses the realised report dates, which companies announce weeks ahead (small look-ahead if a date moved).
- **Not in the loop:** LSTM / Kronos / LightGBM, regime rules, leveraged-ETF trading, the mean-variance optimiser (a rule-based weighting stands in), PM/Risk/CIO judgement. This is the Quant sleeve, not the fund.
- **Single regime, 3 years:** one bull-market window, ablations are multiple comparisons on the same data; none of the differences is claimed significant.
- **Design parameters** (blend 0.75 (old group), 20% cap, 60-session IC, −8% trailing stop (old group; −12% in section 0), 15 names, 5-day cooldown after a stop) were fixed in advance, not tuned on this sample.
- **Monte Carlo** resamples this same in-sample history; it states the spread of 15-day outcomes, not a forecast.

_Generated by `backtest/run_backtest.py` in {secs:.0f}s from `data/cache/bt/` (prices + earnings history)._
"""


if __name__ == "__main__":
    main()
