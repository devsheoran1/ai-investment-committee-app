"""
Dot-com stress test (Oct 24): 9 original SPDR sector ETFs + SPY, 1999-01 -> 2003-12. What it tests: the regime-allocator states, the ensemble's
price strategies at SECTOR level (sector_rotation, ts_trend and friends) and the C1.3 regime / soft-hedge rules, against SPY.
What it cannot test (state this wherever the numbers are quoted):
  * a stock-level 1999 test is impossible: the dot-com losers were delisted and are not in any free feed (pure survivorship), so only ETFs;
  * the 12-month signals need a year of ETF history (SPDRs started Dec 1998), so until ~Dec 1999 the book is equal-weight sectors;
  * C1.3 uses 3 of its 4 signals (VIX > 25, SPY below 50 and 200dma, VIX +30% in a day): HYG/LQD did not exist;
  * SPXU did not exist: the hedge is modelled as -3x the daily SPY return on the hedge weight (no fees, no decay beyond daily compounding);
  * a 9-name cross-section is tiny, nothing here is statistically meaningful beyond "did the rules lose less in a bear market".
    /opt/miniconda3/bin/python3 -m backtest.dotcom
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config
import strategies
from backtest import data_bt
from backtest.run_backtest import md_table, num, pct
from strategies import regime as R
from strategies.ensemble import reweight

OUT = os.path.join(config.BASE_DIR, "reports", "backtest")
SECTOR = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials", "XLK": "Information Technology",
          "XLP": "Consumer Staples", "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary"}
NAMES = ["xs_momentum_12_1", "ts_trend", "high_52w", "residual_momentum", "st_reversal_1w", "low_beta", "sector_rotation"]
START, END = "1999-01-04", "2003-12-31"
TOP = 3
SOFT_H, OFF_H = 0.05, 0.15            # SPXU-proxy weight with one signal / in full risk-off (midpoint of 0-10%, IPS A11 10-15%)


def load() -> tuple[dict, pd.Series]:
    pkl = os.path.join(data_bt.CACHE, "dotcom_full.pkl")
    if os.path.exists(pkl):
        return pd.read_pickle(pkl)
    import yfinance as yf
    f = data_bt._dl(list(SECTOR) + ["SPY"], "1996-06-01", "2004-12-31")
    vix = yf.download("^VIX", start="1996-06-01", end="2004-12-31", progress=False, auto_adjust=True)["Close"].squeeze()
    pd.to_pickle((f, vix), pkl)
    return f, vix


def c13_signals(spy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    vix = vix.reindex(spy.index).ffill()
    return pd.DataFrame({"VIX > 25": vix > 25,
                         "SPY below 50dma and 200dma": (spy < spy.rolling(50).mean()) & (spy < spy.rolling(200).mean()),
                         "VIX up > 30% in one day": vix.pct_change() > 0.30})


def build():
    f, vix = load()
    close, opn, vol = [f[k].sort_index().dropna(subset=["SPY"]) for k in ("close", "open", "volume")]
    uni = pd.DataFrame({"symbol": list(SECTOR), "sector": list(SECTOR.values()), "industry": np.nan})
    at = {t: "stock" for t in SECTOR} | {"SPY": "benchmark"}
    d = strategies.build_data(close, vol, pd.concat([uni, pd.DataFrame({"symbol": ["SPY"], "sector": ["Benchmark"], "industry": [np.nan]})]), at)
    strats = strategies.registry(NAMES)
    z = {s.name: s.z(d).reindex(columns=list(SECTOR)) for s in strats}
    return close, opn, z, vix


def target_weights(close, z, vix, *, allocator: bool, hedge: bool, top: int = TOP) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    spy = close["SPY"]
    states = R.regime_states(spy)
    sig = c13_signals(spy, vix)
    n_sig = sig.sum(axis=1)
    idx = close.index
    wk = pd.Series(idx.isocalendar().week.astype(int).values, index=idx)
    dec = wk.ne(wk.shift(-1)).astype(bool)
    cols = list(SECTOR) + ["HEDGE"]
    rows = {}
    base = {k: 1.0 / len(NAMES) for k in NAMES}
    hist = {k: True for k in NAMES}
    for i, day in enumerate(idx):
        if not dec.iat[i] or day < pd.Timestamp(START) - pd.Timedelta(days=7):
            continue
        w = base if not allocator else R.apply_state(base, states.iat[i], hist, {"weight_cap": 0.30})
        num_ = den = 0.0
        zz = pd.Series(0.0, index=list(SECTOR)); dd = pd.Series(0.0, index=list(SECTOR))
        for k in NAMES:
            v = z[k].iloc[i]
            zz += w[k] * v.fillna(0.0); dd += w[k] * v.notna()
        ens = (zz / dd.replace(0, np.nan)).dropna()
        if len(ens) < top:
            sel = pd.Series(1.0 / len(SECTOR), index=list(SECTOR))                          # warm-up: equal-weight sectors
        else:
            sel = pd.Series(0.0, index=list(SECTOR)); sel[ens.nlargest(top).index] = 1.0 / top
        h = 0.0
        if hedge:
            h = OFF_H if n_sig.iat[i] >= 2 else (SOFT_H if n_sig.iat[i] == 1 else 0.0)
        rows[day] = pd.concat([sel * (1 - h), pd.Series({"HEDGE": h})])
    W = pd.DataFrame(rows).T.reindex(columns=cols)
    return W, states, n_sig


def run_weights(W: pd.DataFrame, close, opn) -> pd.Series:
    """Weights decided at close t are filled at the open of t+1. Day return = old weights over the gap + new weights open->close."""
    idx = close.index
    spy_gap = opn["SPY"] / close["SPY"].shift(1) - 1
    spy_day = close["SPY"] / opn["SPY"] - 1
    spy_cc = close["SPY"].pct_change()
    gap = (opn / close.shift(1) - 1).assign(HEDGE=-3 * spy_gap)[W.columns]
    day = (close / opn - 1).assign(HEDGE=-3 * spy_day)[W.columns]
    cc = close.pct_change().assign(HEDGE=-3 * spy_cc)[W.columns]
    eff = W.reindex(idx).ffill().shift(1)                       # weights in force at the start of each day
    new = W.reindex(idx).ffill().shift(2)                       # ... and the ones before the latest fill
    changed = eff.ne(new).any(axis=1)
    r = pd.Series(0.0, index=idx)
    r[~changed] = (eff * cc).sum(axis=1)[~changed]
    r[changed] = ((new * gap).sum(axis=1) + (eff * day).sum(axis=1))[changed]
    first = W.index[0]
    r[idx <= first] = 0.0
    return r.fillna(0.0)


def stats(r: pd.Series) -> dict:
    nav = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return {"CAGR": float(nav.iloc[-1] ** (1 / yrs) - 1), "Vol": float(r.std() * np.sqrt(252)), "Sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "MaxDD": float((nav / nav.cummax() - 1).min()), "Total": float(nav.iloc[-1] - 1)}


def main() -> dict:
    close, opn, z, vix = build()
    out = {}
    series = {}
    for lab, kw in (("S1 sector ETFs: ensemble top-3, no allocator", dict(allocator=False, hedge=False)),
                    ("S2 + regime allocator", dict(allocator=True, hedge=False)),
                    ("S3 + C1.3 soft hedge (no allocator)", dict(allocator=False, hedge=True)),
                    ("S4 + allocator + C1.3 soft hedge", dict(allocator=True, hedge=True))):
        W, states, n_sig = target_weights(close, z, vix, **kw)
        series[lab] = run_weights(W, close, opn)
    ew = pd.DataFrame(1.0 / len(SECTOR), index=series[next(iter(series))].index, columns=list(SECTOR))
    series["EW9 sector ETFs (rebalanced weekly)"] = run_weights(ew.assign(HEDGE=0.0).iloc[::5], close, opn)
    series["SPY buy & hold"] = close["SPY"].pct_change().fillna(0.0)
    idx = series["S1 sector ETFs: ensemble top-3, no allocator"].index
    windows = {"1999-2003": (START, END), "2000-03 (signals live)": ("2000-01-03", END), "Mar 2000-Oct 2002 (peak to trough)": ("2000-03-24", "2002-10-09")}
    tbl = {}
    for wname, (a, b) in windows.items():
        tbl[wname] = {lab: stats(s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]) for lab, s in series.items()}
    years = pd.DataFrame({lab: {y: float((1 + s[s.index.year == y]).prod() - 1) for y in range(1999, 2004)} for lab, s in series.items()}).T
    st = states[(states.index >= START) & (states.index <= END)]
    share = (st.value_counts() / len(st)).round(3).to_dict()
    sg = c13_signals(close["SPY"], vix)
    sg = sg[(sg.index >= START) & (sg.index <= END)]
    sig_share = {"0 signals": float((sg.sum(axis=1) == 0).mean()), "1 signal (soft-hedge window)": float((sg.sum(axis=1) == 1).mean()),
                 ">=2 signals (risk-off)": float((sg.sum(axis=1) >= 2).mean())}
    json.dump({"windows": tbl, "years": years.to_dict(), "regime_state_share": share, "c13_share": sig_share},
              open(os.path.join(OUT, "dotcom_1999_2003.json"), "w"), indent=1)
    open(os.path.join(OUT, "dotcom_1999_2003.md"), "w").write(to_md(tbl, years, share, sig_share))
    print(open(os.path.join(OUT, "dotcom_1999_2003.md")).read())
    return tbl


def to_md(tbl, years, share, sig_share) -> str:
    out = ["# Dot-com stress test, 9 SPDR sector ETFs, 1999-01 → 2003-12", "", __doc__.split("What it tests:")[1].split("/opt/")[0].strip().join(["**What it tests:** ", ""]), ""]
    for wname, d in tbl.items():
        t = pd.DataFrame({lab: {"CAGR": pct(v["CAGR"]), "Vol": pct(v["Vol"]), "Sharpe": num(v["Sharpe"]), "Max DD": pct(v["MaxDD"]), "Total": pct(v["Total"])}
                          for lab, v in d.items()}).T
        t.index.name = wname
        out += [f"## {wname}", "", md_table(t), ""]
    y = years.copy()
    y = y.apply(lambda c: c.map(lambda v: pct(v)))
    y.columns = [str(c) for c in y.columns]
    y.index.name = "Calendar-year return"
    out += ["## Calendar years", "", md_table(y), "",
            f"SPY regime-allocator state share of days: {share}. C1.3 signal share (3 of 4 signals): {sig_share}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
