"""
Performance attribution vs SPY (Stage 7, Oct 20): daily Brinson-Fachler, linked over the period with Carino smoothing.

Per day t (start-of-day holdings, close-to-close returns):
    Sector calls  (allocation)  = sum_s (w_fund,s - w_SPY,s) x (R_sectorETF,s - R_SPY)
    Stock picking (selection)   = sum_s  w_fund,s x (R_fund stocks in s - R_sectorETF,s)       (incl. interaction)
    Leverage sleeve             = sum_e  w_e x (R_e - R_SPY), split into
                                    beta  = w_e x (leverage_e x R_underlying,e - R_SPY)
                                    decay = w_e x (R_e - leverage_e x R_underlying,e)  (decay / tracking residual)
    Cash drag                   = w_cash x (0 - R_SPY)
    Benchmark tracking          = sum_s w_SPY,s x R_sectorETF,s - R_SPY   (sector ETFs vs SPY itself)
These sum EXACTLY to R_fund,holdings - R_SPY. The gap to the NAV return (fills at execution prices, intraday
timing) is reported separately as "execution / timing", so nothing is hidden.

Carino: linked effect = sum_t effect_t x k_t / k, k_t = ln((1+R_f,t)/(1+R_b,t)) / (R_f,t - R_b,t), so the linked
parts sum to the compounded excess return.

SPY sector weights: yfinance funds_data (Morningstar sectors mapped to GICS), fetched once and cached with its as-of date.
Selection is also attributed to the agent whose weighted score drove each position at entry (reflection signals).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

import config

MS_TO_GICS = {"technology": "Information Technology", "healthcare": "Health Care", "financial_services": "Financials",
              "consumer_cyclical": "Consumer Discretionary", "consumer_defensive": "Consumer Staples",
              "communication_services": "Communication Services", "industrials": "Industrials", "energy": "Energy",
              "utilities": "Utilities", "realestate": "Real Estate", "basic_materials": "Materials"}
FALLBACK_SPY_WEIGHTS = {"Information Technology": 0.40, "Financials": 0.11, "Communication Services": 0.10,
                        "Health Care": 0.09, "Consumer Discretionary": 0.09, "Industrials": 0.075,
                        "Consumer Staples": 0.045, "Energy": 0.035, "Utilities": 0.02, "Real Estate": 0.02, "Materials": 0.015}
PARTS = ["sector_calls", "stock_picking", "leverage_beta", "leverage_decay", "cash_drag", "benchmark_tracking"]


def spy_sector_weights() -> tuple[dict[str, float], str]:
    path = os.path.join(config.CACHE_DIR, "spy_sector_weights.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["weights"], d["asof"]
    try:
        import yfinance as yf
        raw = yf.Ticker("SPY").funds_data.sector_weightings
        w = {MS_TO_GICS[k]: float(v) for k, v in raw.items() if k in MS_TO_GICS}
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        asof = str(dt.date.today())
    except Exception:  # noqa: BLE001
        w, asof = FALLBACK_SPY_WEIGHTS, "fallback (approximate, Oct 2026)"
    with open(path, "w") as f:
        json.dump({"weights": w, "asof": asof}, f)
    return w, asof


def day_effects(w_start: dict[str, float], rets: dict[str, float], r_spy: float, sectors: dict[str, str],
                kinds: dict[str, str], spy_w: dict[str, float], sector_ret: dict[str, float],
                underlying_ret: dict[str, float]) -> dict:
    """One day's Brinson-Fachler effects. w_start: start-of-day weights (cash = 1 - sum)."""
    eff = {p: 0.0 for p in PARTS}
    by_sector: dict[str, dict] = {}
    cash = 1 - sum(w_start.values())
    per_name = {}
    for t, w in w_start.items():
        r = rets.get(t, 0.0)
        k = kinds.get(t, "stock")
        if k == "stock":
            s = sectors.get(t, "Unknown")
            d = by_sector.setdefault(s, {"w": 0.0, "wr": 0.0})
            d["w"] += w; d["wr"] += w * r
        else:
            lev = config.LEVERAGE_FACTOR.get(t, 1.0)
            ur = underlying_ret.get(t)
            beta_part = w * ((lev * ur if ur is not None else r) - r_spy)
            eff["leverage_beta"] += beta_part
            eff["leverage_decay"] += w * (r - (lev * ur if ur is not None else r))
    for s in set(by_sector) | set(spy_w):
        wf = by_sector.get(s, {}).get("w", 0.0)
        rf = by_sector[s]["wr"] / wf if wf else 0.0
        rb = sector_ret.get(s, r_spy)
        eff["sector_calls"] += (wf - spy_w.get(s, 0.0)) * (rb - r_spy)
        eff["stock_picking"] += wf * (rf - rb)
    for t, w in w_start.items():
        if kinds.get(t, "stock") == "stock":
            per_name[t] = w * (rets.get(t, 0.0) - sector_ret.get(sectors.get(t, "Unknown"), r_spy))
    eff["cash_drag"] = cash * (0.0 - r_spy)
    eff["benchmark_tracking"] = sum(spy_w.get(s, 0.0) * sector_ret.get(s, r_spy) for s in spy_w) - r_spy
    r_fund = sum(w * rets.get(t, 0.0) for t, w in w_start.items())
    return {"effects": eff, "r_fund": r_fund, "r_spy": r_spy, "per_name": per_name}


def carino(days: list[dict]) -> dict:
    """Link daily effects so they sum to the compounded excess return."""
    Rf = math.prod(1 + d["r_fund"] for d in days) - 1
    Rb = math.prod(1 + d["r_spy"] for d in days) - 1

    def k_of(a, b):
        return (math.log1p(a) - math.log1p(b)) / (a - b) if abs(a - b) > 1e-12 else 1 / (1 + a)
    K = k_of(Rf, Rb)
    linked = {p: 0.0 for p in PARTS}
    names: dict[str, float] = {}
    for d in days:
        kt = k_of(d["r_fund"], d["r_spy"])
        for p in PARTS:
            linked[p] += d["effects"][p] * kt / K
        for t, v in d.get("per_name", {}).items():
            names[t] = names.get(t, 0.0) + v * kt / K
    return {"fund": Rf, "spy": Rb, "excess": Rf - Rb, "parts": linked, "per_name": names}


def holdings_by_day(trades: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[str, int]]:
    """End-of-day share counts from the transaction log."""
    t = trades.sort_values("id")
    out, sh = {}, {}
    for d in dates:
        for r in t[t["date"] == str(d.date())].itertuples():
            sh[r.symbol] = sh.get(r.symbol, 0) + (r.shares if r.action == "BUY" else -r.shares)
        out[d] = {k: v for k, v in sh.items() if v > 0}
    return out


def run(close: pd.DataFrame, sector_px: pd.DataFrame, sectors: dict, kinds: dict, trades: pd.DataFrame,
        snapshots: pd.DataFrame, reflections: pd.DataFrame | None = None) -> dict | None:
    """Daily + cumulative attribution for the fund in the current DB. None before the first full day."""
    if trades is None or trades.empty or snapshots is None or len(snapshots) < 2:
        return None
    from data import SECTOR_ETFS
    spy_w, spy_asof = spy_sector_weights()
    start = pd.Timestamp(trades["date"].min())
    dates = close.index[close.index >= start]
    if len(dates) < 2:
        return None
    held = holdings_by_day(trades, dates)
    nav = snapshots.set_index("date")["total_equity"]
    rets_all = close.pct_change()
    sec_rets = sector_px.pct_change()
    days = []
    for prev, d in zip(dates[:-1], dates[1:]):
        nav_prev = nav.get(str(prev.date()))
        if nav_prev is None:
            continue
        w = {t: s * float(close[t].loc[prev]) / nav_prev for t, s in held[prev].items() if t in close.columns}
        rets = {t: float(rets_all[t].loc[d]) for t in w if pd.notna(rets_all[t].loc[d])}
        r_spy = float(rets_all[config.BENCHMARK].loc[d])
        sector_ret = {s: float(sec_rets[e].loc[d]) for s, e in SECTOR_ETFS.items()
                      if e in sec_rets.columns and d in sec_rets.index and pd.notna(sec_rets[e].loc[d])}
        und = {t: float(rets_all[config.ETF_UNDERLYING[t]].loc[d]) for t in w
               if t in config.ETF_UNDERLYING and config.ETF_UNDERLYING[t] in rets_all.columns}
        e = day_effects(w, rets, r_spy, sectors, kinds, spy_w, sector_ret, und)
        nav_d = nav.get(str(d.date()))
        e["date"] = str(d.date())
        e["r_nav"] = (nav_d / nav_prev - 1) if nav_d else None
        days.append(e)
    if not days:
        return None
    cum = carino(days)
    nav_ret = (nav.iloc[-1] / config.STARTING_CAPITAL - 1)
    spy_ret = cum["spy"]
    cum["execution_timing"] = (nav_ret - spy_ret) - cum["excess"]
    by_agent = _by_agent(cum["per_name"], reflections)
    p = cum["parts"]
    line = (f"{cum['excess']:+.2%} vs SPY = {p['sector_calls'] + p['benchmark_tracking']:+.2%} sector calls, "
            f"{p['stock_picking']:+.2%} selection, {p['leverage_beta'] + p['leverage_decay']:+.2%} leverage "
            f"(beta {p['leverage_beta']:+.2%}, decay {p['leverage_decay']:+.2%}), {p['cash_drag']:+.2%} cash")
    return {"line": line, "cumulative": cum, "daily": [{"date": d["date"], **{k: round(v, 6) for k, v in d["effects"].items()},
                                                         "r_fund": round(d["r_fund"], 6), "r_spy": round(d["r_spy"], 6)}
                                                        for d in days],
            "by_agent": by_agent, "spy_weights_asof": spy_asof,
            "nav_excess": nav_ret - spy_ret}


def _by_agent(per_name: dict[str, float], refl: pd.DataFrame | None) -> dict[str, float]:
    """Selection attributed to the agent with the largest weighted contribution to the name's composite at entry."""
    if refl is None or refl.empty:
        return {}
    W = config.SIGNAL_WEIGHTS
    out: dict[str, float] = {}
    for t, v in per_name.items():
        rows = refl[(refl["ticker"] == t) & (refl["action"] == "BUY")]
        if rows.empty:
            out["unknown"] = out.get("unknown", 0.0) + v
            continue
        sig = json.loads(rows.iloc[0]["signals"] or "{}")
        contrib = {a: abs(W[a] * (sig.get(a) or 0.0)) for a in ("quant", "sentiment", "research")}
        top = max(contrib, key=contrib.get) if any(contrib.values()) else "unknown"
        out[top] = out.get(top, 0.0) + v
    return out
