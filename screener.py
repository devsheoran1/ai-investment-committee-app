"""
Screening funnel (IPS Part B, two stages as decided Oct 20). Pure Python, no LLM.

    Hard filters   price, 20d dollar volume, 1y history, 3m vol (stocks)                       (B3-B8)
    Stage 1        all ~500 stocks, price only, z-scored WITHIN SECTOR (B10):
                   12-1m momentum / 12m vol 39% · industry momentum (1m, GICS sub-industry peers) 24% ·
                   trend (price vs 50dma and 200dma) 25% · 1-week reversal 12%                -> top 60
    Stage 2        those 60: Stage-1 score 57% · earnings revisions + post-earnings drift 27% ·
                   quality (ROE, net margin, D/E vs sector) 16%                               -> top 30
                   A missing Stage-2 input reweights pro rata, never counts as 0.
    Shortlist      top 30 + current holdings + eligible leveraged ETFs (+ inverse in risk-off)   (B11)

Effective weights: 12-1m 22 · industry 14 · trend 14 · revisions+drift 27 · quality 16 · reversal 7 · value 0 ·
low-vol 0 (vol-scaling replaces it). All weights live in config.SCREEN_STAGE1/2.
Brief §6: the agents only ever see the shortlist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _factors(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for t in close.columns:
        p = close[t].dropna()
        if len(p) < 30:
            continue
        r = np.log(p).diff().dropna()
        v = volume[t].reindex(p.index).fillna(0) if t in volume.columns else pd.Series(0, index=p.index)
        vol12 = float(r.tail(252).std() * np.sqrt(252)) if len(r) > 60 else np.nan
        mom = float(p.iloc[-21] / p.iloc[-252] - 1) if len(p) > 252 else np.nan
        rows[t] = {
            "price": float(p.iloc[-1]),
            "adv20_usd": float((p * v).tail(20).mean()),
            "history_days": int(len(p)),
            "vol_3m": float(r.tail(63).std() * np.sqrt(252)),
            "mom_12_1": mom,
            "mom_12_1_volscaled": mom / vol12 if vol12 and vol12 > 0 and not np.isnan(mom) else np.nan,
            "ret_1m": float(p.iloc[-1] / p.iloc[-22] - 1) if len(p) > 22 else np.nan,
            "trend": float(np.nanmean([p.iloc[-1] / p.tail(50).mean() - 1,
                                       p.iloc[-1] / p.tail(200).mean() - 1 if len(p) >= 200 else np.nan])),
            "reversal_1w": -float(p.iloc[-1] / p.iloc[-6] - 1) if len(p) > 6 else np.nan,   # short-term reversal
        }
    return pd.DataFrame(rows).T


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def _blend(df: pd.DataFrame, weights: dict[str, float], by_sector: pd.Series | None) -> pd.Series:
    """Weighted z-score blend. Missing inputs for a name reweight the rest pro rata (never count as 0)."""
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for k, w in weights.items():
        x = df[k].astype(float)
        z = x.groupby(by_sector).transform(_z) if by_sector is not None else _z(x)
        z = z.clip(-3, 3)
        ok = z.notna()
        num[ok] += w * z[ok]
        den[ok] += w
    return (num / den.replace(0, np.nan)).fillna(-9)


def industry_momentum(f: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """1m return of the stock's GICS sub-industry peers (equal weight, excluding itself); sector peers if < 2."""
    out = {}
    ind = meta.reindex(f.index)["industry"] if "industry" in meta.columns else pd.Series(index=f.index, dtype=object)
    sec = meta.reindex(f.index)["sector"]
    r1m = f["ret_1m"]
    for t in f.index:
        peers = [u for u in f.index[ind == ind.get(t)] if u != t] if pd.notna(ind.get(t)) else []
        if len(peers) < 2:
            peers = [u for u in f.index[sec == sec.get(t)] if u != t]
        out[t] = float(r1m[peers].mean()) if peers else np.nan
    return pd.Series(out)


def screen(universe: pd.DataFrame, market: dict, holdings: list[str] | None = None,
           regime: str = "neutral", stage2_inputs=None) -> dict:
    """stage2_inputs(tickers) -> {ticker: {"revisions": float|None, "drift": float|None, "quality": float|None}}.
    Returns {'shortlist', 'table', 'funnel', 'excluded'}."""
    S = config.SCREEN
    holdings = list(holdings or [])
    close, volume = market["close"], market["volume"]
    f = _factors(close, volume)
    meta = universe.set_index("symbol")
    f = f.join(meta[["name", "sector", "asset_type"]], how="inner")
    excluded: dict[str, str] = {}

    def drop(mask, reason):
        nonlocal f
        for t in f.index[mask]:
            excluded[t] = reason
        f = f[~mask]

    n0 = len(f)
    drop(f["price"] < S["min_price"], f"price < ${S['min_price']}")
    drop(f["adv20_usd"] < S["min_adv_usd"], f"20d $ volume < ${S['min_adv_usd'] / 1e6:.0f}m")
    drop(f["history_days"] < S["min_history_days"], "less than 1y of price history")
    drop((f["asset_type"] == "stock") & (f["vol_3m"] > S["max_vol_3m_stocks"]),
         f"3m realised vol > {S['max_vol_3m_stocks']:.0%}")
    n_filtered = len(f)

    stocks = f[f["asset_type"] == "stock"].copy()
    stocks["industry_mom_1m"] = industry_momentum(stocks, meta)
    sec = stocks["sector"] if S["sector_neutral"] else None
    stocks["stage1_score"] = _blend(stocks, config.SCREEN_STAGE1["weights"], sec).round(3)
    stocks = stocks.sort_values("stage1_score", ascending=False)
    stocks["stage1_rank"] = np.arange(1, len(stocks) + 1)
    top60 = stocks.index[:config.SCREEN_STAGE1["keep"]].tolist()

    st2 = stocks.loc[top60].copy()
    inputs = {}
    if stage2_inputs is not None:
        try:
            inputs = stage2_inputs(top60) or {}
        except Exception:  # noqa: BLE001  stage 2 is an overlay: never kill the funnel
            inputs = {}
    st2["revisions_drift"] = [_rev_drift(inputs.get(t) or {}) for t in st2.index]
    st2["quality"] = [(inputs.get(t) or {}).get("quality") for t in st2.index]
    st2["stage1"] = st2["stage1_score"]
    st2["screen_score"] = _blend(st2, config.SCREEN_STAGE2["weights"], st2["sector"] if S["sector_neutral"] else None).round(3)
    st2 = st2.sort_values("screen_score", ascending=False)
    st2["screen_rank"] = np.arange(1, len(st2) + 1)
    top = st2.index[:config.SCREEN_STAGE2["keep"]].tolist()

    lev = f.index[f["asset_type"] == "leveraged_etf"].tolist()
    inv = f.index[f["asset_type"] == "inverse_etf"].tolist()
    etfs = lev + (inv if regime == "risk_off" else [])   # inverse only in risk-off (IPS A11)
    held_ok = [h for h in holdings if h in close.columns]
    shortlist = list(dict.fromkeys(top + held_ok + etfs))

    table = pd.concat([st2, stocks.drop(index=top60), f[f["asset_type"] != "stock"]])
    return {
        "shortlist": shortlist,
        "table": table,
        "excluded": excluded,
        "stage2_inputs": {t: inputs.get(t) for t in top60},
        "funnel": {"universe": int(len(universe)), "with_price_data": int(n0),
                   "passed_hard_filters": int(n_filtered), "stocks_ranked": int(len(stocks)),
                   "stage1_kept": len(top60), "stage2_with_inputs": int(sum(1 for t in top60 if inputs.get(t))),
                   "shortlist_stocks": len(top), "holdings_added": len([h for h in held_ok if h not in top]),
                   "etfs_added": len([e for e in etfs if e not in top]), "shortlist_total": len(shortlist)},
    }


def _rev_drift(d: dict):
    """Earnings revisions (30d change in current-quarter and next-year EPS estimates) + post-earnings drift,
    averaged over what is available; None if neither exists."""
    vals = [v for v in (d.get("revisions"), d.get("drift")) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan
