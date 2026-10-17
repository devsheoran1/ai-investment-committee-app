"""
Screening funnel (IPS Part B). Pure Python, no LLM: ~513 names -> shortlist.

    1. Hard filters   price, 20d dollar volume, 1y history, 3m vol (stocks)  (B3-B8)
    2. Ranking        momentum-led factor blend, z-scored within sector     (B9-B10)
    3. Shortlist      top N stocks + all current holdings + eligible ETFs    (B11)

Brief §6: never send the model a big blob. The agents only see the shortlist.
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
        rows[t] = {
            "price": float(p.iloc[-1]),
            "adv20_usd": float((p * v).tail(20).mean()),
            "history_days": int(len(p)),
            "vol_3m": float(r.tail(63).std() * np.sqrt(252)),
            "mom_12_1": float(p.iloc[-21] / p.iloc[-252] - 1) if len(p) > 252 else np.nan,
            "mom_3m": float(p.iloc[-1] / p.iloc[-63] - 1) if len(p) > 63 else np.nan,
            "trend_200dma": float(p.iloc[-1] / p.tail(200).mean() - 1) if len(p) >= 200 else np.nan,
        }
    return pd.DataFrame(rows).T


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def screen(universe: pd.DataFrame, market: dict, holdings: list[str] | None = None,
           regime: str = "neutral") -> dict:
    """Returns {'shortlist': [...], 'table': DataFrame, 'funnel': {...counts}, 'excluded': {ticker: reason}}."""
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
    W = S["factor_weights"]
    raw = {"mom_12_1": stocks["mom_12_1"], "mom_3m": stocks["mom_3m"],
           "trend_200dma": stocks["trend_200dma"], "low_vol": -stocks["vol_3m"]}
    score = pd.Series(0.0, index=stocks.index)
    for k, w in W.items():
        x = raw[k].astype(float).fillna(raw[k].median())
        z = x.groupby(stocks["sector"]).transform(_z) if S["sector_neutral"] else _z(x)
        score += w * z.clip(-3, 3)
    stocks["screen_score"] = score.round(3)
    stocks = stocks.sort_values("screen_score", ascending=False)
    stocks["screen_rank"] = np.arange(1, len(stocks) + 1)

    top = stocks.index[:S["shortlist_size"]].tolist()
    lev = f.index[f["asset_type"] == "leveraged_etf"].tolist()
    inv = f.index[f["asset_type"] == "inverse_etf"].tolist()
    # Inverse ETFs only ever enter in risk-off (IPS A11); held ones stay visible so they can be exited
    etfs = lev + (inv if regime == "risk_off" else [])
    held_ok = [h for h in holdings if h in close.columns]
    shortlist = list(dict.fromkeys(top + held_ok + etfs))

    table = pd.concat([stocks, f[f["asset_type"] != "stock"]])
    return {
        "shortlist": shortlist,
        "table": table,
        "excluded": excluded,
        "funnel": {"universe": int(len(universe)), "with_price_data": int(n0),
                   "passed_hard_filters": int(n_filtered), "stocks_ranked": int(len(stocks)),
                   "shortlist_stocks": len(top), "holdings_added": len([h for h in held_ok if h not in top]),
                   "etfs_added": len([e for e in etfs if e not in top]), "shortlist_total": len(shortlist)},
    }
