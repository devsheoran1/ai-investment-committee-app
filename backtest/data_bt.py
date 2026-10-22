"""Backtest inputs (cached under data/cache): 5y of prices (3y window + 2y warm-up for the 12-1m / 252d signals),
benchmarks (SPY, MTUM, RSP) and point-in-time earnings history from yfinance. Network only here; the engine is offline.

    python -m backtest.data_bt          # fetch + cache everything (resumable; Yahoo may rate-limit)
"""
from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config
import data
import prices

CACHE = os.path.join(config.CACHE_DIR, "bt")
os.makedirs(CACHE, exist_ok=True)
BENCH = ["SPY", "MTUM", "RSP"]
PRICES_PKL = os.path.join(CACHE, "prices_5y.pkl")
EARN_PKL = os.path.join(CACHE, "earnings_history.pkl")


def universe_frame() -> pd.DataFrame:
    u = pd.read_csv(config.UNIVERSE_CSV, dtype={"cik": str})
    etf = pd.DataFrame({"symbol": list(config.LEVERAGED_ETFS), "name": list(config.LEVERAGED_ETFS.values()),
                        "sector": "Leveraged ETF", "industry": np.nan, "cik": np.nan, "asset_type": "leveraged_etf"})
    return pd.concat([u[["symbol", "name", "sector", "industry", "cik", "asset_type"]], etf], ignore_index=True)


def load_prices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    if os.path.exists(PRICES_PKL) and not refresh:
        return pd.read_pickle(PRICES_PKL)
    u = universe_frame()
    tickers = list(dict.fromkeys(u["symbol"].tolist() + BENCH))
    fields = prices.download_history(tickers, "5y")
    idx = fields["close"]["SPY"].dropna().index
    out = {k: v.reindex(idx).sort_index() for k, v in fields.items()}
    pd.to_pickle(out, PRICES_PKL)
    return out


def _one_earnings(t: str):
    try:
        import yfinance as yf
        ed = yf.Ticker(t).get_earnings_dates(limit=24)
        if ed is None or not len(ed):
            return t, []
        ed = ed.dropna(subset=["Reported EPS"])
        return t, [{"date": str(i.date()), "surprise_pct": None if pd.isna(r["Surprise(%)"]) else float(r["Surprise(%)"])}
                   for i, r in ed.iterrows()]
    except Exception:  # noqa: BLE001
        return t, None


def load_earnings(tickers: list[str], budget_s: float = 900.0) -> dict[str, list]:
    """Resumable: tickers already fetched are kept; failures are retried on the next call."""
    have: dict = pd.read_pickle(EARN_PKL) if os.path.exists(EARN_PKL) else {}
    todo = [t for t in tickers if have.get(t) is None]
    t0 = time.time()
    for i in range(0, len(todo), 30):
        if time.time() - t0 > budget_s:
            break
        with ThreadPoolExecutor(max_workers=4) as ex:
            for t, r in ex.map(_one_earnings, todo[i:i + 30]):
                if r is not None:
                    have[t] = r
        pd.to_pickle(have, EARN_PKL)
        time.sleep(1.0)
    return {t: v for t, v in have.items() if v is not None}


if __name__ == "__main__":
    px = load_prices()
    print("prices", px["close"].shape, px["close"].index[0].date(), px["close"].index[-1].date(),
          [b for b in BENCH if b in px["close"].columns])
    u = universe_frame()
    e = load_earnings(u.loc[u.asset_type == "stock", "symbol"].tolist())
    print("earnings history for", len(e), "names; with events:", sum(1 for v in e.values() if v))
