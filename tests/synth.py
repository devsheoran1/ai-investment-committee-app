"""Shared synthetic market for the strategy / backtest tests: 5 sectors x 14 stocks + SPY + 8 leveraged ETFs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import strategies

SECTORS = ["Energy", "Health Care", "Financials", "Industrials", "Utilities"]


def market(T=420, seed=1, n_per=14):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=T)
    stocks = [f"{s[:2].upper()}{i}" for s in SECTORS for i in range(n_per)]
    sec = {t: SECTORS[i // n_per] for i, t in enumerate(stocks)}
    ind = {t: f"{sec[t]}-{(i % n_per) % 3}" for i, t in enumerate(stocks)}
    mkt = rng.normal(0.0004, 0.01, T)
    drift = rng.normal(0, 0.0008, len(stocks))
    r = np.array([0.9 * mkt + drift[j] + rng.normal(0, 0.012, T) for j in range(len(stocks))]).T
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=stocks)
    close["SPY"] = 400 * np.exp(np.cumsum(mkt))
    etfs = [f"LV{i}" for i in range(8)]
    for e in etfs:
        close[e] = 50 * np.exp(np.cumsum(3 * mkt + rng.normal(0, 0.01, T)))
    vol = pd.DataFrame(rng.integers(2_000_000, 8_000_000, close.shape).astype(float), index=idx, columns=close.columns)
    opn = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.002, close.shape))
    uni = pd.DataFrame({"symbol": stocks + etfs + ["SPY"],
                        "sector": [sec[t] for t in stocks] + ["Leveraged ETF"] * len(etfs) + ["ETF"],
                        "industry": [ind[t] for t in stocks] + [np.nan] * (len(etfs) + 1),
                        "asset_type": ["stock"] * len(stocks) + ["leveraged_etf"] * len(etfs) + ["benchmark"],
                        "name": stocks + etfs + ["SPY"], "cik": np.nan})
    at = dict(zip(uni.symbol, uni.asset_type))
    return close, vol, opn, uni, at, stocks, etfs


def data(T=420, seed=1, **kw):
    close, vol, opn, uni, at, stocks, etfs = market(T, seed)
    return strategies.build_data(close, vol, uni, at, **kw), (close, vol, opn, uni, at, stocks, etfs)
