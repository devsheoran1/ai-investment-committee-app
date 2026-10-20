"""
LightGBM forecaster for the horse race (IPS C3.6, Stage 4). Pooled cross-sectional gradient boosting on:
1/3/12-month momentum, 20-day vol, RSI-14, distance from the 50dma, industry momentum (1m return of the stock's
GICS sub-industry peers in the S&P 500), 1-week reversal, and post-earnings drift. Target: next 5-day return.
Monthly walk-forward over the last 250 trading days (same protocol as the LSTM), plus a live forecast.

LightGBM and PyTorch each bundle their own OpenMP runtime and abort when loaded in the same process, so the
training runs in a subprocess (forecasters/lgbm_worker.py) that never imports torch. Never raises: None on failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import config

FEATURES = ["mom_1m", "mom_3m", "mom_12_1", "vol_20d", "rsi_14", "dist_50dma", "industry_mom_1m", "reversal_1w", "pead"]
H = 5


def _rsi(px: pd.DataFrame, n=14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _pead_panel(close: pd.DataFrame, tickers, earnings: dict) -> pd.DataFrame:
    """Historical drift score per date: +/-1 after a report whose surprise and 3-day excess reaction agree,
    decaying over PEAD_WINDOW_DAYS. Only uses data up to each date."""
    out = pd.DataFrame(0.0, index=close.index, columns=tickers)
    spy = close[config.BENCHMARK]
    for t in tickers:
        for rep in (earnings.get(t) or {}).get("last4") or []:
            d = pd.Timestamp(rep["date"])
            pos = int(close.index.searchsorted(d)) + (1 if rep.get("time_et", "00:00") >= "16:00" else 0)
            if pos < 1 or pos + 1 >= len(close) or rep.get("surprise_pct") is None:
                continue
            ann = close[t].iloc[pos + 1] / close[t].iloc[pos - 1] - spy.iloc[pos + 1] / spy.iloc[pos - 1]
            sign = 1 if rep["surprise_pct"] > 0 and ann > 0 else -1 if rep["surprise_pct"] < 0 and ann < 0 else 0
            for k in range(config.PEAD_WINDOW_DAYS + 1):
                i = pos + 1 + k                                  # known from the session after the reaction window
                if i < len(close):
                    out.iloc[i, out.columns.get_loc(t)] = sign * (1 - k / config.PEAD_WINDOW_DAYS)
    return out


def features(close: pd.DataFrame, tickers: list[str], industries: dict[str, str], earnings: dict | None = None):
    """Long panel: rows (date, ticker) with FEATURES and the forward 5-day return target."""
    px = close[tickers]
    allpx = close.drop(columns=[c for c in close.columns if c not in industries], errors="ignore")
    r1m_all = allpx / allpx.shift(21) - 1
    ind = pd.Series(industries).reindex(r1m_all.columns)
    ind_mom = r1m_all.T.groupby(ind).transform("mean").T                 # each stock's industry 1m return
    f = {
        "mom_1m": px / px.shift(21) - 1, "mom_3m": px / px.shift(63) - 1, "mom_12_1": px.shift(21) / px.shift(252) - 1,
        "vol_20d": np.log(px).diff().rolling(20).std() * np.sqrt(252), "rsi_14": _rsi(px),
        "dist_50dma": px / px.rolling(50).mean() - 1, "industry_mom_1m": ind_mom.reindex(columns=tickers),
        "reversal_1w": px / px.shift(5) - 1, "pead": _pead_panel(close, tickers, earnings or {}),
    }
    y = px.shift(-H) / px - 1
    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long["y"] = y.stack(future_stack=True)
    long.index.names = ["date", "ticker"]
    return long


def forecast(close: pd.DataFrame, tickers: list[str], industries: dict[str, str], earnings: dict | None = None,
             wf_days: int = 250, retrain_every: int = 21) -> dict | None:
    try:
        long = features(close, tickers, industries, earnings)
        dates = close.index
        date_pos = {d: i for i, d in enumerate(dates)}
        long = long.reset_index()
        long["t"] = long["date"].map(date_pos)
        long = long.dropna(subset=FEATURES[:6])                         # pead/industry may be NaN early
        T = len(dates)
        first = max(300, T - H - wf_days)
        tmp = tempfile.mkdtemp(prefix="lgbm_")
        inp, out = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(inp, X=long[FEATURES].fillna(0).values.astype(np.float32), y=long["y"].values.astype(np.float32),
                 t=long["t"].values.astype(np.int32), T=np.int32(T), H=np.int32(H), first=np.int32(first),
                 every=np.int32(retrain_every))
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgbm_worker.py")
        env = {k: v for k, v in os.environ.items()}
        r = subprocess.run([sys.executable, "-I", worker, inp, out], capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": r.stderr[-300:]}
        res = np.load(out)
        pred = res["pred"]                                              # NaN where not predicted
        long["pred"] = pred
        oos = long[(long["t"] >= first) & long["pred"].notna() & long["y"].notna()]
        live = long[long["t"] == T - 1].set_index("ticker")["pred"]
        oos_rows = [{"ticker": tk, "offset": int(T - 1 - H - tt), "pred": float(p), "y": float(yy)}
                    for tk, tt, p, yy in zip(oos["ticker"], oos["t"], oos["pred"], oos["y"])]
        hit = float((np.sign(oos["pred"]) == np.sign(oos["y"])).mean()) if len(oos) else 0.5
        return {"model": f"LightGBM (9 features, monthly walk-forward {wf_days}d)", "oos_preds": oos_rows,
                "forecasts": {t: {"pred_5d_return_pct": round(float(live.get(t, np.nan)) * 100, 2)} for t in tickers
                              if t in live.index and not np.isnan(live.get(t))},
                "oos_directional_hit_rate": round(hit, 3), "n_oos": int(len(oos)),
                "feature_importance": json.loads(str(res["importance"]))}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:300]}
