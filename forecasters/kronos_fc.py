"""
Kronos-mini forecaster (github.com/shiyu-coder/Kronos, MIT; weights NeoQuasar/Kronos-mini, 4.1M params,
tokenizer NeoQuasar/Kronos-Tokenizer-2k). A foundation model for candlesticks, trained on ~12B K-line
records. We sample several 5-day paths per ticker and summarise them:

    exp_ret_5d   mean simple return of the final close vs the last observed close
    p_up         share of sampled paths that end above the last close
    dispersion   std of the sampled 5d returns (the model's own uncertainty)

Never raises: any failure (download, load, memory) returns None and the committee carries on LSTM-only.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("kronos")

MODEL_ID = "NeoQuasar/Kronos-mini"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
MAX_CONTEXT = 2048            # Kronos-mini context length (README table)
BATCH = 16                    # series per forward pass: keeps peak RAM low on an 8GB Mac
_STATE: dict = {}


def device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(dev: str):
    if dev not in _STATE:
        from vendor.kronos.model import Kronos, KronosTokenizer
        tok = KronosTokenizer.from_pretrained(TOKENIZER_ID).to(dev).eval()
        mdl = Kronos.from_pretrained(MODEL_ID).to(dev).eval()
        _STATE[dev] = (tok, mdl)
    return _STATE[dev]


def _infer(dev, frames: list[pd.DataFrame], dates: pd.DatetimeIndex, future: pd.DatetimeIndex,
           pred_len: int, sample_count: int) -> np.ndarray:
    """frames: equal-length OHLCV DataFrames. Returns simple 5d returns, shape (n_series, sample_count)."""
    import torch
    from vendor.kronos.model.kronos import auto_regressive_inference, calc_time_stamps
    tok, mdl = _load(dev)
    cols = ["open", "high", "low", "close", "volume", "amount"]
    x_stamp = calc_time_stamps(pd.Series(dates)).values.astype(np.float32)
    y_stamp = calc_time_stamps(pd.Series(future)).values.astype(np.float32)
    out = []
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        xs, means, stds, last = [], [], [], []
        for df in chunk:
            x = df[cols].values.astype(np.float32)
            mu, sd = x.mean(axis=0), x.std(axis=0)
            xs.append(np.clip((x - mu) / (sd + 1e-5), -5, 5))
            means.append(mu); stds.append(sd); last.append(float(df["close"].iloc[-1]))
        n = len(chunk)
        xb = torch.from_numpy(np.stack(xs)).to(dev)
        xsb = torch.from_numpy(np.repeat(x_stamp[None], n, axis=0)).to(dev)
        ysb = torch.from_numpy(np.repeat(y_stamp[None], n, axis=0)).to(dev)
        z = auto_regressive_inference(tok, mdl, xb, xsb, ysb, MAX_CONTEXT, pred_len, clip=5, T=1.0,
                                      top_k=0, top_p=0.9, sample_count=sample_count, return_samples=True)
        z = z[:, :, -pred_len:, 3]                                    # close channel, (n, S, pred_len)
        for j in range(n):
            close_final = z[j, :, -1] * (stds[j][3] + 1e-5) + means[j][3]
            out.append(close_final / last[j] - 1.0)
        del xb, xsb, ysb, z
    return np.array(out)


def predict(market: dict, tickers: list[str], end_pos: int | None = None, lookback: int = 400,
            pred_len: int = 5, sample_count: int = 5, dev: str | None = None) -> dict[str, dict]:
    """Forecast pred_len trading days after row `end_pos` (default: the last row) of the market frames.
    Uses only rows <= end_pos, so it is replay-safe. Tickers with gaps in the window are skipped."""
    close = market["close"]
    end_pos = len(close) - 1 if end_pos is None else end_pos
    L = min(lookback, end_pos + 1)
    sl = slice(end_pos - L + 1, end_pos + 1)
    idx = close.index[sl]
    future = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=pred_len)
    frames, names = [], []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].iloc[sl]
        o = market["open"][t].iloc[sl] if t in market["open"].columns else c
        h = market["high"][t].iloc[sl] if "high" in market and t in market["high"].columns else np.maximum(o, c)
        lo = market["low"][t].iloc[sl] if "low" in market and t in market["low"].columns else np.minimum(o, c)
        v = market["volume"][t].iloc[sl] if t in market["volume"].columns else pd.Series(0.0, index=idx)
        df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}).ffill()
        if df.isna().any().any() or len(df) < L:
            continue
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        frames.append(df.reset_index(drop=True)); names.append(t)
    if not frames:
        return {}
    rets = _infer(dev or device(), frames, idx, future, pred_len, sample_count)
    return {t: {"exp_ret_5d": float(r.mean()), "p_up": float((r > 0).mean()), "dispersion": float(r.std())}
            for t, r in zip(names, rets)}


def forecast(market: dict, tickers: list[str], lookback: int = 400, sample_count: int = 5) -> dict | None:
    """Live forecast for the committee. None on any failure (caller falls back to LSTM-only)."""
    t0 = time.time()
    try:
        try:
            res = predict(market, tickers, lookback=lookback, sample_count=sample_count)
        except Exception as e:  # noqa: BLE001  an MPS hiccup should not cost us the model: retry on CPU
            log.warning("Kronos on %s failed (%s); retrying on cpu", device(), str(e)[:100])
            _STATE.clear()
            res = predict(market, tickers, lookback=lookback, sample_count=sample_count, dev="cpu")
        if not res:
            return None
        return {"model": f"Kronos-mini ({MODEL_ID}, {sample_count} samples)", "forecasts": res,
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        log.warning("Kronos unavailable, LSTM-only: %s", str(e)[:200])
        return None
