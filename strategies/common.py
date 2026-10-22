"""Panel helpers shared by the strategy modules (all vectorised: one call = every date)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def rets(d) -> pd.DataFrame:
    return d.close.pct_change(fill_method=None)


def event_panel(d, events: dict | None, key: str, window: int) -> pd.DataFrame:
    """Value of the latest event (from the first bar AFTER its date), decaying linearly to 0 over `window` bars."""
    idx = d.close.index
    out = pd.DataFrame(np.nan, index=idx, columns=d.close.columns)
    for t, evs in (events or {}).items():
        if t not in out.columns:
            continue
        arr = np.full(len(idx), np.nan)
        for e in sorted(evs, key=lambda x: x["date"]):
            v = e.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            p = idx.searchsorted(pd.Timestamp(e["date"]), side="right")      # announcement time unknown -> next bar
            if p >= len(idx):
                continue
            n = min(window, len(idx) - p)
            arr[p:p + n] = float(v) * (1 - np.arange(n) / window)
        out[t] = arr
    return out


def snapshot_panel(d, values: dict | None) -> pd.DataFrame:
    """Only today's value exists (no history): last row filled, everything else NaN."""
    out = pd.DataFrame(np.nan, index=d.close.index, columns=d.close.columns)
    for t, v in (values or {}).items():
        if t in out.columns and v is not None and not (isinstance(v, float) and np.isnan(v)):
            out.iloc[-1, out.columns.get_loc(t)] = float(v)
    return out
