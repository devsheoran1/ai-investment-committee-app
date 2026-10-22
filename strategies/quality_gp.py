"""Gross profitability / quality (Novy-Marx). yfinance has no gross profit / assets, so the proxy is the average
rank of gross margin and return on assets. Today's snapshot only (no point-in-time history)."""
import pandas as pd

from .common import snapshot_panel

NAME = "quality_gp"
DESC = "average rank of gross margin and ROA (proxy for gross profit / assets)"
FMT = "quality rank {raw:.2f}"
HISTORY = False


def raw(d):
    f = pd.DataFrame({t: {"gm": (v or {}).get("gross_margin"), "roa": (v or {}).get("roa")}
                      for t, v in (d.fundamentals or {}).items()}).T
    if f.empty:
        return snapshot_panel(d, {})
    f = f.astype(float)
    score = f.rank(pct=True).mean(axis=1, skipna=True)
    return snapshot_panel(d, score.dropna().to_dict())
