"""Leveraged-ETF trend filter: leveraged funds decay in chop, so only trend-following exposure makes sense.
Absolute (not cross-sectional) signal: z = distance above/below the 200dma in units of 10%."""
import numpy as np

NAME = "etf_trend_200dma"
DESC = "price vs 200dma for leveraged ETFs only (z = distance / 10%, clipped +-3)"
FMT = "{raw:+.1%} vs 200dma"
UNIVERSE = "etf"


def raw(d):
    c = d.close[d.leveraged()]
    return c / c.rolling(200, min_periods=150).mean() - 1


def z(d, raw_panel):
    return (raw_panel / 0.10).fillna(0.0).where(raw_panel.notna())
