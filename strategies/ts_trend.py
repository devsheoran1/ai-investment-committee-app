"""Time-series momentum (Moskowitz-Ooi-Pedersen): own 12-month return per unit of own volatility."""
import numpy as np

from .common import rets

NAME = "ts_trend"
DESC = "12-month return per unit of own volatility (trend persistence)"
FMT = "vol-scaled 12m trend {raw:+.2f}"
UNIVERSE = "all"


def raw(d):
    c = d.close
    vol = rets(d).rolling(252, min_periods=200).std() * np.sqrt(252)
    return (c / c.shift(252) - 1) / vol.replace(0, np.nan)
