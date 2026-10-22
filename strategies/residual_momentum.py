"""Residual momentum (Blitz-Huij-Martens): 12-1m return after removing the market beta, scaled by residual vol."""
import numpy as np

from .common import rets

NAME = "residual_momentum"
DESC = "12-1m idiosyncratic return (market beta removed, beta known one day earlier) per unit of residual vol"
FMT = "residual 12-1m score {raw:+.2f}"


def raw(d):
    r = rets(d)
    m = r[d.benchmark]
    stocks = d.stocks()
    r = r[stocks]
    beta = r.rolling(252, min_periods=126).cov(m).div(m.rolling(252, min_periods=126).var(), axis=0).shift(1)
    e = r - beta.mul(m, axis=0)
    num = e.shift(21).rolling(231, min_periods=150).sum()
    den = e.rolling(252, min_periods=150).std() * np.sqrt(231)
    return num / den.replace(0, np.nan)
