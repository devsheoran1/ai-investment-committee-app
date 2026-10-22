"""Low-beta / betting-against-beta (Frazzini-Pedersen), Vasicek-shrunk 1y beta to the benchmark."""
from .common import rets

NAME = "low_beta"
DESC = "minus the 1y beta to the benchmark (shrunk 1/3 toward 1)"
FMT = "beta {raw:.2f} (sign flipped)"


def raw(d):
    r = rets(d)
    m = r[d.benchmark]
    stocks = d.stocks()
    beta = r[stocks].rolling(252, min_periods=126).cov(m).div(m.rolling(252, min_periods=126).var(), axis=0)
    return -(2 / 3 * beta + 1 / 3)
