"""High-volume return premium (Gervais-Kaniel-Mingelgrin): unusually heavy trading predicts a positive month."""
import numpy as np

NAME = "high_volume_premium"
DESC = "log of last-5-day volume over the 3-month average volume"
FMT = "volume {raw:+.2f} log-ratio vs 3m"


def raw(d):
    v = d.volume.reindex(columns=d.stocks()).replace(0, np.nan)
    return np.log(v.rolling(5, min_periods=3).mean() / v.rolling(63, min_periods=40).mean())
