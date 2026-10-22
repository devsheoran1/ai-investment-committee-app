"""Analyst EPS-estimate revisions (Chan-Jegadeesh-Lakonishok): 30-day change of the current-quarter / next-year estimate."""
import numpy as np

from .common import snapshot_panel

NAME = "eps_revisions"
DESC = "average 30-day % revision of the current-quarter and next-year EPS estimate"
FMT = "30d EPS-estimate revision {raw:+.1%}"
HISTORY = False                 # yfinance only serves today's revision snapshot


def raw(d):
    vals = {}
    for t, r in (d.revisions or {}).items():
        xs = [x for x in (r.get("rev_30d_0q"), r.get("rev_30d_1y")) if x is not None]
        if xs:
            vals[t] = float(np.clip(np.mean(xs), -0.5, 0.5))
    return snapshot_panel(d, vals)
