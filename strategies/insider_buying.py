"""Insider buying (Seyhun, Cohen-Malloy-Pomorski): open-market purchases reported on SEC Form 4 in the last 90 days.
Data comes from strategies/edgar.py (cached daily); if EDGAR fails the signal is simply absent."""
import numpy as np

from .common import snapshot_panel

NAME = "insider_buying"
DESC = "log(1 + $ of open-market insider purchases, Form 4 code P) over 90 days, weighted by distinct buyers"
FMT = "{raw:.1f} insider-buy score"
HISTORY = False
NEUTRAL = False                 # sparse signal: whole-cross-section z, mostly zeros


def raw(d):
    vals = {}
    for t, r in (d.insiders or {}).items():
        if r is None:
            continue
        buy, sell, n = float(r.get("buy_usd", 0)), float(r.get("sell_usd", 0)), int(r.get("buyers", 0))
        vals[t] = float(np.log1p(max(buy - 0.25 * sell, 0.0)) * (1 + 0.5 * max(n - 1, 0)))
    return snapshot_panel(d, vals)
