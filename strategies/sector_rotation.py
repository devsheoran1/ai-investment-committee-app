"""Sector rotation (Moskowitz-Grinblatt, Faber): 3-month return of the stock's sector against the median sector.
Sector return = the SPDR sector ETF when it is in the price panel, else the equal-weight mean of its constituents."""
import numpy as np
import pandas as pd

NAME = "sector_rotation"
DESC = "3m return of the stock's sector minus the median sector (a sector-level signal, not sector-neutral)"
FMT = "sector 3m relative return {raw:+.1%}"
NEUTRAL = False
ETF = {"Information Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
       "Consumer Staples": "XLP", "Consumer Discretionary": "XLY", "Industrials": "XLI", "Utilities": "XLU",
       "Materials": "XLB", "Real Estate": "XLRE", "Communication Services": "XLC"}


def raw(d):
    c = d.close
    stocks = d.stocks()
    r63 = c / c.shift(63) - 1
    sec_ret = {}
    for s in d.sector.reindex(stocks).dropna().unique():
        e = ETF.get(s)
        if e in c.columns:
            sec_ret[s] = r63[e]
        else:
            cols = [t for t in stocks if d.sector.get(t) == s]
            if len(cols) >= 3:
                sec_ret[s] = r63[cols].mean(axis=1)
    if not sec_ret:
        return pd.DataFrame(np.nan, index=c.index, columns=stocks)
    sr = pd.DataFrame(sec_ret)
    rel = sr.sub(sr.median(axis=1), axis=0)
    out = pd.DataFrame({t: rel[d.sector[t]] if d.sector.get(t) in rel.columns else np.nan for t in stocks})
    return out
