"""Earnings-announcement premium (Frazzini-Lamont, Barber et al.): stocks earn extra return into their report."""
import numpy as np
import pandas as pd

NAME = "earnings_premium"
DESC = "report due within the next 5 trading days (1.0 = tomorrow ... 0.17 = in 5 days)"
FMT = "reports in the next 5 trading days, strength {raw:.2f}"
WINDOW = 5


def raw(d):
    idx = d.close.index
    out = pd.DataFrame(np.nan, index=idx, columns=d.close.columns)
    known = set(d.earn_events or {}) | set(d.next_earnings or {})        # names whose calendar we actually have
    out[[t for t in d.stocks() if t in known]] = 0.0
    dates: dict[str, list] = {}
    for t, evs in (d.earn_events or {}).items():
        dates.setdefault(t, []).extend(pd.Timestamp(e["date"]) for e in evs)
    for t, nd in (d.next_earnings or {}).items():
        dates.setdefault(t, []).append(pd.Timestamp(nd))
    for t, ds in dates.items():
        if t not in out.columns or d.asset_type.get(t) != "stock":
            continue
        arr = np.zeros(len(idx))
        for e in ds:
            if e > idx[-1]:                                  # report after the last bar: count business days ahead
                n = len(pd.bdate_range(idx[-1], e)) - 1
                if 1 <= n <= WINDOW:
                    arr[-1] = max(arr[-1], (WINDOW + 1 - n) / (WINDOW + 1))
                continue
            p = idx.searchsorted(e)
            for n in range(1, WINDOW + 1):
                if 0 <= p - n < len(idx):
                    arr[p - n] = max(arr[p - n], (WINDOW + 1 - n) / (WINDOW + 1))
        out[t] = arr
    return out
