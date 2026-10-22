"""Turn-of-the-month effect (Ariel, Lakonishok-Smidt): equities earn most of their return from the last trading day
of a month to the 3rd of the next. TILT ONLY: a uniform market-wide offset, never a stock-picking vote."""
import numpy as np
import pandas as pd

NAME = "turn_of_month"
DESC = "+1 on the last trading day of the month and the first 3 of the next, else 0 (tilt only)"
FMT = "turn-of-month window {raw:+.0f}"
TILT = True
UNIVERSE = "all"


def flags(index: pd.DatetimeIndex) -> pd.Series:
    """1.0 on the last business day of each month and the first three of the next (holidays ignored)."""
    days = index.normalize().values.astype("datetime64[D]")
    month_start = index.to_period("M").start_time.values.astype("datetime64[D]")
    nth = np.busday_count(month_start, days)                          # 0 = first business day of the month
    is_last = np.array([d == (pd.Timestamp(e) if pd.Timestamp(e).weekday() < 5 else pd.Timestamp(e) - pd.offsets.BDay(1))
                        for d, e in zip(index.normalize(), index.to_period("M").end_time.normalize())])
    return pd.Series(((nth < 3) | is_last).astype(float), index=index)


def raw(d):
    f = flags(d.close.index)
    return pd.DataFrame({t: f for t in d.close.columns})


def in_window(day) -> bool:
    """Is a calendar day inside the window? Business-day approximation (holidays ignored) for the live decision day."""
    day = pd.Timestamp(day).normalize()
    month = pd.bdate_range(day.replace(day=1), day + pd.offsets.MonthEnd(0))
    pos = month.get_indexer([day])[0] if day in month else -1
    if pos < 0:
        return False
    if pos < 3 or pos == len(month) - 1:
        return True
    return False
