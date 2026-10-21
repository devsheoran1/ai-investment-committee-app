"""
NYSE trading calendar (Stage 8). Hardcoded 2026 holidays and early closes (source: NYSE holiday schedule), so no
phantom snapshot is ever logged on a weekend or holiday and half-days are labelled.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

HOLIDAYS_2026 = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
                 "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}
EARLY_CLOSE_2026 = {"2026-11-27": "13:00 ET", "2026-12-24": "13:00 ET"}


def _d(day) -> dt.date:
    return day if isinstance(day, dt.date) and not isinstance(day, dt.datetime) else \
        (day.date() if isinstance(day, dt.datetime) else dt.date.fromisoformat(str(day)[:10]))


def is_trading_day(day) -> bool:
    d = _d(day)
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS_2026


def early_close(day) -> str | None:
    return EARLY_CLOSE_2026.get(_d(day).isoformat())


def previous_trading_day(day) -> dt.date:
    d = _d(day) - dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


# Oct 21: official-close helpers, all in America/New_York so DST shifts (US vs EU) can't move the close.
ET = ZoneInfo("America/New_York")


def close_dt(day) -> dt.datetime:
    """Official NYSE close of `day` as an aware ET datetime (16:00, or 13:00 on early-close days)."""
    hh = int((early_close(day) or "16:00").split(":")[0])
    return dt.datetime.combine(_d(day), dt.time(hh, 0), tzinfo=ET)


def last_completed_session(now: dt.datetime | None = None) -> dt.date:
    """Most recent trading day whose official close is already past."""
    n = (now or dt.datetime.now(ET)).astimezone(ET)
    d = n.date()
    if is_trading_day(d) and n >= close_dt(d):
        return d
    return previous_trading_day(d)
