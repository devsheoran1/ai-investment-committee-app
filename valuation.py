"""
One marking rule for NAV, P&L, SPY and alpha (Oct 23). Same date for all of them, everywhere (KPIs, report, forum post).

  * A position is never marked with a close dated BEFORE its fill session. The fill session is the NYSE session whose
    price the fill used (a fill at 00:03 Madrid = 18:03 ET is the same-day close). If prices.csv / the download has no
    close on or after that session yet, the position is valued at its fill price and flagged "stale mark".
  * SPY is measured from the inception snapshot's SPY to the SAME mark date as the fund. If the data has no close at or
    after the inception session, SPY is taken from the latest snapshot (so fund and SPY are both 0.00% at inception).
Pure functions, no network. The caller supplies the close panel (prices.csv or the live download).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

import config
import nyse_calendar as cal

MADRID = cal.ZoneInfo("Europe/Madrid")


def session_of(ts) -> dt.date:
    """NYSE session a timestamp belongs to: the same ET date from 09:30 ET on (open, or after the close),
    the previous trading day before the open or on a non-trading day. Naive timestamps are Madrid time."""
    t = pd.Timestamp(ts)
    t = t.tz_localize(MADRID) if t.tzinfo is None else t
    et = t.astimezone(cal.ET)
    if cal.is_trading_day(et.date()) and et.time() >= dt.time(9, 30):
        return et.date()
    return cal.previous_trading_day(et.date())


def fills(trades: pd.DataFrame) -> dict[str, dict]:
    """{symbol: {"session", "price"}} for the latest fill of each symbol (trades = db.get_trades())."""
    out: dict[str, dict] = {}
    if trades is None or trades.empty:
        return out
    t = trades.copy()
    hour = t["hour"] if "hour" in t.columns else "00:00:00"
    t["_ts"] = pd.to_datetime(t["date"].astype(str) + " " + pd.Series(hour, index=t.index).astype(str), errors="coerce")
    for sym, g in t.sort_values(["_ts"]).groupby("symbol"):
        r = g.iloc[-1]
        out[sym] = {"session": session_of(r["_ts"]) if pd.notna(r["_ts"]) else None, "price": float(r["price"])}
    return out


@dataclass
class Marks:
    px: dict[str, float]                      # price used for every position
    stale: dict[str, str] = field(default_factory=dict)   # symbol -> why it is a fill-price mark
    mark_date: dt.date | None = None          # date of the market data behind the marks (also SPY's date)
    spy_now: float | None = None
    spy_start: float | None = None
    inception: dt.date | None = None

    @property
    def spy_return(self) -> float | None:
        return None if not (self.spy_now and self.spy_start) else self.spy_now / self.spy_start - 1


def mark(positions: dict, close: pd.DataFrame, trades: pd.DataFrame, snaps: pd.DataFrame,
         bench: str | None = None) -> Marks:
    bench = bench or config.BENCHMARK
    fl = fills(trades)
    px, stale = {}, {}
    for t, p in positions.items():
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        f = fl.get(t, {})
        need = f.get("session")
        if len(s) and (need is None or s.index[-1].date() >= need):
            px[t] = float(s.iloc[-1])
        else:
            px[t] = float(f.get("price") or p["avg_cost"])
            stale[t] = (f"stale mark: no close on/after fill session {need}" if len(s) else "stale mark: no price data")
    inc = session_of(snaps["ts"].iloc[0]) if len(snaps) and "ts" in snaps.columns and pd.notna(snaps["ts"].iloc[0]) else None
    spy_start = float(snaps["benchmark_px"].iloc[0]) if len(snaps) and snaps["benchmark_px"].iloc[0] else None
    b = close[bench].dropna() if bench in close.columns else pd.Series(dtype=float)
    if len(b) and (inc is None or b.index[-1].date() >= inc):
        spy_now, mark_date = float(b.iloc[-1]), b.index[-1].date()
    elif len(snaps):                                   # data older than the inception: no move to measure yet
        spy_now, mark_date = float(snaps["benchmark_px"].iloc[-1]), session_of(snaps["ts"].iloc[-1])
    else:
        spy_now, mark_date = None, None
    return Marks(px, stale, mark_date, spy_now, spy_start, inc)


def pct(x: float, d: int = 2, signed: bool = True) -> str:
    """'+1.20%' / '-0.35%' / '0.00%': a value that rounds to zero never shows a sign ('-0.00%')."""
    r = round(float(x) * 100, d)
    if r == 0:
        r = 0.0
    return f"{r:+.{d}f}%" if signed and r != 0 else f"{r:.{d}f}%"
