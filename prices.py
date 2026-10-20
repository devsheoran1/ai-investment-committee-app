"""
Price accuracy layer (Oct 21): NAV must never be wrong because of a missing or bad price.

    history     yfinance in chunks of 50 -> per-ticker retry (2 tries) -> Stooq daily closes. Coverage logged.
    live        Finnhub quote, cross-checked against yfinance. Gap > 1% = "price unverified" (order blocked);
                Finnhub down = yfinance alone, flagged "single-source price".
    snapshot    Polygon grouped-daily close (one call, all US stocks) vs yfinance; gap > 0.5% logged,
                Polygon wins. Free tier only serves finished days, so a same-day 403 falls back to yfinance.
    stale       a held name no source prices -> last verified close, flagged "stale price", no trading.

Free tiers (checked Oct 21): Finnhub 60 calls/min; Polygon 5 calls/min. Keys come from .env and are never logged.
QUALITY is the per-run data-quality record shown in the dashboard header and the daily report.
"""
from __future__ import annotations

import io
import logging
import os
import time

import pandas as pd
import requests

import config

log = logging.getLogger("prices")

CHUNK = 50
LIVE_TOL = 0.01         # Finnhub vs yfinance at execution
SNAP_TOL = 0.005        # Polygon vs yfinance at the 21:45 snapshot
STALE, UNVERIFIED, SINGLE = "stale price", "price unverified", "single-source price"
BLOCKING = (STALE, UNVERIFIED)

QUALITY: dict = {"coverage": None, "missing": [], "sources": {}, "flags": {}}


class _Throttle:
    def __init__(self, per_min: int):
        self.gap, self.last = 60.0 / per_min, 0.0

    def wait(self):
        dt = self.last + self.gap - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        self.last = time.monotonic()


_FINNHUB = _Throttle(55)     # free tier 60/min, keep a margin
_POLYGON = _Throttle(5)


def _key(name: str) -> str | None:
    return os.getenv(name) or None


# --------------------------------------------------------------------------- #
# history (screening)
# --------------------------------------------------------------------------- #
def _yf(tickers, period):
    import yfinance as yf
    return yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True, group_by="column")


def _split(raw, tickers) -> dict[str, pd.DataFrame]:
    out = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        df = raw[f] if isinstance(raw.columns, pd.MultiIndex) else raw[[f]].rename(columns={f: tickers[0]})
        out[f.lower()] = df[[c for c in df.columns if df[c].notna().any()]]
    return out


def _stooq(ticker: str) -> pd.Series | None:
    """Stooq daily closes (no key). US tickers are <sym>.us, '.' -> '-'."""
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={ticker.lower().replace('.', '-')}.us&i=d", timeout=10)
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        return df["Close"].dropna() if "Close" in df and len(df) else None
    except Exception:  # noqa: BLE001
        return None


def download_history(tickers: list[str], period: str, fetch=_yf, stooq=_stooq, pause: float = 1.0) -> dict[str, pd.DataFrame]:
    fields = {k: pd.DataFrame() for k in ("open", "high", "low", "close", "volume")}

    def merge(part):
        for k, v in part.items():
            if not v.empty:
                fields[k] = fields[k].drop(columns=list(v.columns), errors="ignore").join(v, how="outer") \
                    if not fields[k].empty else v

    def missing():
        c = fields["close"]
        return [t for t in tickers if t not in c.columns or c[t].isna().all()]

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            raw = fetch(chunk, period)
            if raw is not None and not raw.empty:
                merge(_split(raw, chunk))
        except Exception:  # noqa: BLE001
            pass
    for t in missing():
        for attempt in range(2):
            try:
                time.sleep(pause * (attempt + 1) * 0.5)
                raw = fetch([t], period)
                if raw is not None and not raw.empty:
                    merge(_split(raw, [t]))
                    break
            except Exception:  # noqa: BLE001
                continue
    src = {t: "yfinance" for t in tickers if t not in missing()}
    for t in missing():
        s = stooq(t)
        if s is not None and len(s):
            s = s.rename(t)
            for k in ("open", "high", "low", "close"):   # Stooq fallback = closes only; OHLC set to close
                fields[k] = fields[k].join(s, how="outer") if not fields[k].empty else s.to_frame()
            src[t] = "stooq"
    miss = missing()
    QUALITY.update(coverage=(len(tickers) - len(miss), len(tickers)), missing=miss,
                   sources={"history": {s: sum(v == s for v in src.values()) for s in set(src.values())}})
    log.warning("%d/%d priced; missing: %s", len(tickers) - len(miss), len(tickers), ", ".join(miss) or "none")
    return fields


# --------------------------------------------------------------------------- #
# live (execution)
# --------------------------------------------------------------------------- #
def finnhub_quote(ticker: str) -> float | None:
    key = _key("FINNHUB_API_KEY")
    if not key:
        return None
    try:
        _FINNHUB.wait()
        c = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": ticker, "token": key}, timeout=10).json().get("c")
        return float(c) if c and c > 0 else None
    except Exception:  # noqa: BLE001
        return None


def yf_last(tickers: list[str]) -> dict[str, float]:
    try:
        raw = _yf(tickers, "5d")
        c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
        return {t: float(c[t].dropna().iloc[-1]) for t in c.columns if c[t].notna().any()}
    except Exception:  # noqa: BLE001
        return {}


def live_prices(tickers: list[str], last_verified: dict[str, float] | None = None,
                finnhub=finnhub_quote, yfin=yf_last) -> dict[str, dict]:
    """{ticker: {"px", "source", "flag"}}. flag in BLOCKING = do not trade that name."""
    yfp = yfin(list(tickers))
    out = {}
    for t in tickers:
        f, y = finnhub(t), yfp.get(t)
        if f and y:
            gap = abs(f / y - 1)
            out[t] = {"px": f, "source": "finnhub+yfinance",
                      "flag": f"{UNVERIFIED} (Finnhub {f:.2f} vs yfinance {y:.2f}, {gap:.1%})" if gap > LIVE_TOL else None}
        elif f or y:
            out[t] = {"px": f or y, "source": "finnhub" if f else "yfinance", "flag": SINGLE}
        elif last_verified and t in last_verified:
            out[t] = {"px": last_verified[t], "source": "last verified close", "flag": STALE}
        else:
            out[t] = {"px": None, "source": "none", "flag": STALE}
    _record(out)
    return out


def blocked(info: dict | None) -> str | None:
    flag = (info or {}).get("flag") or ""
    return flag if flag.startswith(BLOCKING) else None


# --------------------------------------------------------------------------- #
# snapshot (21:45)
# --------------------------------------------------------------------------- #
def polygon_grouped(day: str) -> dict[str, float]:
    key = _key("POLYGON_API_KEY")
    if not key:
        return {}
    try:
        _POLYGON.wait()
        j = requests.get(f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day}",
                         params={"adjusted": "true", "apiKey": key}, timeout=30).json()
        return {r["T"]: float(r["c"]) for r in j.get("results") or []}
    except Exception:  # noqa: BLE001
        return {}


def snapshot_prices(tickers: list[str], yf_px: dict[str, float], day: str,
                    last_verified: dict[str, float] | None = None, polygon=polygon_grouped) -> dict[str, dict]:
    poly = polygon(day)
    out = {}
    for t in tickers:
        p, y = poly.get(t.replace("-", ".")) or poly.get(t), yf_px.get(t)
        if p and y:
            gap = abs(p / y - 1)
            if gap > SNAP_TOL:
                log.warning("%s: Polygon %.2f vs yfinance %.2f (%.2f%%) -> Polygon", t, p, y, gap * 100)
            out[t] = {"px": p, "source": "polygon", "flag": f"gap {gap:.1%} vs yfinance" if gap > SNAP_TOL else None}
        elif p or y:
            out[t] = {"px": p or y, "source": "polygon" if p else "yfinance", "flag": None if p else SINGLE}
        elif last_verified and t in last_verified:
            out[t] = {"px": last_verified[t], "source": "last verified close", "flag": STALE}
    _record(out)
    return out


def _record(info: dict[str, dict]) -> None:
    QUALITY["flags"].update({t: v["flag"] for t, v in info.items() if v.get("flag")})
    for t, v in info.items():
        if not v.get("flag") and t in QUALITY["flags"]:
            QUALITY["flags"].pop(t)
    srcs = QUALITY["sources"].setdefault("live", {})
    for v in info.values():
        srcs[v["source"]] = srcs.get(v["source"], 0) + 1


def quality_line() -> str:
    cov = QUALITY.get("coverage")
    parts = [f"coverage {cov[0]}/{cov[1]} ({cov[0] / cov[1]:.1%})" if cov else "coverage n/a"]
    for k, d in QUALITY.get("sources", {}).items():
        parts.append(f"{k}: " + ", ".join(f"{s} {n}" for s, n in d.items()))
    flags = QUALITY.get("flags") or {}
    parts.append("flags: " + ("; ".join(f"{t} {f}" for t, f in flags.items()) if flags else "none"))
    return " · ".join(parts)
