"""
Shared pieces of the strategy ensemble (Oct 23). Pure Python / pandas, no LLM tokens.

A strategy is a module with:
    NAME, DESC (one line), FMT (how a raw value reads in the reason), UNIVERSE ("stock" | "etf" | "all"),
    HISTORY (True if the signal can be rebuilt point-in-time for the IC history), TILT (True = calendar tilt, no z)
    raw(d) -> DataFrame[dates x tickers]   the raw signal for EVERY date at once (vectorised, so the same code
                                           feeds the live run, the rolling IC and the backtest)
The z-score is cross-sectional per date, sector-neutral (z within the GICS sector), winsorised at +-3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINSOR = 3.0
MIN_SECTOR_NAMES = 5          # a sector with fewer names falls back to the whole cross-section


@dataclass
class Data:
    """Everything a strategy may read. All frames are indexed by date and cut at the decision date by the caller."""
    close: pd.DataFrame
    volume: pd.DataFrame
    sector: pd.Series                      # symbol -> GICS sector ("Leveraged ETF" / "Inverse ETF" for ETFs)
    industry: pd.Series                    # symbol -> GICS sub-industry (NaN for ETFs)
    asset_type: pd.Series                  # symbol -> stock | leveraged_etf | inverse_etf
    benchmark: str = "SPY"
    earn_events: dict | None = None        # ticker -> [{"date": "YYYY-MM-DD", "surprise_pct": float}] (point-in-time)
    next_earnings: dict | None = None      # ticker -> "YYYY-MM-DD"
    revisions: dict | None = None          # ticker -> {"rev_30d_0q": x, "rev_30d_1y": y}   (today's snapshot only)
    fundamentals: dict | None = None       # ticker -> {"gross_margin":, "roa":, ...}        (today's snapshot only)
    news: dict | None = None               # ticker -> [{"finbert": x, ...}]                 (today's headlines only)
    insiders: dict | None = None           # ticker -> {"buyers": n, "buy_usd": x, ...}      (SEC Form 4, last 90d)
    extra: dict = field(default_factory=dict)

    def stocks(self) -> list[str]:
        return [t for t in self.close.columns if self.asset_type.get(t) == "stock"]

    def leveraged(self) -> list[str]:
        return [t for t in self.close.columns if self.asset_type.get(t) == "leveraged_etf"]


def winsorise(z):
    return z.clip(-WINSOR, WINSOR)


def _row_z(x: pd.DataFrame) -> pd.DataFrame:
    sd = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)


def sector_neutral_z(raw: pd.DataFrame, sector: pd.Series, neutral: bool = True) -> pd.DataFrame:
    """Per-date z-score within sector (sectors with < MIN_SECTOR_NAMES names use the whole cross-section), +-3."""
    if raw.empty:
        return raw
    whole = _row_z(raw)
    if not neutral:
        return winsorise(whole).fillna(0.0).where(raw.notna())
    out = whole.copy()
    sec = sector.reindex(raw.columns)
    for s in sec.dropna().unique():
        cols = list(sec.index[sec == s])
        if len(cols) >= MIN_SECTOR_NAMES:
            out[cols] = _row_z(raw[cols])
    return winsorise(out).fillna(0.0).where(raw.notna())


def active(raw: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return raw.reindex(columns=[c for c in names if c in raw.columns])


class Strategy:
    """Thin wrapper so every module exposes the same surface (name, desc, raw, z, reason)."""

    def __init__(self, mod):
        self.mod = mod
        self.name: str = mod.NAME
        self.desc: str = mod.DESC
        self.fmt: str = getattr(mod, "FMT", "{raw:+.2f}")
        self.universe: str = getattr(mod, "UNIVERSE", "stock")
        self.history: bool = bool(getattr(mod, "HISTORY", True))
        self.tilt: bool = bool(getattr(mod, "TILT", False))
        self.neutral: bool = bool(getattr(mod, "NEUTRAL", True))

    def names(self, d: Data) -> list[str]:
        if self.universe == "stock":
            return d.stocks()
        if self.universe == "etf":
            return d.leveraged()
        return d.stocks() + d.leveraged()               # "all" = stocks + leveraged longs (inverse ETFs are hedges only)

    def raw(self, d: Data) -> pd.DataFrame:
        return self.mod.raw(d)

    def z(self, d: Data, raw: pd.DataFrame | None = None) -> pd.DataFrame:
        raw = self.raw(d) if raw is None else raw
        if hasattr(self.mod, "z"):                       # absolute (non cross-sectional) signals, e.g. the ETF trend filter
            return winsorise(self.mod.z(d, raw))
        return sector_neutral_z(active(raw, self.names(d)), d.sector, self.neutral)

    def reason(self, t: str, raw_val, z_val: float) -> str:
        try:
            body = self.fmt.format(raw=raw_val)
        except (ValueError, TypeError):
            body = str(raw_val)
        return f"{self.name}: {body} (z {z_val:+.1f})"
