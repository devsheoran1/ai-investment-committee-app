"""Glue between the orchestrator and the strategy package: gathers the live inputs and runs the ensemble."""
from __future__ import annotations

import pandas as pd

import config

from . import build_data, events_from_earnings, run


def fetch_quality(tickers: list[str]) -> dict[str, dict]:
    """gross margin + ROA per stock (yfinance .info, cached per day). Failure -> {} (quality_gp is then skipped)."""
    import data
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            import yfinance as yf
            info = yf.Ticker(t).info or {}
            return t, {"gross_margin": info.get("grossMargins"), "roa": info.get("returnOnAssets")}
        except Exception:  # noqa: BLE001
            return t, {}

    def all_():
        with ThreadPoolExecutor(max_workers=6) as ex:
            return dict(ex.map(one, tickers))
    try:
        return data._cached(data._key("quality_gp", tickers), all_, keep_if=lambda v: sum(bool(x) for x in v.values()) >= 0.3 * max(1, len(v)))
    except Exception:  # noqa: BLE001
        return {}


def run_live(close, volume, universe: pd.DataFrame, asset_types: dict, tickers: list[str], earnings: dict, news: dict,
             ciks: dict, stock_names: list[str], replay: bool = False, synthetic: bool = False, asof=None) -> dict | None:
    """The ensemble as of the last bar of `close`. Replay (look-ahead guard): no insiders, no revisions snapshot,
    no news (news is {} there already), no quality snapshot."""
    if not config.ENSEMBLE.get("enabled", True):
        return None
    ev, nxt = events_from_earnings(earnings)
    if replay:
        ev = {t: [e for e in evs if pd.Timestamp(e["date"]) < pd.Timestamp(asof)] for t, evs in ev.items()}
    revisions = None if replay else {t: {"rev_30d_0q": e.get("rev_30d_0q"), "rev_30d_1y": e.get("rev_30d_1y")}
                                     for t, e in (earnings or {}).items() if e}
    insiders = quality = None
    if not replay and not synthetic:
        from .edgar import get_insiders
        insiders = get_insiders({t: ciks.get(t) for t in stock_names})
        quality = fetch_quality(stock_names)
    d = build_data(close, volume, universe, asset_types, benchmark=config.BENCHMARK, earn_events=ev, next_earnings=nxt,
                   revisions=revisions, fundamentals=quality, news=news if news else None, insiders=insiders)
    return run(d, [t for t in tickers if t in close.columns], decision_day=asof)
