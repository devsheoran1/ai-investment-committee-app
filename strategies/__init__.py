"""
Multi-strategy ensemble (Oct 23). `run(...)` is what the orchestrator calls; the backtest imports the same pieces.
Every strategy is pure Python (no LLM tokens). A failure anywhere = `None` and Quant keeps its legacy score.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd

import config

from .base import Data, Strategy
from .ensemble import cap_weights, ic_stats, learn_weights, rank_ic, score_cross_section  # noqa: F401
from .regime import apply_state, current_state, family_mult

MODULES = ["xs_momentum_12_1", "ts_trend", "industry_momentum", "residual_momentum", "high_52w", "st_reversal_1w",
           "pead_sue", "eps_revisions", "earnings_premium", "quality_gp", "low_beta", "high_volume_premium",
           "insider_buying", "news_surprise", "sector_rotation", "etf_trend_200dma", "turn_of_month"]


def registry(names: list[str] | None = None) -> list[Strategy]:
    return [Strategy(importlib.import_module(f"strategies.{m}")) for m in (names or MODULES)]


def build_data(close, volume, universe: pd.DataFrame, asset_types: dict, benchmark: str = "SPY", **kw) -> Data:
    sector = pd.Series(dict(zip(universe["symbol"], universe["sector"])))
    industry = pd.Series(dict(zip(universe["symbol"], universe["industry"]))) if "industry" in universe.columns else pd.Series(dtype=object)
    at = pd.Series(asset_types)
    return Data(close=close, volume=volume, sector=sector, industry=industry, asset_type=at, benchmark=benchmark, **kw)


def events_from_earnings(earnings: dict | None) -> tuple[dict, dict]:
    """data.get_earnings() output -> (past events with surprise, next report dates)."""
    ev, nxt = {}, {}
    for t, e in (earnings or {}).items():
        ev[t] = [{"date": r["date"], "surprise_pct": r.get("surprise_pct")} for r in e.get("last4", []) if r.get("date")]
        if e.get("next_date"):
            nxt[t] = e["next_date"]
    return ev, nxt


def tom_tilt(decision_day=None) -> float:
    from .turn_of_month import in_window
    day = pd.Timestamp(decision_day) if decision_day is not None else pd.Timestamp.today()
    return config.ENSEMBLE["tom_tilt"] if in_window(day) else 0.0


def log_snapshot_z(panels: dict, strategies: list[Strategy], asof: str) -> int:
    """Oct 24: append today's z of every snapshot-only signal (no point-in-time history) to reports/snapshot_z_log.csv,
    so a measured IC can replace the 0.5 x 1/N prior once ~60 sessions exist. One row per asof x strategy x ticker
    (re-running the same day replaces that day). Live fund only; tests and replays never write."""
    live_db = os.path.abspath(config.DB_PATH) == os.path.join(config.BASE_DIR, "finance.db")
    if not live_db and "FUND_SNAPSHOT_Z_LOG" not in os.environ:
        return 0
    rows = []
    for s in strategies:
        if s.tilt or s.history or s.name not in panels:
            continue
        z = panels[s.name].iloc[-1].dropna()
        rows += [(asof, s.name, t, round(float(v), 4)) for t, v in z.items()]
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=["asof", "strategy", "ticker", "z"])
    path = config.SNAPSHOT_Z_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[old["asof"] != asof], new], ignore_index=True)
    new.to_csv(path, index=False)
    return len(rows)


def run(d: Data, tickers: list[str], decision_day=None, names: list[str] | None = None) -> dict | None:
    """Ensemble as of the last bar of d.close, for `tickers` (the shortlist + holdings). None on failure."""
    try:
        strategies = registry(names)
        panels = {s.name: s.z(d) for s in strategies if not s.tilt}
        raws = {s.name: s.raw(d) for s in strategies if not s.tilt}
        # a snapshot strategy with no data at all (EDGAR down, no news...) is skipped, not counted as 0
        live = [s for s in strategies if s.tilt or panels[s.name].iloc[-1].notna().any()]
        skipped = [s.name for s in strategies if s not in live]
        weights = learn_weights(d, live, {k: v for k, v in panels.items() if k in {s.name for s in live}})
        has_hist = {s.name: s.history for s in live if not s.tilt}
        state = current_state(d.close, d.benchmark)                       # Oct 24 regime allocator (SPY-based, Python)
        w_final = apply_state(weights["weights"], state, has_hist)
        zrows = {n: p.iloc[-1] for n, p in panels.items()}
        rrows = {n: p.iloc[-1] for n, p in raws.items()}
        scores = score_cross_section(zrows, rrows, live, w_final, tickers)
        try:
            log_snapshot_z(panels, live, str(d.close.index[-1].date()))
        except Exception:  # noqa: BLE001  logging must never break a run
            pass
        return {"scores": scores, "weights": w_final, "weights_pre_regime": weights["weights"], "regime_state": state,
                "regime_mult": family_mult(state) if config.REGIME_ALLOC.get("enabled") else {},
                "has_history": has_hist, "ic_table": weights["table"], "mode": weights["mode"],
                "allocated": round(sum(w_final.values()), 4), "skipped": skipped, "tom_tilt": tom_tilt(decision_day),
                "asof": str(d.close.index[-1].date()),
                "descriptions": {s.name: s.desc for s in strategies}}
    except Exception:  # noqa: BLE001  never break a run: Quant falls back to its legacy score
        import traceback
        traceback.print_exc()
        return None
