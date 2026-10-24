"""
Regime allocator over strategy families (Oct 24). Python rules, SPY only, no LLM:

    calm_uptrend       SPY > 200dma and 20d vol < its 1y median
    drawdown_high_vol  SPY > 5% below its 1y high and 20d vol > its 1y median
    rebound            SPY < 200dma but > 5% above its 63d low
    normal             anything else (also: not enough history)
First match wins in that order. Each state multiplies the family weights (config.REGIME_ALLOC), then renormalises.
The Quant LLM may additionally tilt ONE family by +-20% a day with a written reason (`tilt`); it never touches Risk limits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

from .ensemble import reweight

STATES = ("calm_uptrend", "drawdown_high_vol", "rebound", "normal")


from .ensemble import family_of  # noqa: E402,F401  (single definition, shared with the weight rule)


def regime_states(spy: pd.Series) -> pd.Series:
    """State label for every date (vectorised; the live run reads the last value, the backtest the whole series)."""
    R = config.REGIME_ALLOC
    spy = spy.dropna()
    vol = spy.pct_change().rolling(R["vol_window"]).std() * np.sqrt(252)
    med = vol.rolling(R["vol_median_window"], min_periods=R["vol_median_window"] // 2).median()
    ma200 = spy.rolling(200, min_periods=150).mean()
    dd = spy / spy.rolling(R["peak_window"], min_periods=60).max() - 1
    off_low = spy / spy.rolling(R["low_window"], min_periods=20).min() - 1
    calm = (spy > ma200) & (vol < med)
    ddhv = (dd <= R["dd_from_peak"]) & (vol > med)
    reb = (spy < ma200) & (off_low > R["rebound_from_low"])
    return pd.Series(np.select([calm, ddhv, reb], STATES[:3], default="normal"), index=spy.index)


def current_state(close: pd.DataFrame, benchmark: str = "SPY") -> str:
    if not config.REGIME_ALLOC.get("enabled") or benchmark not in close.columns:
        return "normal"
    s = regime_states(close[benchmark])
    return str(s.iloc[-1]) if len(s) else "normal"


def family_mult(state: str) -> dict[str, float]:
    return dict(config.REGIME_ALLOC["multipliers"].get(state, config.REGIME_ALLOC["multipliers"]["normal"]))


def _per_strategy(mult_by_family: dict[str, float], names) -> dict[str, float]:
    return {k: mult_by_family.get(family_of(k), 1.0) for k in names}


def apply_state(weights: dict, state: str, has_history: dict[str, bool], cfg: dict | None = None) -> dict:
    if not config.REGIME_ALLOC.get("enabled"):
        return dict(weights)
    return reweight(weights, _per_strategy(family_mult(state), weights), has_history, cfg)


def tilt(weights: dict, family: str, pct: float, has_history: dict[str, bool], cfg: dict | None = None) -> dict:
    """Scale one family by 1 + pct (|pct| <= LLM_TILT_MAX), renormalise, re-cap."""
    pct = float(np.clip(pct, -config.LLM_TILT_MAX, config.LLM_TILT_MAX))
    return reweight(weights, _per_strategy({family: 1.0 + pct}, weights), has_history, cfg)


def rescore(strategies_detail: dict, weights: dict, cfg: dict | None = None) -> float | None:
    """Ensemble score (-1..1) of one ticker under new weights, from the per-strategy z already in the detail."""
    cfg = {**config.ENSEMBLE, **(cfg or {})}
    num = den = 0.0
    for name, v in strategies_detail.items():
        w = weights.get(name, 0.0)
        if w > 0:
            num += w * v["z"]
            den += w
    return None if den <= 0 else float(np.clip(num / den / cfg["z_to_score"], -1, 1))
