"""
Ensemble: rolling out-of-sample IC -> weights -> one score per ticker (Oct 23, A2/A3).

    IC_k      = cross-sectional Spearman(z_s at day k, 5-day forward return k -> k+5), only days whose forward return
                is fully realised by the decision date (walk-forward, no look-ahead); last 60 such sessions
    t         = mean(IC) / (std(IC) / sqrt(n))
    shrunk IC = mean(IC) x max(0, 1 - 1/t^2)    (positive-part James-Stein: t <= 1 -> 0, noisy t shrunk hardest)
    weight    = shrunk IC, floored at 0, normalised, capped at 20% each (excess redistributed), t < 1 -> 0
    < 60 sessions of IC (system-wide, or a snapshot-only signal with no history) -> equal-weight prior (config knob)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

from .base import Data, Strategy


def rank_ic(z: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 20) -> pd.Series:
    """Per-date Spearman correlation between z and the forward return (NaN where < min_names names)."""
    cols = z.columns.intersection(fwd.columns)
    a, b = z[cols], fwd[cols]
    ok = a.notna() & b.notna()
    a, b = a.where(ok), b.where(ok)
    ic = a.rank(axis=1).corrwith(b.rank(axis=1), axis=1)
    return ic.where(ok.sum(axis=1) >= min_names)


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.shift(-horizon) / close - 1


def ic_stats(ic: pd.Series, sessions: int, horizon: int | None = None) -> dict:
    """Mean rank IC over the last `sessions` daily ICs and its t-stat. Daily ICs of a `horizon`-day forward return
    overlap (each return window shares horizon-1 days with the next), so the effective sample is n / horizon, not n;
    the naive t would be inflated by sqrt(horizon) (2.2x for 5 days)."""
    horizon = horizon or config.ENSEMBLE["ic_horizon"]
    s = ic.dropna().tail(sessions)
    n = len(s)
    if n < 2:
        return {"n": n, "ic": None, "t": None, "shrunk_ic": 0.0}
    sd = float(s.std(ddof=1))
    mean = float(s.mean())
    t = mean / (sd / np.sqrt(max(n / horizon, 1.0))) if sd > 0 else 0.0
    shrink = max(0.0, 1.0 - 1.0 / t ** 2) if abs(t) > 1e-9 else 0.0
    return {"n": n, "ic": round(mean, 4), "t": round(t, 2), "shrunk_ic": round(max(mean, 0.0) * shrink, 5)}


def cap_weights(w: dict[str, float], cap: float, receivers: set[str] | None = None) -> dict[str, float]:
    """Cap each weight at `cap` and hand the excess pro rata to the names still below it. The total is preserved
    unless the caps make it unreachable (all positive receivers capped): then it is simply lower ('unallocated').
    `receivers` limits who may take the excess (default: every positive name)."""
    w = {k: max(float(v), 0.0) for k, v in w.items()}
    capped: set[str] = set()
    for _ in range(len(w) + 1):
        over = {k for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(w[k] - cap for k in over)
        for k in over:
            w[k] = cap
        capped |= over
        free = {k for k, v in w.items() if v > 0 and k not in capped and (receivers is None or k in receivers)}
        room = sum(w[k] for k in free)
        if not free or room <= 0:
            break
        for k in free:
            w[k] += excess * w[k] / room
    return {k: round(v, 4) for k, v in w.items()}


def learn_weights(d: Data, strategies: list[Strategy], panels: dict[str, pd.DataFrame] | None = None,
                  cfg: dict | None = None) -> dict:
    """Weights as of the last row of d.close. panels = precomputed z panels (shared with the backtest).

    Every scoring strategy starts with an equal prior mass of 1/N. Strategies with a full 60-session IC record
    (validated) pool their mass and split it by shrunk IC (t < 1 -> 0; a validated pool with no positive evidence
    leaves its mass UNALLOCATED, so the ensemble earns less say in Quant, it is not re-given to weak signals).
    Strategies with < 60 sessions keep the equal prior; snapshot-only data with no history gets 0.5 x 1/N capped at 5%
    (ENSEMBLE["snapshot_prior"/"snapshot_cap"]) until a measured 60-session IC replaces it."""
    cfg = {**config.ENSEMBLE, **(cfg or {})}
    fwd = forward_returns(d.close, cfg["ic_horizon"])
    table = {}
    scorers = [s for s in strategies if not s.tilt]
    for s in scorers:
        if not s.history:
            table[s.name] = {"n": 0, "ic": None, "t": None, "shrunk_ic": 0.0, "validated": False}
            continue
        z = panels[s.name] if panels is not None else s.z(d)
        st = ic_stats(rank_ic(z, fwd, 8 if s.universe == "etf" else 20), cfg["ic_sessions"])
        st["validated"] = st["n"] >= cfg["ic_sessions"]
        table[s.name] = st
    weights, mode = weights_from_table(table, {s.name: s.history for s in scorers}, cfg)
    return {"weights": weights, "table": table, "mode": mode, "allocated": round(sum(weights.values()), 4)}


def weights_from_table(table: dict, has_history: dict[str, bool], cfg: dict | None = None) -> tuple[dict, str]:
    """IC table -> weights. Shared by the live run and the backtest so both use exactly the same rule."""
    cfg = {**config.ENSEMBLE, **(cfg or {})}
    names = list(has_history)
    n = max(len(names), 1)
    validated = [k for k in names if table[k]["validated"]]
    prior = [k for k in names if not table[k]["validated"] and (has_history[k] or cfg["unvalidated"] == "equal")]
    w: dict[str, float] = {k: 0.0 for k in names}
    for k in prior:
        # snapshot-only signals (no history to measure) get half the equal prior, capped; price-based ones keep 1/N
        w[k] = 1.0 / n if has_history[k] else min(cfg["snapshot_prior"] / n, cfg["snapshot_cap"])
    ic_pos = {k: table[k]["shrunk_ic"] for k in validated if (table[k]["t"] or 0) >= cfg["min_t"] and table[k]["shrunk_ic"] > 0}
    if ic_pos:
        tot = sum(ic_pos.values())
        for k, v in ic_pos.items():
            w[k] = len(validated) / n * v / tot
    weights = cap_weights(w, cfg["weight_cap"], receivers=set(ic_pos))        # excess only flows to proven strategies
    if not validated:
        mode = f"equal-weight prior (< {cfg['ic_sessions']} sessions of IC)"
    else:
        mode = (f"IC-weighted: {len(ic_pos)}/{len(validated)} validated strategies cleared t >= {cfg['min_t']:.0f}"
                f" (shrunk IC, cap {cfg['weight_cap']:.0%}); {len(prior)} at the equal-weight prior")
    return weights, mode


def score_cross_section(zrows: dict[str, pd.Series], rawrows: dict[str, pd.Series], strategies: list[Strategy],
                        weights: dict[str, float], tickers: list[str], cfg: dict | None = None) -> dict[str, dict]:
    """Per ticker: ensemble z (weights re-normalised over the strategies that apply), score -1..1, votes, reasons."""
    cfg = {**config.ENSEMBLE, **(cfg or {})}
    by_name = {s.name: s for s in strategies}
    out = {}
    for t in tickers:
        num = den = 0.0
        votes_long = votes_short = n_apply = 0
        detail = {}
        for name, s in by_name.items():
            if s.tilt:
                continue
            z = zrows[name].get(t) if name in zrows else None
            if z is None or pd.isna(z):
                continue
            w = weights.get(name, 0.0)
            raw_val = rawrows[name].get(t)
            detail[name] = {"z": round(float(z), 2), "w": w, "reason": s.reason(t, raw_val, float(z))}
            n_apply += 1                                   # breadth check: every strategy with data votes, weighted or not
            votes_long += z >= cfg["vote_z"]
            votes_short += z <= -cfg["vote_z"]
            if w > 0:
                num += w * float(z)
                den += w
        ens_z = num / den if den > 0 else None
        # fewer weights than 1.0 in total (caps unreachable) = unallocated share -> proportionally less conviction
        score = None if ens_z is None else float(np.clip(ens_z / cfg["z_to_score"], -1, 1))
        out[t] = {"ens_z": None if ens_z is None else round(ens_z, 3), "score": None if score is None else round(score, 3),
                  "votes_long": int(votes_long), "votes_short": int(votes_short), "n_strategies": n_apply,
                  "min_votes": min(cfg["min_votes"], n_apply), "strategies": detail}
    return out
