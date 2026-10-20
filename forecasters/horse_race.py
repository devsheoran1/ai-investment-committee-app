"""
Model horse race (IPS C3.4 / C3.6, rebuilt Oct 20): LSTM vs Kronos-mini vs LightGBM vs naive 5-day momentum,
scored out of sample on the SAME 20 non-overlapping 5-day origins (offsets 0, 5, ..., 95 trading days back) and the
same tickers. Nothing here looks ahead: every forecast at an origin used only data up to that origin.

    scoreboard   per model: hit rate, mean rank IC across origins, IC t-stat (on the common origins)
    trust (C3.4) hit rate over each model's own walk-forward set (LSTM / LightGBM: last 250 trading days, Kronos: the
                 20 origins), >= 100 predictions, vs the naive "always up" baseline on the same rows:
                 < 52% or not beating the baseline -> 0%; 52-55% and beating it -> 25%; > 55% -> 50% cap;
                 plus an IC t-stat floor: t < 1 -> 0%. Total model weight capped at 50%; the rest stays on technicals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kronos_fc

H = 5
N_ORIGINS = 20
MODELS = ("lstm", "kronos", "lgbm", "naive")
LABEL = {"lstm": "LSTM", "kronos": "Kronos", "lgbm": "LightGBM", "naive": "naive-5d-mom"}


def _score(rows: pd.DataFrame, model: str) -> dict:
    d = rows.dropna(subset=[model, "y"])
    d = d[d["y"] != 0]
    if len(d) < 10:
        return {"n": int(len(d)), "hit": None, "ic": None, "ic_t": None, "origins": 0}
    ics = [g[model].corr(g["y"], method="spearman") for _, g in d.groupby("offset") if len(g) >= 8]
    ics = [x for x in ics if np.isfinite(x)]
    ic = float(np.mean(ics)) if ics else None
    sd = float(np.std(ics, ddof=1)) if len(ics) >= 3 else 0.0
    t = float(ic / (sd / np.sqrt(len(ics)))) if ic is not None and sd > 0 else None
    return {"n": int(len(d)), "origins": len(ics), "hit": float((np.sign(d[model]) == np.sign(d["y"])).mean()),
            "ic": ic, "ic_t": t}


def trust_weight(hit: float | None, baseline: float | None, n: int, ic_t: float | None) -> float:
    """IPS C3.4 tiers vs the always-up baseline, plus the t < 1 floor."""
    if hit is None or baseline is None or n < 100 or ic_t is None or ic_t < 1:
        return 0.0
    if hit < 0.52 or hit <= baseline:
        return 0.0
    return 0.25 if hit <= 0.55 else 0.50


def _trust_set(oos: list[dict], use_p_up: bool = False):
    if not oos:
        return None, None, 0
    d = pd.DataFrame(oos).dropna(subset=["y"])
    d = d[d["y"] != 0]
    if not len(d):
        return None, None, 0
    sig = np.sign(d["p_up"] - 0.5) if use_p_up and "p_up" in d else np.sign(d["pred"])
    return float((sig == np.sign(d["y"])).mean()), float((d["y"] > 0).mean()), int(len(d))


def run(close: pd.DataFrame, market: dict, tickers: list[str], lstm: dict, lgbm: dict | None,
        kronos_live: dict | None, n_origins: int = N_ORIGINS, kronos_samples: int = 1) -> dict:
    L = len(close) - 1
    offsets = [k * H for k in range(n_origins)]
    lstm_map = {(r["ticker"], r["offset"]): r["pred"] for r in lstm.get("oos_preds", [])}
    lgbm_map = {(r["ticker"], r["offset"]): r["pred"] for r in (lgbm or {}).get("oos_preds", [])}
    rows, kron_oos = [], []
    for k in offsets:
        o = L - k - H
        if o < H + 1:
            continue
        kron = {}
        if kronos_live is not None:
            try:
                kron = kronos_fc.predict({f: v.reindex(close.index) for f, v in market.items()}, tickers,
                                         end_pos=o, sample_count=kronos_samples)
            except Exception:  # noqa: BLE001
                kron = {}
        for t in tickers:
            if t not in close.columns:
                continue
            p = close[t]
            y = float(p.iloc[L - k] / p.iloc[o] - 1)
            kr = kron.get(t, {}).get("exp_ret_5d")
            rows.append({"ticker": t, "offset": k, "y": y, "lstm": lstm_map.get((t, k)), "kronos": kr,
                         "lgbm": lgbm_map.get((t, k)), "naive": float(p.iloc[o] / p.iloc[o - H] - 1)})
            if kr is not None:
                kron_oos.append({"pred": kr, "y": y})
    df = pd.DataFrame(rows)
    score = {m: _score(df, m) for m in MODELS} if not df.empty else {}

    trust = {"lstm": _trust_set(lstm.get("oos_preds", []), use_p_up=True),
             "lgbm": _trust_set((lgbm or {}).get("oos_preds", [])),
             "kronos": _trust_set(kron_oos)}
    weights = {}
    for m, (hit, base, n) in trust.items():
        if m in score:
            score[m].update(trust_hit=hit, always_up=base, n_trust=n)
        weights[m] = trust_weight(hit, base, n, (score.get(m) or {}).get("ic_t"))
    tot = sum(weights.values())
    if tot > 0.5:                                                       # C3.4: models get at most 50% in total
        weights = {m: round(w * 0.5 / tot, 3) for m, w in weights.items()}

    live = {}
    lg_live = (lgbm or {}).get("forecasts") or {}
    for t in tickers:
        a = (lstm.get("forecasts", {}).get(t) or {}).get("pred_5d_return_pct")
        pu = (lstm.get("forecasts", {}).get(t) or {}).get("p_up")
        k = (kronos_live or {}).get(t) or {}
        kr = k.get("exp_ret_5d")
        g = (lg_live.get(t) or {}).get("pred_5d_return_pct")
        signs = [np.sign(x) for x in (a, None if kr is None else kr, g) if x is not None]
        live[t] = {"lstm": a, "lstm_p_up": pu, "kronos_ret": None if kr is None else round(kr * 100, 2),
                   "kronos_p_up": k.get("p_up"), "lgbm": g,
                   "agree": None if len(signs) < 2 else bool(len(set(signs)) == 1)}
    return {"scoreboard": score, "weights": weights, "scoreboard_line": line(score, weights, kronos_live is not None),
            "live": live, "n_origins": int(df["offset"].nunique()) if len(df) else 0,
            "lgbm_importance": (lgbm or {}).get("feature_importance"),
            "oos_rows": df.round(4).to_dict("records") if len(df) <= 1500 else []}


def line(score: dict, weights: dict, kronos_on: bool) -> str:
    def fmt(m):
        s = score.get(m) or {}
        if s.get("hit") is None:
            return f"{LABEL[m]} n/a"
        t = f", t {s['ic_t']:+.1f}" if s.get("ic_t") is not None else ""
        th = f", trust-hit {s['trust_hit']:.0%} vs up {s['always_up']:.0%}" if s.get("trust_hit") is not None else ""
        return f"{LABEL[m]} hit {s['hit']:.0%} IC {s['ic']:+.3f}{t}{th}"
    parts = [fmt(m) for m in MODELS if m != "kronos" or kronos_on]
    n = max((s or {}).get("origins") or 0 for s in score.values()) if score else 0
    w = " · ".join(f"{LABEL[m]} {v:.0%}" for m, v in weights.items() if v > 0)
    note = "no model passes C3.4 (beat always-up, >=52%, t>=1): weight 0, technicals only" if not w else f"model weights {w}"
    return f"OOS ({n} origins, 5d): " + " | ".join(parts) + f" -> {note}"
