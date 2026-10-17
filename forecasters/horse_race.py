"""
Model horse race (Quant desk): LSTM vs Kronos-mini vs a naive baseline, scored out of sample on the
SAME rolling origins and the SAME tickers.

    origin offsets k = 0, 5, 10, ... : the forecast whose 5-day window ends k trading days before today
    (non-overlapping windows, all already realised, so nothing here looks ahead)

    LSTM     predictions held out of its training window (forecaster.forecast -> oos_preds)
    Kronos   re-forecast at each origin using only rows <= origin (forecasters.kronos_fc.predict)
    naive    sign/size of the trailing 5-day return (pure momentum)

Metrics per model: directional hit rate, mean rank IC across origins, IC t-stat (mean / s.e. over origins).
Blend weights = max(mean IC, 0), normalised, over LSTM and Kronos only, so a model with negative IC gets
zero weight. With ~6 origins the t-stats are weak by construction; the scoreboard says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kronos_fc

H = 5
MODELS = ("lstm", "kronos", "naive")
LABEL = {"lstm": "LSTM", "kronos": "Kronos", "naive": "naive-5d-mom"}


def _score(rows: pd.DataFrame, model: str) -> dict:
    d = rows.dropna(subset=[model, "y"])
    d = d[d["y"] != 0]
    if len(d) < 10:
        return {"n": int(len(d)), "hit": None, "ic": None, "ic_t": None}
    ics = [g[model].corr(g["y"], method="spearman") for _, g in d.groupby("offset") if len(g) >= 8]
    ics = [x for x in ics if np.isfinite(x)]
    ic = float(np.mean(ics)) if ics else None
    t = float(np.mean(ics) / (np.std(ics, ddof=1) / np.sqrt(len(ics)))) if len(ics) >= 3 and np.std(ics, ddof=1) > 0 else None
    return {"n": int(len(d)), "origins": len(ics), "hit": float((np.sign(d[model]) == np.sign(d["y"])).mean()),
            "ic": ic, "ic_t": t}


def run(close: pd.DataFrame, market: dict, tickers: list[str], lstm_oos: list[dict],
        lstm_live: dict, kronos_live: dict | None, n_origins: int = 6, sample_count: int = 3) -> dict:
    """close: the exact frame the LSTM was trained on. Returns scoreboard, blend weights and a per-ticker view."""
    L = len(close) - 1
    offsets = [k * H for k in range(n_origins)]
    px = close
    rows = []
    lstm_map = {(r["ticker"], r["offset"]): r["pred"] for r in lstm_oos}
    for k in offsets:
        o = L - k - H                                     # origin row; its 5d window ends at row L-k
        if o < H + 1:
            continue
        kron = {}
        if kronos_live is not None:
            try:
                kron = kronos_fc.predict({f: v.reindex(px.index) for f, v in market.items()}, tickers,
                                         end_pos=o, sample_count=sample_count)
            except Exception:  # noqa: BLE001
                kron = {}
        for t in tickers:
            if t not in px.columns:
                continue
            p = px[t]
            rows.append({"ticker": t, "offset": k,
                         "y": float(p.iloc[L - k] / p.iloc[o] - 1),
                         "lstm": lstm_map.get((t, k)),
                         "kronos": kron.get(t, {}).get("exp_ret_5d"),
                         "naive": float(p.iloc[o] / p.iloc[o - H] - 1)})
    df = pd.DataFrame(rows)
    score = {m: _score(df, m) for m in MODELS} if not df.empty else {}
    # blend weights: IC floored at 0 (LSTM and Kronos only)
    raw = {m: max(score.get(m, {}).get("ic") or 0.0, 0.0) for m in ("lstm", "kronos")}
    tot = sum(raw.values())
    weights = {m: (v / tot if tot > 0 else 0.0) for m, v in raw.items()}

    # live view per ticker + agreement flag
    live = {}
    for t in tickers:
        a = (lstm_live.get(t) or {}).get("pred_5d_return_pct")
        k = (kronos_live or {}).get(t) or {}
        kr = k.get("exp_ret_5d")
        agree = None if a is None or kr is None else bool(np.sign(a) == np.sign(kr))
        live[t] = {"lstm": a, "kronos_ret": None if kr is None else round(kr * 100, 2),
                   "kronos_p_up": k.get("p_up"), "agree": agree}
    return {"scoreboard": score, "weights": weights, "scoreboard_line": line(score, weights, kronos_live is not None),
            "live": live, "n_origins": len([k for k in offsets if L - k - H >= H + 1]),
            "oos_rows": df.round(4).to_dict("records") if len(df) <= 400 else []}


def line(score: dict, weights: dict, kronos_on: bool) -> str:
    def fmt(m):
        s = score.get(m) or {}
        if s.get("hit") is None:
            return f"{LABEL[m]} n/a"
        t = f", t {s['ic_t']:+.1f}" if s.get("ic_t") is not None else ""
        return f"{LABEL[m]} hit {s['hit']:.0%} IC {s['ic']:+.3f}{t}"
    parts = [fmt(m) for m in MODELS if m != "kronos" or kronos_on]
    n = (score.get("lstm") or {}).get("origins") or (score.get("naive") or {}).get("origins") or 0
    w = " ".join(f"{LABEL[m]} {v:.0%}" for m, v in weights.items() if m != "kronos" or kronos_on)
    note = "no model has positive OOS IC: lean on technicals" if sum(weights.values()) == 0 else f"blend weights {w}"
    return f"OOS ({n} origins, 5d): " + " | ".join(parts) + f" -> {note}"
