"""
Quiet live tracker (Oct 23): do the three Quant score variants rank the shortlist's forward returns differently?
    ensemble-only  breakdown.ensemble (multi-strategy ensemble score)
    deployed       the Quant score the committee actually used (legacy/ensemble blend, LLM-owned, after the C3.3 rules)
    legacy         breakdown.legacy (technical + model score, before the ensemble blend)
For every committee run since inception: forward return from the close the run decided on (last completed session at
run start) to the latest close, rank IC of each variant against it, and the top-third basket's return minus the
shortlist average. Averaged over runs. A few days of data is noise; this is a diary, not a verdict.
Pure function of the database and the close panel, no LLM.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

import nyse_calendar as cal

VARIANTS = ("ensemble", "deployed", "legacy")


def _scores(sig: dict) -> dict[str, dict[str, float]]:
    out = {v: {} for v in VARIANTS}
    for t, s in (sig or {}).items():
        bd = s.get("breakdown") or {}
        for v, val in (("ensemble", bd.get("ensemble")), ("deployed", s.get("score")), ("legacy", bd.get("legacy"))):
            if val is not None:
                out[v][t] = float(val)
    return out


def compute(runs: list[dict], close: pd.DataFrame) -> dict | None:
    """runs = [{"started_at": iso, "signals": Quant signals dict}], close = date x ticker panel. None until measurable."""
    if close is None or close.empty:
        return None
    last = close.index[-1]
    rows = {v: [] for v in VARIANTS}
    n_runs = 0
    for r in runs:
        base = pd.Timestamp(cal.last_completed_session(pd.Timestamp(r["started_at"]).to_pydatetime()
                                                       if pd.Timestamp(r["started_at"]).tzinfo else
                                                       pd.Timestamp(r["started_at"]).tz_localize(cal.MADRID).to_pydatetime()))
        if base not in close.index or last <= base:
            continue
        fwd = close.loc[last] / close.loc[base] - 1
        sc = _scores(r.get("signals"))
        used = False
        for v in VARIANTS:
            s = pd.Series(sc[v]).reindex(fwd.index).dropna()
            f = fwd.reindex(s.index).dropna()
            s = s.reindex(f.index)
            if len(s) < 6 or s.nunique() < 3:
                continue
            k = max(2, len(s) // 3)
            rows[v].append({"ic": float(s.rank().corr(f.rank())), "top": float(f[s.nlargest(k).index].mean() - f.mean())})
            used = True
        n_runs += used
    if not n_runs:
        return None
    res = {v: {"ic": sum(x["ic"] for x in rows[v]) / len(rows[v]), "top_excess": sum(x["top"] for x in rows[v]) / len(rows[v]),
               "n": len(rows[v])} for v in VARIANTS if rows[v]}
    return {"n_runs": n_runs, "asof": last.date().isoformat(), "variants": res}


def from_db(db, close: pd.DataFrame) -> dict | None:
    runs = db.get_runs()
    if runs is None or runs.empty:
        return None
    seen = []
    for r in runs.sort_values("id").itertuples():
        if getattr(r, "mode", "") == "recheck":
            continue
        q = next((m["content"] for m in db.get_messages(int(r.id)) if m["agent"] == "Quant Analyst"), None)
        if q and q.get("signals"):
            seen.append({"started_at": r.started_at, "signals": q["signals"]})
    return compute(seen, close)


def line(res: dict | None) -> str:
    if not res:
        return ("Live tracker (ensemble-only vs deployed vs legacy): starts once a close after the first run is in; "
                "until then there is nothing to measure.")
    bits = [f"{v} IC {x['ic']:+.2f}, top third {x['top_excess']:+.1%} vs shortlist"
            for v, x in res["variants"].items()]
    return (f"Live tracker, since inception ({res['n_runs']} run(s), forward return to {res['asof']}): " + " · ".join(bits)
            + ". A few sessions is noise: read it as a diary, not evidence.")
