"""
Reflection memory (TradingAgents-style): the committee grades its own past calls and learns from them.

    record     database.execute_trade(meta=...) stores every executed trade as a call: thesis, invalidation,
               entry price/date, regime, sector, the signals behind it
    evaluate   at the start of each run, calls whose 5-day horizon has elapsed are scored vs SPY.
               ONLY prices <= the run's as-of date are used (replay-safe, no look-ahead)
    lessons    one batched cheap-model call writes a 1-2 sentence lesson per graded call
    retrieve   the PM and CIO get <= 8 lessons (<= ~600 tokens): same ticker > same sector > same
               regime > most recent, plus the committee's running hit rate
    invalidated  holdings whose invalidation price has been hit are flagged; the PM must address them

All maths lives here; the LLM only phrases the lesson.
"""
from __future__ import annotations

import json

import pandas as pd

import config
import database as db

FLAT_BAND = 0.01          # |excess return| below 1% = "flat" (neither right nor wrong)
MAX_LESSONS = 8
MAX_CHARS = 2400          # ~600 tokens
DEFAULT_INVALIDATION_PCT = 0.05   # tighter than the -10% stop-loss so it flags before the forced exit


def verdict_for(action: str, excess: float) -> str:
    """BUY is right if it beat SPY; SELL is right if the stock then lagged SPY."""
    signed = excess if action == "BUY" else -excess
    return "right" if signed > FLAT_BAND else "wrong" if signed < -FLAT_BAND else "flat"


def evaluate(close: pd.DataFrame, asof=None) -> list[dict]:
    """Score every pending call whose horizon has elapsed by `asof` (default: last row). No LLM, no writes."""
    if asof is not None:
        close = close[close.index <= pd.Timestamp(asof)]      # the look-ahead guard
    out = []
    if config.BENCHMARK not in close.columns or close.empty:
        return out
    for r in db.pending_reflections():
        t = r["ticker"]
        if t not in close.columns or not r.get("entry_date"):
            continue
        pos = int(close.index.searchsorted(pd.Timestamp(r["entry_date"])))
        epos = pos + int(r["horizon_days"] or 5)
        if pos >= len(close) or epos > len(close) - 1:
            continue                                            # horizon not yet elapsed as of `asof`
        px_end, spy0, spy1 = float(close[t].iloc[epos]), float(close[config.BENCHMARK].iloc[pos]), \
            float(close[config.BENCHMARK].iloc[epos])
        if not all(x == x and x > 0 for x in (px_end, spy0, spy1, r["entry_px"])):
            continue
        ret, spy_ret = px_end / r["entry_px"] - 1, spy1 / spy0 - 1
        sig = json.loads(r["signals"]) if r.get("signals") else {}
        out.append({**r, "signals": sig, "eval_date": str(close.index[epos].date()), "ret": ret, "spy_ret": spy_ret,
                    "excess_ret": ret - spy_ret, "verdict": verdict_for(r["action"], ret - spy_ret)})
    return out


def save(graded: list[dict], lessons: dict) -> None:
    for g in graded:
        l = lessons.get(str(g["id"])) or {}
        db.finish_reflection(g["id"], g["eval_date"], g["ret"], g["spy_ret"], g["excess_ret"], g["verdict"],
                             l.get("lesson") or deterministic_lesson(g), l.get("tags") or default_tags(g))


def default_tags(g: dict) -> list[str]:
    return [str(x) for x in (g.get("regime"), g.get("sector"), g["verdict"]) if x]


def deterministic_lesson(g: dict) -> str:
    """Fallback and mock lesson: states the facts, no invented causality."""
    return (f"{g['verdict'].upper()}: {g['action']} {g['ticker']} in {g.get('regime') or 'n/a'} returned "
            f"{g['ret']:+.1%} vs SPY {g['spy_ret']:+.1%} ({g['excess_ret']:+.1%}) over {g['horizon_days']}d; "
            f"thesis was '{(g.get('thesis') or '')[:70]}'.")


def track_record() -> dict:
    ev = db.get_reflections(evaluated_only=True)
    if ev.empty:
        return {"calls": 0}
    n_r, n_w, n_f = [(ev["verdict"] == v).sum() for v in ("right", "wrong", "flat")]
    last = ev.head(10)
    lr, lw = (last["verdict"] == "right").sum(), (last["verdict"] == "wrong").sum()
    return {"calls": int(len(ev)), "right": int(n_r), "wrong": int(n_w), "flat": int(n_f),
            "hit_rate": round(float(n_r / (n_r + n_w)), 2) if n_r + n_w else None,
            "last10_hit_rate": round(float(lr / (lr + lw)), 2) if lr + lw else None,
            "avg_excess_ret": round(float(ev["excess_ret"].mean()), 4)}


def lessons_for(tickers: set[str], sectors: set[str], regime: str) -> list[str]:
    """<= 8 lessons within ~600 tokens. Priority: same ticker (3) > same sector (2) > same regime (1) > recent."""
    ev = db.get_reflections(evaluated_only=True)        # newest first
    if ev.empty:
        return []
    ev = ev[ev["lesson"].notna()].copy()
    ev["score"] = (3 * ev["ticker"].isin(tickers) + 2 * ev["sector"].isin(sectors) + 1 * (ev["regime"] == regime))
    ev["recency"] = range(len(ev), 0, -1)               # newest = largest
    ev = ev.sort_values(["score", "recency"], ascending=False)
    out, used = [], 0
    for r in ev.itertuples():
        line = f"[{r.ticker} {r.action} {r.eval_date} {r.verdict} {r.excess_ret:+.1%} {r.regime or ''}] {str(r.lesson)[:260]}"
        if len(out) >= MAX_LESSONS or used + len(line) > MAX_CHARS:
            break
        out.append(line)
        used += len(line)
    return out


def invalidated(positions: dict, last_px: dict) -> list[dict]:
    """Holdings whose latest BUY thesis has an invalidation price that the market has now hit."""
    hit = []
    for t in positions:
        r = db.latest_buy_reflection(t)
        px = last_px.get(t)
        if r and r.get("invalidation_px") and px and px <= r["invalidation_px"]:
            hit.append({"ticker": t, "thesis": (r.get("thesis") or "")[:140],
                        "invalidation": (r.get("invalidation") or "")[:120],
                        "entry_px": round(r["entry_px"], 2), "invalidation_px": round(r["invalidation_px"], 2),
                        "last_px": round(px, 2), "since_entry": round(px / r["entry_px"] - 1, 4)})
    return hit


def trade_notes(order: dict, pm_notes: dict | None, composite: dict | None) -> dict:
    """Thesis + invalidation for an order. PM-supplied when valid, otherwise a deterministic default,
    so every executed trade carries both."""
    px, n = order["price"], (pm_notes or {})
    if order["side"] == "BUY":
        v = n.get("invalidation_px")
        inv_px = float(v) if isinstance(v, (int, float)) and 0.80 * px <= v <= 0.99 * px else round(px * (1 - DEFAULT_INVALIDATION_PCT), 2)
        thesis = n.get("thesis") or (order.get("reason") or "")[:200]
        inv = n.get("invalidation") or f"closes below ${inv_px:.2f} ({inv_px / px - 1:+.0%} from entry) or composite falls under 5/10"
    else:
        inv_px = round(px * 1.05, 2)
        thesis = n.get("thesis") or (order.get("reason") or "")[:200]
        inv = n.get("invalidation") or f"stock trades above ${inv_px:.2f} (+5%) within 5 days: the exit was wrong"
    return {"thesis": str(thesis)[:300], "invalidation": str(inv)[:240], "invalidation_px": round(inv_px, 2),
            "signals": {k: (composite or {}).get(k) for k in ("composite_10", "quant", "sentiment", "research")}}
