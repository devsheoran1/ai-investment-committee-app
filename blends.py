"""
Candidate agent blends (IPS C5.2), scored on live data. SCORING ONLY: no extra fund, no trades, nothing the PM or UI reads.

Every committee run re-ranks the shortlist under each preset in config.SIGNAL_BLENDS from the SAME agent scores the fund used,
logs each blend's top 10 to reports/blend_log.csv, and later fills in what those names did over the next 5 sessions
(close of the decision day -> close 5 sessions later, raw and minus SPY). `summary()` compares blends once data exists.
Live fund only: tests and replays never write (same guard as the snapshot z-log).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config

COLS = ["asof", "blend", "rank", "ticker", "type", "composite_10", "fwd5_pct", "fwd5_vs_spy_pct"]
HORIZON = 5
TOP = 10


def _enabled() -> bool:
    return os.path.abspath(config.DB_PATH) == os.path.join(config.BASE_DIR, "finance.db") or "FUND_BLEND_LOG" in os.environ


def composite(parts: dict, W: dict) -> float:
    """Same rule as analytics.composite_scores: agents with no signal drop out and the rest re-normalise."""
    avail = {k: v for k, v in parts.items() if v is not None}
    tot = sum(W[k] for k in avail) or 1.0
    return float(np.clip(sum(W[k] * v for k, v in avail.items()) / tot, -1, 1))


def rank_all(comp: dict) -> dict[str, list[tuple[str, float, str]]]:
    """{blend: [(ticker, composite_10, type)] best first} from analytics.composite_scores output (per-agent scores kept)."""
    import analytics
    out = {}
    for name, W in config.SIGNAL_BLENDS.items():
        rows = []
        for t, c in comp.items():
            parts = {k: c.get(k) for k in ("quant", "sentiment", "research", "macro")}
            rows.append((t, analytics.to_10(composite(parts, W)), c.get("type", "stock")))
        out[name] = sorted(rows, key=lambda r: -r[1])
    return out


def log_run(comp: dict, asof: str) -> int:
    """Replace today's rows (a day can have up to 3 runs; the last one wins) with each blend's top 10."""
    if not _enabled() or not comp:
        return 0
    rows = [(asof, b, i + 1, t, ty, round(c10, 2), np.nan, np.nan)
            for b, ranked in rank_all(comp).items() for i, (t, c10, ty) in enumerate(ranked[:TOP])]
    new = pd.DataFrame(rows, columns=COLS)
    path = config.BLEND_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[old["asof"] != asof], new], ignore_index=True)
    new.to_csv(path, index=False)
    return len(rows)


def score(close: pd.DataFrame, bench: str | None = None) -> int:
    """Fill fwd5 for logged rows whose 5th session after `asof` now exists in `close`. Returns rows filled."""
    path = config.BLEND_LOG
    if not _enabled() or not os.path.exists(path):
        return 0
    bench = bench or config.BENCHMARK
    df = pd.read_csv(path)
    todo = df["fwd5_pct"].isna()
    idx = close.index
    n = 0
    for asof in df.loc[todo, "asof"].unique():
        pos = int(idx.searchsorted(pd.Timestamp(asof)))
        if pos >= len(idx) or idx[pos] != pd.Timestamp(asof) or pos + HORIZON >= len(idx):
            continue                                                  # decision day not in the data / 5 sessions not elapsed
        r = close.iloc[pos + HORIZON] / close.iloc[pos] - 1
        m = todo & (df["asof"] == asof)
        df.loc[m, "fwd5_pct"] = (df.loc[m, "ticker"].map(r) * 100).round(3)
        df.loc[m, "fwd5_vs_spy_pct"] = df.loc[m, "fwd5_pct"] - float(r.get(bench, np.nan)) * 100
        n += int(m.sum())
    if n:
        df.to_csv(path, index=False)
    return n


def summary(path: str | None = None) -> pd.DataFrame:
    """Per blend: days scored, mean 5-day return of its top 10, mean excess vs SPY, hit rate (share of top-10 names beating SPY)."""
    path = path or config.BLEND_LOG
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path).dropna(subset=["fwd5_pct"])
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["blend", "asof"]).agg(r=("fwd5_pct", "mean"), x=("fwd5_vs_spy_pct", "mean"), h=("fwd5_vs_spy_pct", lambda s: (s > 0).mean()))
    out = g.groupby("blend").agg(days=("r", "size"), top10_5d_ret_pct=("r", "mean"), excess_vs_spy_pct=("x", "mean"), hit_rate=("h", "mean"))
    return out.round(3).sort_values("excess_vs_spy_pct", ascending=False)
