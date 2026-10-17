"""Offline tests for reflection memory: no look-ahead, retrieval priority and caps.
Run:  /opt/miniconda3/bin/python3 tests/test_reflection.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import database as db, reflection


def _frame(n=40):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"SPY": 400 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                         "AAA": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))}, index=idx)


def _fresh_db():
    db._DB["path"] = tempfile.mktemp(suffix=".db"); db.init_db()


def test_no_lookahead():
    _fresh_db(); close = _frame(); pos = 30
    db.execute_trade(1, "AAA", "BUY", 100, float(close.AAA.iloc[pos]), "t",
                     meta={"thesis": "x", "invalidation": "y", "invalidation_px": 1.0,
                           "entry_date": str(close.index[pos].date()), "regime": "neutral", "sector": "Tech", "signals": {}})
    assert reflection.evaluate(close, asof=close.index[pos + 4]) == [], "graded before the 5-day horizon elapsed"
    g = reflection.evaluate(close, asof=close.index[pos + 5])
    assert len(g) == 1 and g[0]["eval_date"] == str(close.index[pos + 5].date())
    wrecked = close.copy(); wrecked.iloc[pos + 6:] = 1e6          # future prices must be invisible
    g2 = reflection.evaluate(wrecked, asof=close.index[pos + 5])
    assert g2[0]["ret"] == g[0]["ret"] and g2[0]["spy_ret"] == g[0]["spy_ret"]
    # a later as-of date does not change the result for an already-elapsed horizon
    g3 = reflection.evaluate(close, asof=close.index[-1])
    assert g3[0]["ret"] == g[0]["ret"]


def test_retrieval_priority_and_caps():
    _fresh_db()
    rows = [("AAA", "Tech", "risk_on"), ("BBB", "Energy", "neutral"), ("CCC", "Tech", "risk_off")] + \
           [(f"Z{i}", "Other", "risk_off") for i in range(20)]
    for i, (t, sec, reg) in enumerate(rows):
        db.execute_trade(1, t, "BUY", 10, 10.0, "r", meta={"thesis": "th", "invalidation": "iv", "invalidation_px": 9.0,
                         "entry_date": "2026-09-01", "regime": reg, "sector": sec, "signals": {}})
    for r in db.pending_reflections():
        db.finish_reflection(r["id"], "2026-09-08", 0.02, 0.01, 0.01, "right", "L " + r["ticker"] + " " + "x" * 300, ["t"])
    out = reflection.lessons_for({"BBB"}, {"Tech"}, "neutral")
    assert out[0].startswith("[BBB") and "[AAA" in out[1] or "[CCC" in out[1], out[:3]   # ticker, then sector
    assert len(out) <= reflection.MAX_LESSONS and sum(map(len, out)) <= reflection.MAX_CHARS


if __name__ == "__main__":
    test_no_lookahead(); test_retrieval_priority_and_caps(); print("ok")
