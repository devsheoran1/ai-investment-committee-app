"""Oct 21 testing handoff: idempotent execution, CIO change -> back to Risk, beta vs independent regression,
ETF notional, off-hours price label.
Run:  /opt/miniconda3/bin/python3 tests/test_handoff.py      (~2 min: one full mock committee run)"""
import datetime as dt, os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config, database as db, nyse_calendar, analytics
import orchestrator
from committee import ChiefInvestmentOfficer
from test_operations import _tmp_fund, _last_day


def _with_tmp(fn):
    def wrapped():
        saved, bk = (config.DB_PATH, config.TRADES_CSV), orchestrator._backup
        orchestrator._backup = lambda *a, **k: None
        try:
            _tmp_fund()
            fn()
        finally:
            orchestrator._backup = bk
            config.DB_PATH, config.TRADES_CSV = saved; db._DB["path"] = config.DB_PATH
    wrapped.__name__ = fn.__name__
    return wrapped


def _proposal():
    rid = db.start_run("mock", "test", "daily")
    o = {"ticker": "AAPL", "side": "BUY", "shares": 100, "price": 200.0, "notional": 20000.0, "reason": "test", "forced": False}
    db.finish_run(rid, "proposed", "x", proposal={"mode": "daily", "orders": [o], "last_px": {"AAPL": 200.0, "SPY": 600.0},
                                                    "benchmark_px": 600.0})
    return rid


@_with_tmp
def test_same_order_cannot_execute_twice():
    rid = _proposal()
    orchestrator.execute_run(rid, prices={"AAPL": 200.0})
    for _ in range(2):                                    # Streamlit rerun / second click
        try:
            orchestrator.execute_run(rid, prices={"AAPL": 200.0})
            raise AssertionError("executed twice")
        except RuntimeError as e:
            assert "no pending proposal" in str(e)
    assert len(db.get_trades()) == 1


@_with_tmp
def test_double_click_race_executes_once():
    rid = _proposal()
    errs = []

    def go():
        try:
            orchestrator.execute_run(rid, prices={"AAPL": 200.0})
        except RuntimeError as e:
            errs.append(e)
    th = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    assert len(db.get_trades()) == 1 and len(errs) == 3, (len(db.get_trades()), len(errs))


def test_needs_risk_recheck():
    a = [{"ticker": "X", "side": "BUY", "shares": 10}]
    k = orchestrator.order_key(a)
    assert not orchestrator.needs_risk_recheck(k, [dict(a[0])])
    assert orchestrator.needs_risk_recheck(k, [{**a[0], "shares": 20}])
    assert orchestrator.needs_risk_recheck(k, a + [{"ticker": "Y", "side": "BUY", "shares": 1}])


@_with_tmp
def test_cio_change_goes_back_to_risk():
    real = ChiefInvestmentOfficer.run

    def editing_cio(self, ctx, llm):                       # approves but doubles the first order after Risk signed off
        out, model = real(self, ctx, llm)
        if ctx.get("orders"):
            ctx["orders"][0]["shares"] *= 2
        return {**out, "decision": "approve"}, model
    ChiefInvestmentOfficer.run = editing_cio
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        ChiefInvestmentOfficer.run = real
    msgs = db.get_messages(res["run_id"])
    rc = [m for m in msgs if m["agent"] == "Risk Manager" and m["stage"] == "risk_recheck"]
    assert rc, "CIO changed the orders but Risk never re-checked"
    assert rc[0]["content"]["decision"] in ("approve", "reject") and rc[0]["content"]["limit_checks"]


def test_beta_matches_independent_regression():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-10-16", periods=300)
    spy = rng.normal(0.0004, 0.01, len(idx))
    rets = pd.DataFrame({"SPY": spy, "A": 1.3 * spy + rng.normal(0, 0.01, len(idx)),
                         "B": 0.6 * spy + rng.normal(0, 0.012, len(idx)), "FAS": 3 * 1.1 * spy + rng.normal(0, 0.01, len(idx))},
                        index=idx)
    px = 100 * (1 + rets).cumprod()
    w = {"A": 0.4, "B": 0.4, "FAS": 0.1}
    rr = analytics.risk_report(w, px, {}, {"FAS": "leveraged_etf"}, "neutral")
    r = px.pct_change().dropna().tail(252)                 # independent: OLS slope of daily book return on SPY, 1y
    port = sum(r[t] * x for t, x in w.items())
    slope = np.polyfit(r["SPY"], port, 1)[0]
    assert abs(rr["beta_vs_spy"] - slope) < 0.006, (rr["beta_vs_spy"], slope)
    assert abs(rr["etf_notional"]["FAS"] - 0.3) < 1e-9 and abs(rr["notional_exposure"] - (0.4 + 0.4 + 0.3)) < 1e-9


def test_off_hours_detection():
    et = nyse_calendar.ET
    assert nyse_calendar.is_open(dt.datetime(2026, 10, 21, 10, 0, tzinfo=et))
    assert not nyse_calendar.is_open(dt.datetime(2026, 10, 21, 9, 0, tzinfo=et))
    assert not nyse_calendar.is_open(dt.datetime(2026, 10, 21, 16, 0, tzinfo=et))
    assert not nyse_calendar.is_open(dt.datetime(2026, 10, 24, 11, 0, tzinfo=et))          # Saturday


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
