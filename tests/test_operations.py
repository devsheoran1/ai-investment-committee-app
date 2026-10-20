"""Stage 6 (Oct 20): Part D operations (D10, D11, D12, C8.5/C8.6) and the replay's no-look-ahead guarantee.
Run:  /opt/miniconda3/bin/python3 tests/test_operations.py      (~3 min: two full mock committee runs)"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config, data, database as db
import orchestrator
from committee import QuantAnalyst, SentimentAnalyst, Trader


def _tmp_fund():
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db")
    config.TRADES_CSV = tempfile.mktemp(suffix=".csv")
    os.environ["FUND_REPORT_DIR"] = tempfile.mkdtemp()
    import report; report.REPORT_DIR = os.environ["FUND_REPORT_DIR"]
    db.init_db()


def test_d10_preflight_excludes_bad_prices():
    idx = pd.bdate_range(end="2026-10-16", periods=5)
    close = pd.DataFrame({"OK": [10, 10, 10, 10, 10.5], "JUMP": [10, 10, 10, 10, 13.0], "NEG": [10, 10, 10, 10, -1.0],
                          "TQQQ": [50, 50, 50, 50, 70.0]}, index=idx, dtype=float)
    bad = orchestrator._preflight(close, ["OK", "JUMP", "NEG", "TQQQ", "GONE"])
    assert set(bad) == {"JUMP", "NEG", "GONE"}, bad               # 3x ETF: 40% move allowed (20% x leverage)


def test_c86_nav_jump_halts_execution():
    o = {"ticker": "AAA", "side": "BUY", "shares": 100, "price": 300.0, "notional": 30000.0}
    out, _ = Trader().run({"orders": [o], "trades_today": 0, "cash": 1e6, "init_mode": False,
                           "nav_jump_halt": "NAV moved +35% vs last snapshot"})
    assert out["execute"] is False and out["approved_orders"] == [] and "HALTED" in out["execution_notes"]


def _last_day():
    return data.get_market_data(data.get_universe()["symbol"].tolist())["close"].index[-1]


def test_d11_d12_failed_agents_mean_no_signal_and_no_new_trades():
    saved = (config.DB_PATH, config.TRADES_CSV)
    q_run, s_run = QuantAnalyst.run, SentimentAnalyst.run
    try:
        _tmp_fund()
        def boom(self, ctx, llm):
            raise RuntimeError("simulated timeout")
        QuantAnalyst.run = SentimentAnalyst.run = boom
        res = orchestrator.run_committee("mock", asof=_last_day())
        assert {"Quant Analyst", "Sentiment Analyst"} <= set(res["failed_agents"])      # D11: logged, no crash
        runs = db.get_runs()
        assert "Insufficient information (IPS D12)" in runs.iloc[0]["summary"], runs.iloc[0]["summary"]
        assert res["orders"] == []                                                     # init + D12: nothing new
    finally:
        QuantAnalyst.run, SentimentAnalyst.run = q_run, s_run
        config.DB_PATH, config.TRADES_CSV = saved; db._DB["path"] = config.DB_PATH


def test_replay_day_has_no_lookahead():
    """Decisions for day d must not change when every price on or after d is changed."""
    saved = (config.DB_PATH, config.TRADES_CSV)
    real_get, real_cached = data.get_market_data, data._cached
    d = _last_day() - pd.offsets.BDay(3)
    d = data.get_market_data(data.get_universe()["symbol"].tolist())["close"].index.asof(d)
    try:
        def uncached(name, fn, *a, **k):                 # models must be recomputed, not read from a cache
            return fn() if name.startswith(("lstm", "lgbm", "race")) else real_cached(name, fn, *a, **k)
        data._cached = uncached
        _tmp_fund()
        a = orchestrator.run_committee("mock", asof=d)
        base = real_get()

        def shocked(*args, **kw):
            return {k: v.mul(pd.Series(np.where(v.index < d, 1.0, 3.0), index=v.index), axis=0) for k, v in base.items()}
        data.get_market_data = shocked
        _tmp_fund()
        b = orchestrator.run_committee("mock", asof=d)
        key = lambda r: sorted((o["ticker"], o["side"], o["shares"]) for o in r["orders"])
        assert key(a) == key(b) and key(a), (key(a)[:3], key(b)[:3])
    finally:
        data.get_market_data, data._cached = real_get, real_cached
        config.DB_PATH, config.TRADES_CSV = saved; db._DB["path"] = config.DB_PATH


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
