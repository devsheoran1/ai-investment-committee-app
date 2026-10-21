"""Oct 21 follow-up: API failures (429/503/timeout) for PM + CIO, books reconciliation, trade/run limits,
restart mid-run, snapshots.
Run:  /opt/miniconda3/bin/python3 tests/test_followup.py      (~4 min: two full mock committee runs)"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx, openai
import config, database as db, llm as llm_mod
import orchestrator
from committee import PortfolioManager, ChiefInvestmentOfficer
from test_operations import _last_day
from test_handoff import _with_tmp


# ----------------------------------------------------------------- API failures --- #
class _HTTPErr(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


def _timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


class _Resp:
    usage = None
    choices = [type("C", (), {"message": type("M", (), {"content": '{"decision": "approve"}'})()})()]


def _client(errors):
    """Fake OpenAI client: raises each error in turn, then answers."""
    seq = list(errors)

    def create(**kw):
        if seq:
            e = seq.pop(0)
            raise e() if callable(e) and not isinstance(e, Exception) else e
        return _Resp()
    return type("X", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})()})()})()


def _llm(errors, sleeps):
    os.environ.setdefault("GROQ_API_KEY", "test-not-a-key")
    c = llm_mod.LLMClient("groq")
    c._client = _client(errors)
    llm_mod.time.sleep = sleeps.append
    return c


def test_429_503_timeout_retry_with_bounded_backoff_then_succeed():
    real = llm_mod.time.sleep
    try:
        for name, err in (("429", lambda: _HTTPErr(429)), ("503", lambda: _HTTPErr(503)), ("timeout", _timeout)):
            for agent in ("Portfolio Manager", "CIO"):
                sleeps = []
                out, model = _llm([err, err], sleeps).chat_json("s", "u", tier="smart", agent_name=agent)
                assert out == {"decision": "approve"}, (name, out)
                assert len(sleeps) == 2 and sleeps[1] > sleeps[0] and max(sleeps) <= 60, (name, sleeps)   # backoff, capped
    finally:
        llm_mod.time.sleep = real


def test_persistent_failure_is_bounded_and_raises():
    real = llm_mod.time.sleep
    try:
        for err in (lambda: _HTTPErr(429), lambda: _HTTPErr(503), _timeout):
            sleeps = []
            try:
                _llm([err] * 50, sleeps).chat_json("s", "u", tier="smart", agent_name="CIO")
                raise AssertionError("no error raised")
            except llm_mod.LLMError as e:
                assert "failed after retries" in str(e)
            assert len(sleeps) == config.LLM_MAX_RETRIES and sum(sleeps) <= 6 * 60, sleeps
    finally:
        llm_mod.time.sleep = real


@_with_tmp
def test_pm_and_cio_down_mean_explicit_fallback_never_an_ai_decision():
    pm_run, cio_run = PortfolioManager.run, ChiefInvestmentOfficer.run

    def down(self, ctx, llm):
        raise llm_mod.LLMError("groq/openai/gpt-oss-120b failed after retries: HTTP 503")
    PortfolioManager.run = ChiefInvestmentOfficer.run = down
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        PortfolioManager.run, ChiefInvestmentOfficer.run = pm_run, cio_run
    assert {"Portfolio Manager", "CIO"} <= set(res["failed_agents"])
    msgs = db.get_messages(res["run_id"])
    for agent in ("Portfolio Manager", "CIO"):
        mine = [m for m in msgs if m["agent"] == agent]
        assert mine and all(str(m["model"]).startswith("FAILED") or m["model"] == "rules" for m in mine), \
            [(m["agent"], m["model"]) for m in mine]                                  # no message posing as an LLM call
        assert all(m["content"].get("agent_failed") for m in mine)
    assert any(m["agent"] == "CIO" and m["model"] == "rules" and "fallback" in m["title"] for m in msgs)
    assert db.get_runs().set_index("id").loc[res["run_id"], "summary"].startswith("CIO unavailable, rules fallback")


# ------------------------------------------------------- reconciliation + limits --- #
@_with_tmp
def test_init_execution_reconciles_and_is_exempt_from_the_cap():
    res = orchestrator.run_committee("mock", asof=_last_day())
    rid = res["run_id"]
    prop = db.get_proposal(rid)
    assert prop["mode"] == "init" and len(prop["orders"]) > config.MAX_TRADES_PER_DAY
    out = orchestrator.execute_run(rid, prices=prop["last_px"])
    assert len(out["trades"]) == len(prop["orders"])                         # init: exempt from the 5-trade cap
    pos, cash, px = db.get_positions(), db.get_cash(), prop["last_px"]
    assert cash >= -1e-6, cash                                               # no negative cash
    nav = cash + sum(p["shares"] * px[t] for t, p in pos.items())
    assert abs(nav - out["nav"]) < 0.01, (nav, out["nav"])                   # NAV = cash + sum(shares x price)
    fills = {o["ticker"]: o["shares"] for o in prop["orders"]}
    with open(config.TRADES_CSV) as fh:
        rows = list(csv.DictReader(fh))
    from_csv = {}
    for r in rows:
        from_csv[r["ticker"]] = from_csv.get(r["ticker"], 0) + int(r["shares"]) * (1 if r["side"] == "BUY" else -1)
    held = {t: p["shares"] for t, p in pos.items() if p["shares"]}
    assert held == {t: s for t, s in from_csv.items() if s}, "DB holdings != trades.csv"
    assert held == {t: s for t, s in fills.items()}, "share counts != fills"
    cost = sum(float(r["total"]) for r in rows) * (1 + config.TRANSACTION_COST_BPS / 1e4)
    assert abs(config.STARTING_CAPITAL - cost - cash) < 1.0, (config.STARTING_CAPITAL - cost, cash)


def _daily(orders):
    rid = db.start_run("mock", "test", "daily")
    db.finish_run(rid, "proposed", "x", proposal={"mode": "daily", "orders": orders,
                                                    "last_px": {o["ticker"]: o["price"] for o in orders}, "benchmark_px": 600.0})
    return rid


def _o(t, side="BUY", sh=10, px=100.0):
    return {"ticker": t, "side": side, "shares": sh, "price": px, "notional": sh * px, "reason": "test", "forced": False}


@_with_tmp
def test_daily_cap_five_executed_and_rejects_do_not_count():
    rid = _daily([_o(f"T{i}") for i in range(7)])
    out = orchestrator.execute_run(rid, prices={f"T{i}": 100.0 for i in range(7)})
    assert len(out["trades"]) == 5 and db.trades_today() == 5
    rid = _daily([_o("T9")])
    assert orchestrator.execute_run(rid, prices={"T9": 100.0})["trades"] == []        # cap reached for the day
    assert db.trades_today() == 5


@_with_tmp
def test_rejected_orders_are_not_trades():
    rid = _daily([_o("BIG", sh=100000, px=100.0), _o("NOPE", side="SELL"), _o("OK1")])   # no cash / nothing held
    out = orchestrator.execute_run(rid, prices={"BIG": 100.0, "NOPE": 100.0, "OK1": 100.0})
    assert [t["symbol"] for t in out["trades"]] == ["OK1"] and db.trades_today() == 1
    assert len(db.get_trades()) == 1


@_with_tmp
def test_max_three_runs_a_day_and_errored_runs_do_not_count():
    for _ in range(3):
        db.start_run("groq", "test", "daily")
    crashed = db.start_run("groq", "test", "daily")
    db.set_run_status(crashed, "error")
    assert db.runs_today() == 3
    real_llm, real_close = orchestrator.LLMClient, orchestrator.record_prior_close
    os.environ.setdefault("GROQ_API_KEY", "test-not-a-key")
    orchestrator.record_prior_close = lambda *a, **k: None
    try:
        orchestrator.run_committee("groq")                  # live path (asof = replay, which is exempt by design)
        raise AssertionError("4th run allowed")
    except orchestrator.RunLimitError:
        pass
    finally:
        orchestrator.LLMClient, orchestrator.record_prior_close = real_llm, real_close


# ---------------------------------------------------------------- restart mid-run --- #
class _Killed(BaseException):
    """Stands in for the process dying (laptop sleep, Streamlit restart)."""


@_with_tmp
def test_restart_mid_execution_never_executes_twice():
    rid = _daily([_o("A1"), _o("A2"), _o("A3")])
    real, n = db.execute_trade, {"k": 0}

    def dies_after_first(*a, **k):
        n["k"] += 1
        if n["k"] == 2:
            raise _Killed()
        return real(*a, **k)
    db.execute_trade = dies_after_first
    try:
        orchestrator.execute_run(rid, prices={"A1": 100.0, "A2": 100.0, "A3": 100.0})
    except _Killed:
        pass
    finally:
        db.execute_trade = real
    # "restart": fresh call, same DB
    db.mark_stale_runs()
    try:
        orchestrator.execute_run(rid, prices={"A1": 100.0, "A2": 100.0, "A3": 100.0})
        raise AssertionError("re-executed after restart")
    except RuntimeError as e:
        assert "no pending proposal" in str(e)
    assert len(db.get_trades()) == 1 and db.get_runs().set_index("id").loc[rid, "status"] == "executing"


@_with_tmp
def test_restart_mid_committee_run_leaves_nothing_to_execute():
    rid = db.start_run("groq", "test", "daily")                 # crashed before finish_run: no proposal
    try:
        orchestrator.execute_run(rid, prices={})
        raise AssertionError("executed a half-finished run")
    except RuntimeError:
        pass
    with db.conn() as c:
        c.execute("UPDATE runs SET started_at='2026-01-01T09:00:00+01:00' WHERE id=?", (rid,))
    db.mark_stale_runs()
    assert db.get_runs().set_index("id").loc[rid, "status"] == "error"


# ---------------------------------------------------------------------- snapshots --- #
@_with_tmp
def test_duplicate_same_day_snapshots_keep_one_row():
    db.execute_trade(None, "A1", "BUY", 10, 100.0, "t")
    db.record_snapshot(None, {"A1": 100.0}, 600.0, day="2026-10-21")
    db.record_snapshot(None, {"A1": 110.0}, 601.0, day="2026-10-21")
    s = db.get_snapshots()
    assert len(s) == 1 and abs(s.iloc[0]["total_equity"] - (db.get_cash() + 1100.0)) < 0.01
    db.record_snapshot(None, {"A1": 120.0}, 602.0, day="2026-10-21", kind="close")
    db.record_snapshot(None, {"A1": 130.0}, 603.0, day="2026-10-21")                # intraday can't overwrite the close
    s = db.get_snapshots()
    assert len(s) == 1 and s.iloc[0]["kind"] == "close" and abs(s.iloc[0]["benchmark_px"] - 602.0) < 1e-9


@_with_tmp
def test_performance_tab_placeholder_under_two_sessions():
    from streamlit.testing.v1 import AppTest
    db.execute_trade(None, "AAPL", "BUY", 10, 100.0, "t")
    db.record_snapshot(None, {"AAPL": 100.0}, 600.0, day="2026-10-21")
    db.record_snapshot(None, {"AAPL": 101.0}, 601.0, day="2026-10-21")                # same session twice
    at = AppTest.from_file(os.path.join(config.BASE_DIR, "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Performance builds up day by day" in m.value for m in at.markdown)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
