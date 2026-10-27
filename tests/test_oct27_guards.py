"""Oct 27: run lock, Research batching by request size, same-day caches, stale-memory scoping, CIO/PM ticker scrub.
Run:  /opt/miniconda3/bin/python3 tests/test_oct27_guards.py"""
import os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, database as db, orchestrator, daycache
from committee import EquityResearchAnalyst, SentimentAnalyst, ChiefInvestmentOfficer
from committee.base import scrub_refs
from committee.equity_research import MAX_REQUEST_CHARS
from test_handoff import _with_tmp


class _FakeLLM:
    provider = "groq"

    def __init__(self):
        self.sizes, self.calls = [], 0

    def chat_json(self, system, user, tier="fast", agent_name=None, mock_fn=None):
        self.sizes.append(len(system) + len(user)); self.calls += 1
        self.last = {"system": system, "user": user, "raw": "{}", "tokens": 100, "latency_s": 0.1}
        return mock_fn(), "fake"


def _fund(n):
    keys = ["pe_fwd", "pe_trailing", "pb", "ev_ebitda", "roe", "roa", "net_margin", "gross_margin", "op_margin", "fcf_bn",
            "debt_to_equity", "current_ratio", "rev_growth", "eps_growth", "mkt_cap_bn", "beta", "div_yield", "peg"]
    return {f"T{i:02d}": {k: round(0.1234567 * (i + j + 1), 6) for j, k in enumerate(keys)} for i in range(n)}


# ---- 5. run lock ---------------------------------------------------------------------------------- #
@_with_tmp
def test_second_run_is_refused_while_one_holds_the_lock_and_uses_no_slot():
    db.init_db()
    before = len(db.get_runs())
    assert db.acquire_run_lock("first") is True
    assert db.acquire_run_lock("second") is False and db.run_in_progress()
    try:
        orchestrator.run_committee("mock")
        raise AssertionError("expected RunBusyError")
    except orchestrator.RunBusyError as e:
        assert "did not use a run slot" in str(e)
    assert len(db.get_runs()) == before and db.runs_today() == 0          # no runs row = no slot used
    db.release_run_lock()
    assert db.run_in_progress() is None and db.acquire_run_lock("again") is True
    db.release_run_lock()


@_with_tmp
def test_a_crashed_run_does_not_block_forever():
    db.init_db()
    assert db.acquire_run_lock("crashed")
    old = (db.now() - dt.timedelta(minutes=db.LOCK_STALE_MIN + 5)).isoformat()
    with db.conn() as c:
        c.execute("UPDATE run_lock SET since=?", (old,))
    assert db.run_in_progress() is None and db.acquire_run_lock("fresh")
    db.release_run_lock()


@_with_tmp
def test_committee_refused_on_non_nyse_days_for_live_providers_only():
    import nyse_calendar
    orig = nyse_calendar.is_trading_day
    nyse_calendar.is_trading_day = lambda d: False
    try:
        try:
            orchestrator.run_committee("groq")
            raise AssertionError("expected MarketClosedError")
        except orchestrator.MarketClosedError:
            pass
        assert db.run_in_progress() is None                              # refused before the lock
    finally:
        nyse_calendar.is_trading_day = orig


# ---- 6. Research batching ------------------------------------------------------------------------- #
def test_research_46_names_split_into_requests_under_14k_chars():
    fund = _fund(46)
    ctx = {"fundamentals": fund, "research_flags": {}, "earnings": {}, "morning_brief": "x" * 900}
    assert len(EquityResearchAnalyst().system_prompt) + len(EquityResearchAnalyst().build_prompt(ctx)) > MAX_REQUEST_CHARS   # one call would 413
    llm = _FakeLLM()
    out, model = EquityResearchAnalyst().run(ctx, llm)
    assert llm.calls > 1 and max(llm.sizes) <= MAX_REQUEST_CHARS, (llm.calls, llm.sizes)
    assert set(out["ratings"]) == set(fund) and out["batches"] == llm.calls and "batches" in model
    assert out["top_pick"] in fund and out["least_preferred"] in fund


# ---- 4. same-day caches --------------------------------------------------------------------------- #
@_with_tmp
def test_research_reuses_same_day_ratings_and_only_asks_for_new_names():
    db.init_db()
    fund = _fund(10)
    ctx = {"fundamentals": fund, "research_flags": {}, "earnings": {}, "morning_brief": "", "day_cache": True, "run_id": 1}
    a, b = _FakeLLM(), _FakeLLM()
    EquityResearchAnalyst().run(ctx, a)
    out, model = EquityResearchAnalyst().run({**ctx, "run_id": 2}, b)
    assert a.calls >= 1 and b.calls == 0 and out["reused"]["from_runs"] == [1] and "reused" in model and len(out["ratings"]) == 10
    fund2 = {**fund, "NEW": dict(fund["T00"], pe_fwd=99.0)}                          # one new name -> only it is sent
    c = _FakeLLM()
    out, _ = EquityResearchAnalyst().run({**ctx, "fundamentals": fund2, "run_id": 3}, c)
    assert c.calls == 1 and "NEW" in out["ratings"] and out["reused"]["names"] == 10


@_with_tmp
def test_sentiment_scores_only_new_headlines():
    db.init_db()
    def news(*titles):
        return [{"title": t, "source": "Reuters", "ts": "2026-10-27 10:00", "finbert": 0.1} for t in titles]
    ctx = {"tickers": ["AAA", "BBB", "CCC"], "news": {"AAA": news("AAA beats"), "BBB": news("BBB falls"), "CCC": []},
           "earnings": {}, "morning_brief": "", "day_cache": True, "run_id": 1}
    a, b = _FakeLLM(), _FakeLLM()
    SentimentAnalyst().run(ctx, a)
    out, model = SentimentAnalyst().run({**ctx, "run_id": 2}, b)
    assert a.calls == 1 and b.calls == 0 and out["reused"]["names"] == 2 and "reused 2/2 names from run 1" in model
    assert out["sentiment"]["CCC"]["score"] is None                                   # no news stays n/a, never 0
    ctx3 = {**ctx, "run_id": 3, "news": {**ctx["news"], "BBB": news("BBB falls", "BBB lawsuit")}}
    c = _FakeLLM()
    out, _ = SentimentAnalyst().run(ctx3, c)
    assert c.calls == 1 and out["reused"]["names"] == 1                               # only BBB (new headline) re-scored


@_with_tmp
def test_day_cache_is_off_for_mock_and_replay_and_keyed_by_day():
    class M: provider = "mock"
    class G: provider = "groq"
    assert not daycache.enabled(M()) and daycache.enabled(G()) and not daycache.enabled(G(), replay=True)
    db.init_db()
    daycache.put("agent", "k", 1, {"out": {"a": 1}})
    assert daycache.get("agent", "k")[1] == 1 and daycache.get("agent", "other") is None
    with db.conn() as c:
        c.execute("UPDATE day_cache SET day='2000-01-01'")
    assert daycache.get("agent", "k") is None                                        # yesterday's output is never reused


# ---- 7. stale memory ------------------------------------------------------------------------------ #
@_with_tmp
def test_reflections_before_the_funds_inception_never_feed_memory():
    db.init_db()
    with db.conn() as c:
        c.execute("UPDATE simulation SET inception_date='2026-10-23' WHERE id=1")
        for d, t in (("2026-10-19T10:00:00+02:00", "IEX"), ("2026-10-24T10:00:00+02:00", "NUE")):
            c.execute("INSERT INTO reflections (decided_at, run_id, ticker, action, entry_px, eval_date, ret, spy_ret, excess_ret, "
                      "verdict, lesson) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (d, 1, t, "BUY", 10.0, "2026-10-30", .1, .0, .1, "right", "x"))
            c.execute("INSERT INTO reflections (decided_at, run_id, ticker, action, entry_px) VALUES (?,?,?,?,?)", (d, 1, t, "BUY", 10.0))
    assert list(db.get_reflections(evaluated_only=True)["ticker"]) == ["NUE"]
    assert [r["ticker"] for r in db.pending_reflections()] == ["NUE"]


def test_scrub_drops_sentences_naming_tickers_outside_the_book():
    uni = {"IEX", "PSX", "KO", "MU", "VLO", "RSI"}
    txt, bad = scrub_refs("Trim IEX on weak momentum. MU looks strong; RSI is high. KO is a defensive add.", {"MU", "VLO"}, uni)
    assert bad == ["IEX", "KO"] and "IEX" not in txt and "KO" not in txt and "MU looks strong" in txt


def test_cio_minutes_only_name_held_proposed_or_ordered_tickers():
    ctx = {"mode": "daily", "risk_decision": {"decision": "approve"}, "no_new_trades": False,
           "current_weights": {"VLO": 0.1}, "orders": [{"ticker": "MU", "side": "BUY", "forced": False}],
           "proposal": {"weights": {"MU": 0.1, "VLO": 0.05}}, "composite": {"MU": {}, "VLO": {}, "IEX": {}, "PSX": {}, "KO": {}},
           "names": {"IEX": "x", "PSX": "x", "KO": "x", "MU": "x", "VLO": "x"}}
    out = {"decision": "approve", "committee_summary": "Buy MU. Trim IEX and PSX given weak momentum. Keep KO.",
           "key_risks": ["MU volatility", "PSX oil exposure"], "dissent_noted": "", "message_to_team": "Watch VLO."}
    r = ChiefInvestmentOfficer().validate(out, ctx)
    assert r["dropped_refs"] == ["IEX", "KO", "PSX"] and "IEX" not in r["committee_summary"] and "MU" in r["committee_summary"]
    assert r["key_risks"] == ["MU volatility"] and r["message_to_team"] == "Watch VLO."




# ---- 8. failed CIO-edit re-check falls back to the Risk-approved list ------------------------------- #
def test_failed_recheck_after_cio_edit_restores_the_risk_approved_orders():
    o = lambda t, side, forced=False: {"ticker": t, "side": side, "forced": forced, "shares": 10}
    approved = [o("STOP", "SELL", True), o("IEX", "BUY"), o("PSX", "BUY")]
    edited = [o("STOP", "SELL", True), o("IEX", "BUY")]                    # CIO trimmed/dropped PSX
    kept = orchestrator.orders_after_failed_recheck(approved, edited, cio_edited=True)
    assert [x["ticker"] for x in kept] == ["STOP", "IEX", "PSX"]
    # any other post-Risk change (not a CIO edit) still keeps forced exits only
    assert [x["ticker"] for x in orchestrator.orders_after_failed_recheck(approved, edited, cio_edited=False)] == ["STOP"]


def test_orchestrator_wires_the_fallback_and_logs_it():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orchestrator.py")).read()
    assert "orders_after_failed_recheck(pre_edit, orders, cio_edited)" in src and '"cio_reverted"' in src


# ---- 4. end to end: the second run of the day reuses agent outputs, and logs tokens ----------------- #
@_with_tmp
def test_second_mock_run_reuses_same_day_outputs_and_logs_tokens():
    from test_operations import _last_day
    orig = daycache.enabled
    daycache.enabled = lambda llm, replay=False: True                  # mock + replay normally never cache; force it for the wiring test
    try:
        asof = _last_day()
        r1 = orchestrator.run_committee("mock", asof=asof)
        r2 = orchestrator.run_committee("mock", asof=asof)
    finally:
        daycache.enabled = orig
    m1, m2 = db.get_messages(r1["run_id"]), db.get_messages(r2["run_id"])
    reused = {m["agent"]: m["model"] for m in m2 if str(m["model"]).startswith("reused") or "reused" in str(m["model"])}
    assert not any("reused" in str(m["model"]) for m in m1)
    assert {"Macro Strategist", "Equity Research Analyst", "Quant Analyst", "Portfolio Manager", "CIO"} <= set(reused)   # replay has no news: Sentiment has nothing to reuse, reused
    tok = next(m["content"] for m in m2 if m["stage"] == "tokens")
    assert tok["reused"] and "tokens_this_run" in tok
    macro2 = next(m["content"] for m in m2 if m["agent"] == "Macro Strategist")
    macro1 = next(m["content"] for m in m1 if m["agent"] == "Macro Strategist")
    assert macro2["regime"] == macro1["regime"]                          # rules fields re-validated, not copied blindly


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
