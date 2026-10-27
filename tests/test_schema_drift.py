"""Oct 23: run-3 Quant bug. The LLM returned flat numbers ({"NUE": -0.1}) instead of {"NUE": {"score": ..}} plus
tickers outside the shortlist -> validate() crashed -> Quant FAILED -> every Q = 0.0. Same drift for the other
analysts, and the CIO on init day (may trim/drop, cannot decline; minutes match the decision).
Run:  /opt/miniconda3/bin/python3 tests/test_schema_drift.py      (~3 min: two full mock committee runs)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database as db, orchestrator
from committee import QuantAnalyst, SentimentAnalyst, EquityResearchAnalyst, MacroStrategist, ChiefInvestmentOfficer
from committee.base import entry
from test_operations import _last_day
from test_handoff import _with_tmp

# Run-3 shape (the original raw output was lost when finance.db was reset; reconstructed from the bug report)
RUN3_QUANT = {"signals": {"NUE": -0.1, "ZZZZ": 0.4, "NOTREAL": "0.9"}, "model_reliability": "medium",
              "note": "momentum strong in materials"}


def test_entry_accepts_both_shapes():
    assert entry({"NUE": -0.1}, "NUE") == {"score": -0.1}
    assert entry({"NUE": {"score": -0.1, "comment": "x"}}, "NUE")["score"] == -0.1
    assert entry({"NUE": "0.3"}, "NUE") == {"score": 0.3}
    assert entry({"NUE": [1, 2]}, "NUE") == {} and entry(["NUE"], "NUE") == {} and entry(None, "NUE") == {}


def _drift(ctx_box):
    def quant(self, ctx):
        ctx_box["q"] = ctx
        t0 = ctx["tickers"][0]
        return {**RUN3_QUANT, "signals": {**RUN3_QUANT["signals"], t0: 0.55}}

    def sent(self, ctx):
        return {"sentiment": {ctx["tickers"][0]: 0.3, "ZZZZ": 1, ctx["tickers"][1]: ["bad"]}, "market_mood": 7}

    def research(self, ctx):
        t = next(iter(ctx["fundamentals"]))
        ctx_box["r"] = t
        return {"ratings": {t: 0.5, "ZZZZ": {"rating": "buy"}}, "top_pick": "ZZZZ", "least_preferred": ["x"]}

    def macro(self, ctx):
        return {"tilt_override": "none", "watch_list": "oil", "confidence": "high", "rationale": None}
    return quant, sent, research, macro


@_with_tmp
def test_run3_shape_does_not_fail_quant_and_missing_names_use_rule_score():
    box, saved = {}, (QuantAnalyst.mock, SentimentAnalyst.mock, EquityResearchAnalyst.mock, MacroStrategist.mock)
    QuantAnalyst.mock, SentimentAnalyst.mock, EquityResearchAnalyst.mock, MacroStrategist.mock = _drift(box)
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        QuantAnalyst.mock, SentimentAnalyst.mock, EquityResearchAnalyst.mock, MacroStrategist.mock = saved
    assert not set(res["failed_agents"]) & {"Quant Analyst", "Sentiment Analyst", "Equity Research Analyst",
                                             "Macro Strategist"}, res["failed_agents"]
    msgs = {m["agent"]: m["content"] for m in db.get_messages(res["run_id"])}
    q = msgs["Quant Analyst"]["signals"]
    t0 = box["q"]["tickers"][0]
    assert "ZZZZ" not in q and "NOTREAL" not in q                                    # outside the shortlist: ignored
    assert q[t0]["fallback"] is False and abs(q[t0]["score"] - 0.55) < 0.3            # flat number accepted (RSI rule may halve)
    fb = [t for t, v in q.items() if v["fallback"]]
    assert fb and all(q[t]["comment"] == "Quant: rule-based fallback" for t in fb)
    assert all(q[t]["score"] * q[t]["rule_score"] >= 0 for t in fb)                   # rule_score (RSI rules may cut it)
    assert sum(1 for t in fb if abs(q[t]["score"]) > 1e-9) >= len(fb) // 2             # not a wall of 0.0
    # Quant LLM call fails outright -> orchestrator.call() runs validate({}): every name on its rule_score, flagged
    out = QuantAnalyst().validate({}, box["q"])
    assert all(v["fallback"] and v["comment"] == "Quant: rule-based fallback" for v in out["signals"].values())
    assert any(abs(v["score"]) > 1e-9 for v in out["signals"].values())
    assert "rule-based fallback" in msgs["Quant Analyst"]["quant_fallback"]
    assert any(abs(v["score"]) > 1e-9 for v in q.values())                            # not all Q = 0.0 any more
    s = msgs["Sentiment Analyst"]["sentiment"]
    assert "ZZZZ" not in s
    r = msgs["Equity Research Analyst"]
    assert r["top_pick"] is None and r["least_preferred"] is None and r["ratings"][box["r"]]["score"] == 0.5
    m = msgs["Macro Strategist"]
    assert m["tilt_override"] is None and m["watch_list"] == []


@_with_tmp
def test_cio_init_cannot_decline_minutes_match_and_trim_drop_go_back_to_risk():
    real = ChiefInvestmentOfficer.mock
    box = {}

    def declines(self, ctx):
        buys = [o["ticker"] for o in ctx["orders"] if o["side"] == "BUY" and not o.get("forced")]
        box["buys"] = buys
        return {"decision": "reject", "committee_summary": "We reject this book: too much energy.",
                "key_risks": "not a list", "dissent_noted": "", "message_to_team": "",
                "drop": [buys[-1], "ZZZZ"], "trim": {buys[0]: 0.5, "ZZZZ": 0.5, buys[1]: "abc"}}
    ChiefInvestmentOfficer.mock = declines
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        ChiefInvestmentOfficer.mock = real
    msgs = db.get_messages(res["run_id"])
    cio = next(m["content"] for m in msgs if m["agent"] == "CIO")
    assert cio["decision"] == "approve" and cio["llm_decision"] == "reject"
    assert cio["committee_summary"].startswith("Decision: APPROVE") and "We reject" not in cio["committee_summary"]
    assert "We reject" in cio["dissent_noted"]
    summary = db.get_runs().set_index("id").loc[res["run_id"], "summary"]
    assert summary.startswith("Decision: APPROVE"), summary
    assert cio["drop"] == [box["buys"][-1]] and cio["trim"] == {box["buys"][0]: 0.5}
    edits = next(m["content"] for m in msgs if m["stage"] == "cio_edits")
    assert any("dropped" in x for x in edits["applied_changes"]) and any("trimmed" in x for x in edits["applied_changes"])
    final = {o["ticker"] for o in res["orders"]}
    # the drop is honoured, or it broke a hard limit on Risk's re-check and the Risk-approved list was restored (Oct 27, logged)
    reverted = any(m["stage"] == "cio_reverted" for m in msgs)
    assert box["buys"][-1] not in final or len(box["buys"]) <= 10 or reverted
    assert not reverted or box["buys"][-1] in final
    assert any(m["stage"] == "risk_recheck" for m in msgs), "CIO edits did not go back to Risk"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
