"""Oct 23: the ensemble feeds Quant's rule_score (A3) without changing the agent's schema, role prompt or output keys.
Run:  /opt/miniconda3/bin/python3 tests/test_quant_ensemble.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from committee.quant_analyst import QuantAnalyst
from orchestrator import _votes_ok
import test_quant_models as tq


def _ens(ctx, score=0.8, allocated=1.0, tom=0.0, long=5, of=9, needed=3):
    scores = {t: {"score": score, "ens_z": score * 1.5, "votes_long": long, "votes_short": 0, "n_strategies": of,
                  "min_votes": needed, "strategies": {"xs_momentum_12_1": {"z": 1.2, "w": 0.2, "reason": "xs_momentum_12_1: 12-1m return +30% (z +1.2)"}}}
              for t in ctx["tickers"]}
    return {"scores": scores, "allocated": allocated, "tom_tilt": tom, "mode": "IC-weighted", "weights": {"xs_momentum_12_1": 0.2},
            "ic_table": {}, "skipped": [], "asof": "2026-10-22"}


def test_without_an_ensemble_the_score_is_exactly_the_legacy_one():
    q = QuantAnalyst()
    ctx = tq._ctx()
    a = q.breakdown(ctx)
    ctx["ensemble"] = None
    b = q.breakdown(ctx)
    assert {t: v["rule_score"] for t, v in a.items()} == {t: v["rule_score"] for t, v in b.items()}
    assert all("ensemble" not in v["breakdown"] for v in a.values())


def test_blend_is_half_legacy_half_ensemble_scaled_by_allocated_weight():
    q = QuantAnalyst()
    ctx = tq._ctx(n=8)
    legacy = {t: v["rule_score"] for t, v in q.breakdown(ctx).items()}
    for alloc in (1.0, 0.4, 0.0):
        ctx["ensemble"] = _ens(ctx, score=0.9, allocated=alloc)
        new = q.breakdown(ctx)
        b = config.ENSEMBLE["blend"] * alloc
        for t in ctx["tickers"]:
            raw_leg = legacy[t]                                   # vol 25% -> no vol adjustment, so rule_score == raw
            exp = (1 - b) * raw_leg + b * 0.9
            assert abs(new[t]["rule_score"] - np.clip(exp, -1, 1)) < 0.011, (alloc, t)
            assert (new[t]["breakdown"].get("ensemble") == 0.9) == True
    # no allocated weight (nothing cleared t >= 1): legacy keeps 100% of its say
    ctx["ensemble"] = _ens(ctx, score=-1.0, allocated=0.0)
    assert {t: v["rule_score"] for t, v in q.breakdown(ctx).items()} == legacy


def test_turn_of_month_is_a_uniform_tilt_not_a_vote():
    q = QuantAnalyst()
    ctx = tq._ctx(n=6)
    ctx["ensemble"] = _ens(ctx, score=0.0, allocated=0.0, tom=0.0)
    base = {t: v["rule_score"] for t, v in q.breakdown(ctx).items()}
    ctx["ensemble"] = _ens(ctx, score=0.0, allocated=0.0, tom=0.03)
    tilt = {t: v["rule_score"] for t, v in q.breakdown(ctx).items()}
    diffs = {round(tilt[t] - base[t], 2) for t in base}
    assert diffs == {0.03}, diffs                                  # same shift for every name: ranking untouched
    order = lambda d: sorted(d, key=lambda t: (-d[t], t))
    assert order(base) == order(tilt)


def test_validate_keeps_schema_and_puts_the_votes_in_the_signal_for_the_raw_output_expander():
    q = QuantAnalyst()
    ctx = tq._ctx(n=6)
    plain = q.validate({"signals": {t: {"score": 0.3} for t in ctx["tickers"]}}, ctx)
    ctx["ensemble"] = _ens(ctx)
    out = q.validate({"signals": {t: {"score": 0.3} for t in ctx["tickers"]}}, ctx)
    assert set(plain) <= set(out) and set(out) - set(plain) == {"strategy_ensemble"}
    t = ctx["tickers"][0]
    assert set(plain["signals"][t]) <= set(out["signals"][t])
    assert set(out["signals"][t]) - set(plain["signals"][t]) == {"votes", "strategies"}
    assert out["signals"][t]["votes"] == {"long": 5, "short": 0, "of": 9, "needed": 3}
    assert out["signals"][t]["score"] == 0.3                       # the LLM's score is still what Quant reports
    # the inter-agent JSON contract is untouched (role_prompt is Dev's to rewrite, so only the schema keys are pinned)
    assert all(k in q.output_schema for k in ('"signals"', '"score"', '"comment"', '"model_reliability"', '"note"'))


def test_mock_and_fallback_scores_use_the_ensemble_blend():
    q = QuantAnalyst()
    ctx = tq._ctx(n=6)
    ctx["ensemble"] = _ens(ctx, score=0.9, allocated=1.0)
    out = q.validate(q.mock(ctx), ctx)
    assert all(abs(v["score"] - v["rule_score"]) < 1e-9 or v["flags"] for v in out["signals"].values())
    rb = q.validate({}, ctx)                                       # LLM returned nothing: rule-based fallback is the blended score
    assert all(v["fallback"] and v["score"] == v["rule_score"] for v in rb["signals"].values())


def test_entry_gate_needs_three_agreeing_strategies_but_only_when_an_ensemble_exists():
    assert _votes_ok(None) and _votes_ok({}) and _votes_ok({"score": 0.5})           # no ensemble -> no extra gate
    assert _votes_ok({"votes": {"long": 3, "short": 0, "of": 9, "needed": 3}})
    assert not _votes_ok({"votes": {"long": 2, "short": 4, "of": 9, "needed": 3}})
    assert _votes_ok({"votes": {"long": 2, "short": 0, "of": 2, "needed": 2}})       # only 2 apply (e.g. an ETF): 2 of 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
