"""Oct 23: IC-weighted ensemble (A2). Shrunk rolling OOS IC, floor 0, cap 20%, t < 1 -> 0, equal prior if < 60 sessions,
walk-forward without look-ahead, live == backtest weights.
Run:  /opt/miniconda3/bin/python3 tests/test_ensemble.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config, strategies
from strategies import ensemble as En
from strategies import registry
import synth

CFG = config.ENSEMBLE


def test_rank_ic_is_one_for_a_perfect_signal_and_minus_one_for_its_inverse():
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(size=(5, 40)), columns=[f"T{i}" for i in range(40)])
    assert np.allclose(En.rank_ic(fwd * 3 + 1, fwd), 1.0)
    assert np.allclose(En.rank_ic(-fwd, fwd), -1.0)
    assert En.rank_ic(fwd.iloc[:, :10], fwd).isna().all()                  # < 20 names -> no IC


def test_ic_t_stat_uses_the_overlap_adjusted_sample():
    rng = np.random.default_rng(1)
    ic = pd.Series(0.03 + rng.normal(0, 0.1, 60))
    st = En.ic_stats(ic, 60, horizon=5)
    naive = ic.mean() / (ic.std(ddof=1) / np.sqrt(60))
    assert abs(st["t"] - naive / np.sqrt(5)) < 0.02                         # 5-day windows overlap: n_eff = 12
    assert st["n"] == 60


def test_shrinkage_zero_below_t1_and_monotone_above():
    base = pd.Series(np.tile([0.05, -0.05], 30))                           # mean 0
    assert En.ic_stats(base + 0.001, 60)["shrunk_ic"] == 0.0               # t ~ 0
    small = pd.Series(np.tile([0.30, -0.20], 30))
    big = pd.Series(np.tile([0.30, -0.02], 30))
    a, b = En.ic_stats(small, 60), En.ic_stats(big, 60)
    assert b["t"] > a["t"] and b["shrunk_ic"] > a["shrunk_ic"] >= 0
    assert b["shrunk_ic"] < b["ic"]                                         # shrunk toward 0
    neg = En.ic_stats(-big, 60)
    assert neg["shrunk_ic"] == 0.0 or neg["ic"] < 0 and neg["shrunk_ic"] <= 0 + 1e-12


def test_cap_weights_cap_redistribution_and_unallocated():
    w = En.cap_weights({"a": 0.7, "b": 0.2, "c": 0.1}, 0.4)
    assert w["a"] == 0.4 and abs(sum(w.values()) - 1.0) < 1e-3 and w["b"] > 0.2
    w = En.cap_weights({"a": 0.7, "b": 0.3}, 0.2)                           # unreachable: 2 x 20% < 100%
    assert w == {"a": 0.2, "b": 0.2}
    w = En.cap_weights({"a": 0.9, "p": 0.1}, 0.2, receivers={"a"})          # excess only flows to receivers
    assert w == {"a": 0.2, "p": 0.1}


def _table(**kw):
    return {k: {"n": v[0], "t": v[1], "shrunk_ic": v[2], "validated": v[0] >= 60} for k, v in kw.items()}


def test_weights_from_table_rules():
    hist = {k: True for k in "abcdef"}
    # everything < 60 sessions -> equal weights
    w, mode = En.weights_from_table(_table(**{k: (30, 2.0, 0.02) for k in hist}), hist)
    assert all(abs(v - 1 / 6) < 1e-3 for v in w.values()) and "equal-weight" in mode
    # t < 1 -> 0 ; cap 20% ; floor 0
    t = _table(a=(60, 3.0, 0.04), b=(60, 2.5, 0.03), c=(60, 1.5, 0.01), d=(60, 0.5, 0.02), e=(60, -2.0, 0.0), f=(60, 0.0, 0.0))
    w, _ = En.weights_from_table(t, hist)
    assert w["d"] == 0 and w["e"] == 0 and w["f"] == 0                      # t<1, negative, zero
    assert max(w.values()) <= CFG["weight_cap"] + 1e-9
    assert sum(w.values()) <= 3 * CFG["weight_cap"] + 1e-9                   # only 3 survivors: the rest is unallocated
    # a validated pool with no evidence leaves its mass unallocated (not re-given)
    t = _table(**{k: (60, 0.2, 0.0) for k in hist})
    w, _ = En.weights_from_table(t, hist)
    assert sum(w.values()) == 0
    # snapshot-only signal (no history): Oct 23 prior = 0.5 x 1/N capped at 5% (here 0.5/7 = 7.1% -> 5%); "zero" policy removes it
    hist2 = {**hist, "snap": False}
    t = _table(a=(60, 3.0, 0.04), b=(60, 2.5, 0.03), c=(60, 1.5, 0.01), d=(60, 0.5, 0.02), e=(60, -2.0, 0.0), f=(60, 0.0, 0.0), snap=(0, None, 0.0))
    w, _ = En.weights_from_table(t, hist2)
    assert abs(w["snap"] - min(CFG["snapshot_prior"] / 7, CFG["snapshot_cap"])) < 1e-3 and abs(w["snap"] - 0.05) < 1e-3
    w, _ = En.weights_from_table(t, hist2, {"unvalidated": "zero"})
    assert w["snap"] == 0


def test_score_cross_section_votes_min_votes_and_missing_strategies():
    d, (close, *_r) = synth.data()
    strats = registry(["xs_momentum_12_1", "ts_trend", "high_52w", "residual_momentum", "etf_trend_200dma"])
    zr = {s.name: s.z(d).iloc[-1] for s in strats}
    rr = {s.name: s.raw(d).iloc[-1] for s in strats}
    w = {s.name: 0.2 for s in strats}
    out = En.score_cross_section(zr, rr, strats, w, ["EN0", "LV0", "NOPE"])
    for t in ("EN0", "LV0"):
        o = out[t]
        assert -1 <= o["score"] <= 1 and o["votes_long"] + o["votes_short"] <= o["n_strategies"]
        assert o["min_votes"] == min(3, o["n_strategies"])
    assert out["LV0"]["n_strategies"] <= 4 and "etf_trend_200dma" in out["LV0"]["strategies"]       # ETFs: fewer strategies apply
    assert "etf_trend_200dma" not in out["EN0"]["strategies"]
    assert out["NOPE"]["score"] is None and out["NOPE"]["n_strategies"] == 0
    # votes count every strategy with data, weighted or not; the score only uses weighted ones
    w0 = {**w, "ts_trend": 0.0}
    o2 = En.score_cross_section(zr, rr, strats, w0, ["EN0"])["EN0"]
    assert o2["n_strategies"] == out["EN0"]["n_strategies"] and o2["strategies"]["ts_trend"]["w"] == 0


def test_walk_forward_weights_ignore_the_future_and_live_equals_backtest():
    from backtest import engine as E
    close, vol, opn, uni, at, stocks, etfs = synth.market(T=520, seed=4)
    px = {"close": close, "open": opn, "volume": vol}
    P = E.prepare(px, uni)
    W = E.weight_history(P)
    k = 430
    cut = {n: v.iloc[:k + 1] for n, v in px.items()}
    Pc = E.prepare(cut, uni)
    Wc = E.weight_history(Pc)
    assert np.allclose(W.iloc[k].values, Wc.iloc[k].values, atol=1e-9), "weights at day k changed when the future was removed"
    # live learn_weights on the truncated data == the backtest row (same rule, same IC window)
    d = strategies.build_data(cut["close"], cut["volume"], uni, at)
    strats = [s for s in registry(E.BACKTESTABLE) if s.name in Pc.names]
    live = En.learn_weights(d, strats)["weights"]
    for n in Pc.names:
        assert abs(live[n] - Wc.iloc[k][n]) < 1e-6, (n, live[n], Wc.iloc[k][n])
    assert (W.sum(axis=1) <= 1 + 1e-6).all() and (W.max(axis=1) <= CFG["weight_cap"] + 1e-6).all()


def test_run_returns_everything_the_quant_agent_needs():
    d, (close, *_r) = synth.data(T=460)
    r = strategies.run(d, ["EN0", "EN1", "LV0"])
    assert set(r) >= {"scores", "weights", "ic_table", "mode", "allocated", "skipped", "tom_tilt", "asof", "descriptions"}
    assert 0 <= r["allocated"] <= 1 and set(r["scores"]) == {"EN0", "EN1", "LV0"}
    assert r["tom_tilt"] in (0.0, CFG["tom_tilt"])
    assert strategies.run(None, ["EN0"]) is None                            # failure -> None, never an exception


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
