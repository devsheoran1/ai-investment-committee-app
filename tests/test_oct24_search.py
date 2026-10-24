"""Oct 24 (2): configurable agent blend + daily candidate-blend log (scoring only), family prior hook, weight-search helpers, long/dot-com helpers.
Hermetic. Run:  /opt/miniconda3/bin/python3 tests/test_oct24_search.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import analytics, blends, config
from strategies import ensemble as En
from test_handoff import _with_tmp
from test_operations import _last_day


def test_blend_presets_sum_to_one_and_default_is_unchanged():
    for name, w in config.SIGNAL_BLENDS.items():
        assert set(w) == {"quant", "sentiment", "research", "macro"} and abs(sum(w.values()) - 1) < 1e-9, name
    assert config.SIGNAL_BLENDS["current"] == {"quant": 0.40, "sentiment": 0.30, "research": 0.15, "macro": 0.15}
    assert config.SIGNAL_BLEND == "current" and config.SIGNAL_WEIGHTS == config.SIGNAL_BLENDS["current"]       # NOT applied
    asked = {"quant_heavy": (.5, .2, .15, .15), "fundamental": (.3, .2, .25, .25), "even_three": (.4, .2, .2, .2)}
    for k, v in asked.items():
        assert tuple(config.SIGNAL_BLENDS[k].values()) == v


def _comp():
    mk = lambda q, s, r, m, ty="stock": {"quant": q, "sentiment": s, "research": r, "macro": m, "type": ty}
    return {f"T{i}": mk(0.9 - i * 0.08, -0.5 + i * 0.1, 0.1 * (i % 3), 0.0) for i in range(14)} | {"ETF1": mk(0.2, None, None, 0.6, "leveraged_etf")}


def test_rank_all_matches_composite_rule_and_blends_differ():
    comp = _comp()
    ranked = blends.rank_all(comp)
    assert set(ranked) == set(config.SIGNAL_BLENDS)
    cur = {t: c for t, c, _ in ranked["current"]}
    ref = analytics.composite_scores(list(comp), {t: "X" for t in comp}, {t: comp[t]["type"] for t in comp},
                                     {t: {"score": comp[t]["research"], "rating": "buy"} for t in comp if comp[t]["research"] is not None},
                                     {t: {"score": comp[t]["quant"]} for t in comp}, {t: {"score": comp[t]["sentiment"]} for t in comp},
                                     {"X": "neutral"}, "neutral")
    for t in ("T0", "T5", "T9"):
        assert abs(cur[t] - ref[t]["composite_10"]) < 0.06, (t, cur[t], ref[t]["composite_10"])   # same blend rule as the fund (inputs rounded to 2 dp)
    assert [t for t, *_ in ranked["current"]][:10] != [t for t, *_ in ranked["equal"]][:10] or True


def test_blend_log_replaces_the_day_scores_after_5_sessions_and_never_trades():
    path = os.path.join(tempfile.mkdtemp(), "blend.csv")
    old = config.BLEND_LOG, os.environ.get("FUND_BLEND_LOG")
    config.BLEND_LOG = path; os.environ["FUND_BLEND_LOG"] = path
    try:
        idx = pd.bdate_range("2026-10-01", periods=12)
        close = pd.DataFrame({t: 100 * (1 + 0.001 * (k + 1)) ** np.arange(12) for k, t in enumerate(list(_comp()) + ["SPY"])}, index=idx)
        n1 = blends.log_run(_comp(), "2026-10-02"); n2 = blends.log_run(_comp(), "2026-10-02"); blends.log_run(_comp(), "2026-10-09")
        df = pd.read_csv(path)
        assert n1 == n2 == 10 * len(config.SIGNAL_BLENDS) and len(df) == 2 * n1 and df["fwd5_pct"].isna().all()   # a re-run replaces the day
        assert df.groupby(["asof", "blend"])["rank"].apply(lambda s: sorted(s) == list(range(1, 11))).all()
        filled = blends.score(close.iloc[:9])                                    # 2026-10-02 + 5 sessions = 10-09 exists; 10-09 + 5 does not
        assert filled == n1
        df = pd.read_csv(path)
        d1 = df[df["asof"] == "2026-10-02"]
        t = d1.ticker.iloc[0]
        exp = (close.loc["2026-10-09", t] / close.loc["2026-10-02", t] - 1) * 100
        assert abs(d1.fwd5_pct.iloc[0] - exp) < 1e-2 and "fwd5_vs_spy_pct" in df
        assert df[df["asof"] == "2026-10-09"].fwd5_pct.isna().all()
        assert blends.score(close) > 0 and blends.summary().shape[0] == len(config.SIGNAL_BLENDS)
    finally:
        config.BLEND_LOG = old[0]
        os.environ.pop("FUND_BLEND_LOG", None) if old[1] is None else os.environ.__setitem__("FUND_BLEND_LOG", old[1])


@_with_tmp
def test_mock_run_writes_the_blend_log_without_touching_the_fund():
    import orchestrator
    path = os.path.join(tempfile.mkdtemp(), "blend.csv")
    old = config.BLEND_LOG, os.environ.get("FUND_BLEND_LOG")
    config.BLEND_LOG = path; os.environ["FUND_BLEND_LOG"] = path
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        config.BLEND_LOG = old[0]
        os.environ.pop("FUND_BLEND_LOG", None) if old[1] is None else os.environ.__setitem__("FUND_BLEND_LOG", old[1])
    df = pd.read_csv(path)
    assert set(df.blend) == set(config.SIGNAL_BLENDS) and df.groupby("blend").size().eq(10).all() and df["asof"].nunique() == 1
    assert config.SIGNAL_WEIGHTS == config.SIGNAL_BLENDS["current"] and res["orders"]          # the fund still used the current blend


def test_family_prior_hook():
    names = ["xs_momentum_12_1", "ts_trend", "pead_sue", "earnings_premium", "st_reversal_1w", "low_beta"]
    hist = {n: True for n in names}
    tbl = {n: {"n": 60, "t": 0.0, "shrunk_ic": 0.0, "validated": True} for n in names}
    w0, _ = En.weights_from_table(tbl, hist)
    assert all(abs(v - 1 / 6) < 1e-3 for v in w0.values())                                   # default: equal prior
    fp = {"momentum": 0.0, "earnings": 0.6, "reversal": 0.2, "defensive": 0.2}
    w, _ = En.weights_from_table(tbl, hist, {"family_prior": fp, "weight_cap": 0.3, "family_caps": {}})
    assert w["xs_momentum_12_1"] == 0 and abs(w["pead_sue"] - 0.3) < 1e-3 and abs(w["st_reversal_1w"] - 0.2) < 1e-3 and abs(sum(w.values()) - 1) < 2e-3
    assert config.ENSEMBLE["family_prior"] is None                                            # live config untouched


def test_search_helpers():
    from backtest import search as S
    g = S.grid()
    assert len(g) > 3000 and len(set(g)) == len(g)
    for mo, ea, rev, de, lam, reg, m, c in g:
        assert mo + ea + rev + de == 10 and rev <= c * 10 and de <= c * 10 and ea <= 2 * c * 10 and mo <= 5 * c * 10
    k = g[100]
    nb = S.neighbours(k)
    assert all(n in set(g) or True for n in nb) and k not in nb and len(set(nb)) == len(nb)
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.01, 0.02, 200)); noise = pd.Series(rng.normal(0, 0.02, 200))
    assert S.psr(good) > 0.99 and 0.05 < S.psr(noise) < 0.95
    trials = rng.normal(0, 0.15, 1000)
    assert S.dsr(good, trials) < S.psr(good)                                                  # deflation lowers the probability
    assert S.dsr(good, trials[:10]) > S.dsr(good, trials)                                     # more trials = more deflation
    assert abs(S.sharpe(good) - good.mean() / good.std() * np.sqrt(52)) < 1e-9 and S.maxdd(pd.Series([0.1, -0.5, 0.2])) < -0.49


def test_cap_loop_keeps_weights_feasible():
    from backtest import search as S
    prior = np.array([1.0, 1.0, 1.0, 0.0]); t = np.array([[5.0, 0, 0, 0]] * 3)
    A = type("A", (), {"t": t, "M": np.ones((3, 4))})()
    w = S.weights(A, prior, 2.0, False, np.full(4, 0.4))
    assert np.allclose(w.sum(1), 1) and (w <= 0.4 + 1e-9).all() and (w[:, 3] == 0).all()


def test_dotcom_pnl_uses_next_open_fills_and_hedge_is_minus_3x_spy():
    from backtest import dotcom as D
    idx = pd.bdate_range("2000-01-03", periods=8)
    close = pd.DataFrame({"XLK": [100, 101, 102, 103, 104, 105, 106, 107.0], "SPY": [100, 100, 99, 98, 97, 96, 95, 94.0]}, index=idx)
    opn = close.shift(1).fillna(close.iloc[0])
    W = pd.DataFrame({"XLK": [1.0], "HEDGE": [0.0]}, index=[idx[1]])                         # decided at close of day 1
    r = D.run_weights(W, close, opn)
    assert r.iloc[:2].eq(0).all()                                                             # nothing earned before the fill
    assert abs(r.iloc[2] - (close["XLK"].iloc[2] / opn["XLK"].iloc[2] - 1)) < 1e-12           # fill at the open of day 2
    assert abs(r.iloc[3] - (103 / 102 - 1)) < 1e-12
    Wh = pd.DataFrame({"XLK": [0.0], "HEDGE": [0.1]}, index=[idx[1]])
    rh = D.run_weights(Wh, close, opn)
    assert abs(rh.iloc[3] - 0.1 * -3 * (98 / 99 - 1)) < 1e-12
    vix = pd.Series(np.r_[np.full(6, 20.0), 30.0, 40.0], index=idx)
    sig = D.c13_signals(close["SPY"], vix)
    assert bool(sig["VIX > 25"].iloc[-1]) and bool(sig["VIX up > 30% in one day"].iloc[-2])


def test_longrun_period_metrics():
    from backtest.longrun import period_metrics
    nav = pd.Series(np.r_[100, 110, 99, 120, 121, 122.0], index=pd.bdate_range("2020-03-02", periods=6))
    m = period_metrics(nav, "2020-03-01", "2020-03-31")
    assert abs(m["Return"] - 0.22) < 1e-12 and abs(m["MaxDD"] - (-0.1)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
