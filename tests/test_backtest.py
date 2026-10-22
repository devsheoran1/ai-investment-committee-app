"""Oct 23: backtest engine + Monte Carlo (C6, C7). Offline, synthetic market.
Run:  /opt/miniconda3/bin/python3 tests/test_backtest.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config
from backtest import engine as E
import synth


def _prep(T=560, seed=5):
    close, vol, opn, uni, at, stocks, etfs = synth.market(T=T, seed=seed)
    px = {"close": close, "open": opn, "volume": vol}
    return E.prepare(px, uni), px, uni


def _run(P, **kw):
    W = E.weight_history(P)
    sc, vt, al = E.ensemble_panels(P, W)
    dep = E.as_deployed(P, sc, al)
    return E.simulate(P, dep, vt, start=str(P.close.index[300].date()), **kw), dep, vt


def test_simulation_respects_the_fund_rules():
    P, px, uni = _prep()
    r, _, _ = _run(P)
    assert r["min_cash"] >= -1e-6                                                # never spends cash it does not have
    days = sorted(r["fills_by_day"])
    assert len(days) > 5
    daily = [r["fills_by_day"][d] for d in days[1:]]                              # after the init day: <= 5 trades/day
    assert max(daily) <= config.MAX_TRADES_PER_DAY, max(daily)
    nav = r["nav"]
    assert nav.index[0] == P.close.index[300] and nav.notna().all() and (nav > 0).all()
    assert abs(nav.iloc[0] / config.STARTING_CAPITAL - 1) < 0.15


def test_fills_happen_at_the_open_of_the_day_after_the_signal():
    P, px, uni = _prep(T=420)
    i0 = 330
    sc = pd.DataFrame(0.0, index=P.close.index, columns=P.stocks)
    sc.loc[:, P.stocks[::5]] = 0.5                                                # constant signal: buy these 14 on day 1
    r = E.simulate(P, sc, None, start=str(P.close.index[i0].date()), vote_gate=False, stop=False, trailing=False, freq="W")
    assert r["fills"], "no fills"
    for day, t, side, sh, price in r["fills"]:
        assert abs(price - P.open.loc[day, t]) < 1e-9, (day, t)                  # every fill is at that day's OPEN
        assert day > P.close.index[i0 - 1]                                        # and never on the signal day itself
    first_day = r["fills"][0][0]
    assert first_day == P.close.index[i0]
    assert r["fills_by_day"][first_day] >= 10                                     # the init day is exempt from the 5-trade cap
    # the signal is generated from the previous close: a gap on the fill day changes the price paid, not the share count basis
    assert all(side == "BUY" for _d, _t, side, *_ in r["fills"][:10])


def test_nothing_before_day_k_depends_on_prices_after_k():
    P, px, uni = _prep(T=520)
    k = 420
    W = E.weight_history(P)
    sc, vt, al = E.ensemble_panels(P, W)
    dep = E.as_deployed(P, sc, al)
    start = str(P.close.index[350].date())
    base = E.simulate(P, dep, vt, start=start)["nav"]
    px2 = {n: v.copy() for n, v in px.items()}
    rng = np.random.default_rng(9)
    for n in ("close", "open"):
        px2[n].iloc[k + 1:] = px2[n].iloc[k + 1:] * rng.uniform(0.5, 1.5, px2[n].iloc[k + 1:].shape)
    P2 = E.prepare(px2, uni)
    W2 = E.weight_history(P2)
    sc2, vt2, al2 = E.ensemble_panels(P2, W2)
    alt = E.simulate(P2, E.as_deployed(P2, sc2, al2), vt2, start=start)["nav"]
    common = base.index[base.index <= P.close.index[k]]
    assert np.allclose(base[common].values, alt[common].values, atol=1e-6), "NAV before k moved when future prices changed"


def test_cap_book_enforces_name_sector_and_total_limits():
    sector = {f"S{i}": ("A" if i < 8 else "B") for i in range(12)}
    w = E.cap_book({f"S{i}": 1.0 + (i == 0) * 6 for i in range(12)}, sector, 0.97)
    assert max(w.values()) <= 0.20 + 1e-9
    assert sum(v for k, v in w.items() if sector[k] == "A") <= 0.35 + 1e-9
    assert sum(w.values()) <= 0.97 + 1e-9
    assert E.cap_book({}, sector, 0.97) == {}


def test_metrics_on_known_series():
    idx = pd.bdate_range("2024-01-02", periods=253)
    nav = pd.Series(1_000_000 * 1.001 ** np.arange(253), index=idx)
    m = E.metrics(nav, traded_one_way=2_000_000)
    assert m["MaxDD"] == 0 and m["Vol"] < 1e-6
    assert abs(m["Total"] - (1.001 ** 252 - 1)) < 1e-9
    assert 0.2 < m["CAGR"] < 0.4
    assert abs(m["Turnover/yr"] - 2_000_000 / nav.mean() / ((idx[-1] - idx[0]).days / 365.25)) < 1e-9
    dd = pd.Series([100, 110, 99, 105.0], index=pd.bdate_range("2024-01-02", periods=4))
    assert abs(E.metrics(dd)["MaxDD"] - (99 / 110 - 1)) < 1e-12


def test_buy_hold_starts_at_the_first_open():
    P, px, uni = _prep(T=420)
    s = E.buy_hold(P.close, P.open, "SPY", str(P.close.index[330].date()))
    assert abs(s.iloc[0] - config.STARTING_CAPITAL * P.close["SPY"].iloc[330] / P.open["SPY"].iloc[330]) < 1e-6


def test_standalone_strategy_report_has_ic_and_sharpe():
    P, px, uni = _prep(T=560)
    out = E.standalone(P, "xs_momentum_12_1", top=10, start=str(P.close.index[300].date()))
    assert set(out) >= {"Sharpe", "CAGR", "MeanIC", "IC_t", "Turnover/yr"} and np.isfinite(out["Sharpe"])
    # a deliberately perfect signal scores a near-perfect IC
    P.z["xs_momentum_12_1"] = (P.close.shift(-5) / P.close - 1)[P.stocks].rank(axis=1)
    from strategies.ensemble import forward_returns, rank_ic
    P.ic["xs_momentum_12_1"] = rank_ic(P.z["xs_momentum_12_1"], forward_returns(P.close, 5)[P.stocks])
    assert E.standalone(P, "xs_momentum_12_1", top=10, start=str(P.close.index[300].date()))["MeanIC"] > 0.99


def test_monte_carlo_is_reproducible_bounded_and_internally_consistent():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=750)
    spy = pd.Series(rng.normal(0.0004, 0.01, 750), index=idx)
    strat = 0.9 * spy + pd.Series(rng.normal(0, 0.004, 750), index=idx)
    a = E.monte_carlo(strat, spy, paths=4000, seed=3)
    b = E.monte_carlo(strat, spy, paths=4000, seed=3)
    assert a["P(beat benchmark)"] == b["P(beat benchmark)"] and np.array_equal(a["_ret"], b["_ret"])
    assert 0 <= a["P(DD <= -10%)"] <= a["P(DD <= -7%)"] <= 1
    assert a["horizon"] == 15 and a["paths"] == 4000
    assert a["return"]["p5"] < a["return"]["p50"] < a["return"]["p95"]
    assert a["max_dd"]["p5"] <= a["max_dd"]["p95"] <= 0
    c = E.monte_carlo(spy, spy, paths=2000, seed=1)                                # same series vs itself
    assert c["P(beat benchmark)"] == 0.0 and np.allclose(c["_excess"], 0)
    # a strategy that is strictly worse every day can never beat the benchmark
    worse = E.monte_carlo(spy - 0.01, spy, paths=2000, seed=1)
    assert worse["P(beat benchmark)"] == 0.0 and worse["P(loss)"] > 0.9
    # block bootstrap keeps joint draws: with beta 1 and no noise the excess is exactly the constant daily gap compounded
    flat = E.monte_carlo(spy + 0.001, spy, paths=1000, seed=2)
    assert flat["P(beat benchmark)"] > 0.97


def test_turn_of_month_study_and_etf_filter_run():
    P, px, uni = _prep(T=520)
    st = E.tom_study(P, start=str(P.close.index[300].date()))
    assert st["window_days"] > 5 and np.isfinite(st["t"])
    out = E.etf_filter_standalone(P, start=str(P.close.index[300].date()))
    assert len(out) == 2 and all(np.isfinite(v["Sharpe"]) for v in out.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
