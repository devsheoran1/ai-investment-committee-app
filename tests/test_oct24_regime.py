"""Oct 24: shrinkage weights, regime allocator, LLM family tilt, soft SPXU hedge, entry bar, trailing stop, snapshot z-log.
Hermetic, mock only. Run:  /opt/miniconda3/bin/python3 tests/test_oct24_regime.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import analytics, config
from strategies import ensemble as En, regime as R, registry
from strategies import log_snapshot_z
from committee.quant_analyst import QuantAnalyst
from committee.portfolio_manager import PortfolioManager
from committee.risk_manager import classify
import test_quant_models as tq

NAMES = ["xs_momentum_12_1", "ts_trend", "residual_momentum", "industry_momentum", "high_52w", "pead_sue", "eps_revisions",
         "earnings_premium", "st_reversal_1w", "quality_gp", "low_beta", "insider_buying"]
HIST = {n: n not in ("eps_revisions", "quality_gp", "insider_buying") for n in NAMES}


def _table(ts, n=60):
    return {k: {"n": n, "t": ts.get(k, 0.0), "shrunk_ic": 0.0, "validated": n >= 60} for k in NAMES}


# ---- weights ------------------------------------------------------------------------------------------------------- #
def test_shrinkage_never_all_zero_monotone_in_t_and_caps():
    w, mode = En.weights_from_table(_table({k: -3.0 for k in NAMES}), HIST)             # every t negative
    assert sum(w.values()) > 0.99 and all(w[k] > 0 for k in NAMES) and "shrinkage" in mode
    w, _ = En.weights_from_table(_table({"ts_trend": 1.0, "high_52w": 3.0, "pead_sue": 6.0, "low_beta": 0.0}), HIST)
    assert w["high_52w"] > w["ts_trend"] > w["low_beta"] > 0                           # more t = more say
    assert w["pead_sue"] <= 0.30 + 1e-9 and w["pead_sue"] > 0.20                       # PEAD may go to 30%
    assert all(w[k] <= 0.20 + 1e-9 for k in NAMES if k != "pead_sue")
    assert sum(w.values()) > 0.99
    assert all(abs(w[k] - 0.5 / len(NAMES)) < 1e-4 for k in NAMES if not HIST[k])      # snapshot-only: 0.5 x 1/N, <= 5%
    w, _ = En.weights_from_table(_table({"ts_trend": 5.0}, n=30), HIST)                 # < 60 sessions: t is not trusted
    assert abs(w["ts_trend"] - w["low_beta"]) < 1e-9


# ---- regime states + allocator ------------------------------------------------------------------------------------- #
def _spy(kind):
    n = 400
    rng = np.random.default_rng(3)
    quiet = rng.normal(0.0006, 0.004, n)
    if kind == "calm":
        r = quiet
    elif kind == "drawdown":                      # quiet year, then a -12% slide with big daily moves
        r = quiet.copy(); r[-25:] = rng.normal(-0.007, 0.02, 25)
    elif kind == "rebound":                       # long decline below the 200dma, then +7% bounce off the low
        r = np.r_[rng.normal(0.0004, 0.004, 150), rng.normal(-0.0025, 0.006, 200), np.full(50, 0.0013)]
    else:                                         # flat/choppy: neither calm nor stressed
        r = rng.normal(0.0, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2025-01-01", periods=n))


def test_regime_states_follow_the_spec():
    assert R.regime_states(_spy("calm")).iloc[-1] == "calm_uptrend"
    assert R.regime_states(_spy("drawdown")).iloc[-1] == "drawdown_high_vol"
    assert R.regime_states(_spy("rebound")).iloc[-1] == "rebound"
    assert R.regime_states(_spy("calm").iloc[:50]).iloc[-1] == "normal"                 # not enough history -> normal
    cfg = config.REGIME_ALLOC["multipliers"]
    assert cfg["calm_uptrend"] == {"momentum": 1.2, "earnings": 1.0, "reversal": 0.5, "defensive": 0.8}
    assert cfg["drawdown_high_vol"] == {"momentum": 0.5, "earnings": 1.5, "reversal": 1.5, "defensive": 1.5}
    assert cfg["rebound"] == {"momentum": 0.3, "earnings": 1.2, "reversal": 2.0, "defensive": 1.0}
    assert set(cfg["normal"].values()) == {1.0}


def test_apply_state_shifts_family_mass_preserves_total_and_caps():
    base, _ = En.weights_from_table(_table({}), HIST)
    fam = lambda w, f: sum(v for k, v in w.items() if R.family_of(k) == f)
    assert R.apply_state(base, "normal", HIST) == base
    for st, direction in (("calm_uptrend", +1), ("drawdown_high_vol", -1), ("rebound", -1)):
        w = R.apply_state(base, st, HIST)
        assert abs(sum(w.values()) - sum(base.values())) < 2e-3                           # renormalised to the same total
        assert (fam(w, "momentum") - fam(base, "momentum")) * direction > 0, st
        assert w["pead_sue"] <= 0.30 + 1e-9 and all(w[k] <= 0.20 + 1e-9 for k in w if k != "pead_sue")
        assert all(abs(w[k] - base[k]) < 1e-9 or HIST[k] or w[k] <= 0.05 + 1e-9 for k in w)    # snapshot caps hold
    assert fam(R.apply_state(base, "rebound", HIST), "reversal") > 1.5 * fam(base, "reversal")


# ---- LLM family tilt ----------------------------------------------------------------------------------------------- #
def _ens_ctx(n=6):
    ctx = tq._ctx(n=n)
    w = {k: round(1 / 6, 4) for k in ("xs_momentum_12_1", "ts_trend", "high_52w", "st_reversal_1w", "low_beta", "pead_sue")}
    zs = {"xs_momentum_12_1": 1.5, "ts_trend": 1.0, "high_52w": 1.0, "st_reversal_1w": -1.0, "low_beta": -0.5, "pead_sue": 0.5}
    scores = {}
    for t in ctx["tickers"]:
        det = {k: {"z": z, "w": w[k], "reason": k} for k, z in zs.items()}
        e = R.rescore(det, w)
        scores[t] = {"score": round(e, 3), "ens_z": 0, "votes_long": 3, "votes_short": 0, "n_strategies": 6, "min_votes": 3, "strategies": det}
    ctx["ensemble"] = {"scores": scores, "allocated": 1.0, "tom_tilt": 0.0, "mode": "m", "weights": w, "ic_table": {}, "skipped": [],
                       "asof": "2026-10-23", "has_history": {k: True for k in w}, "regime_state": "normal"}
    return ctx


def test_family_tilt_is_bounded_needs_a_reason_and_is_logged():
    q = QuantAnalyst()
    ctx = _ens_ctx()
    sig = {t: {"score": 0.2} for t in ctx["tickers"]}
    base = q.validate({"signals": sig}, ctx)
    assert "family_tilt" not in base
    no_reason = q.validate({"signals": sig, "family_tilt": {"family": "momentum", "pct": 0.2, "reason": ""}}, ctx)
    assert no_reason["family_tilt"]["applied"] is False
    assert {t: v["score"] for t, v in no_reason["signals"].items()} == {t: v["score"] for t, v in base["signals"].items()}
    bad_family = q.validate({"signals": sig, "family_tilt": {"family": "crypto", "pct": 0.2, "reason": "x"}}, ctx)
    assert "family_tilt" not in bad_family
    up = q.validate({"signals": sig, "family_tilt": {"family": "momentum", "pct": 0.9, "reason": "trend is clean"}}, ctx)
    ft = up["family_tilt"]
    assert ft["applied"] and ft["pct"] == 0.2 and ft["reason"] == "trend is clean"        # clipped to +20%
    t0 = ctx["tickers"][0]
    assert up["signals"][t0]["score"] > base["signals"][t0]["score"]                      # momentum z is positive: more weight = higher score
    assert abs(up["signals"][t0]["score"] - base["signals"][t0]["score"]) < 0.05           # ... but only slightly
    down = q.validate({"signals": sig, "family_tilt": {"family": "momentum", "pct": -0.2, "reason": "crowded"}}, ctx)
    assert down["signals"][t0]["score"] < base["signals"][t0]["score"]
    assert abs(sum(ft["weights_after"].values()) - sum(ft["weights_before"].values())) < 2e-3
    assert "tilted_weights" in up["strategy_ensemble"]


# ---- soft SPXU hedge ----------------------------------------------------------------------------------------------- #
def _rr(inv, regime="neutral"):
    return {"regime": regime, "n_positions": 12, "cash_weight": 0.01, "weights": {f"S{i}": 0.0825 for i in range(12)},
            "sector_weights": {"Energy": 0.2, "Financials": 0.2}, "leveraged_etf_weight": 0.0, "leveraged_by_etf": {},
            "inverse_etf_weight": inv, "inverse_by_etf": {"SPXU": inv} if inv else {}, "portfolio_vol_ann": 0.15,
            "var_95_1d": 0.015, "effective_n_positions": 11.0, "top3_max_pair_corr": 0.5, "top3_holdings": ["S0", "S1", "S2"],
            "adv_fraction": {"S0": 0.01}, "beta_vs_spy": 1.0, "max_pairwise_corr": 0.6, "dd_state": {"level": "ok"},
            "current_weights": {}, "etf_effective_caps": {}}


def _inv_row(res):
    return next(r for r in res["limit_checks"] if r["rule"].startswith("Inverse only"))


def test_risk_enforces_only_the_25pct_cap_inside_the_soft_hedge_window():
    for kw in ({}, {"soft_hedge_open": False}):
        res = classify(_rr(0.06), **kw)
        assert _inv_row(res)["status"] == "fail"                                        # inverse outside risk-off, no signal: still a breach
    ok = classify(_rr(0.06), soft_hedge_open=True)
    assert _inv_row(ok)["status"] == "pass"
    over = classify(_rr(0.26), soft_hedge_open=True)
    assert any(r["rule"].startswith("Inverse total") and r["status"] == "fail" for r in over["limit_checks"])


def test_pm_sizes_the_soft_hedge_and_nobody_else_can():
    ctx = {"composite": {"AAA": {"composite": 0.5, "composite_10": 7.5}, "SPXU": {"composite": -0.5, "composite_10": 2.0, "type": "inverse_etf"}},
           "current_weights": {}, "eligible_new": ["AAA"], "forced_exits": [], "mode": "daily",
           "macro_view": {"equity_exposure_target": 0.98}, "soft_hedge_open": True}
    pm = PortfolioManager()
    v = pm.validate({"convictions": {"AAA": 0.5}, "soft_hedge": 0.25}, ctx)
    assert v["soft_hedge"] == 0.10 and v["convictions"]["SPXU"] == 0.5               # clipped to 10%
    v = pm.validate({"convictions": {"AAA": 0.5}}, ctx)
    assert v["soft_hedge"] == 0.0 and v["convictions"]["SPXU"] == -1.0               # the PM must ask for it
    v = pm.validate({"convictions": {"AAA": 0.5}, "soft_hedge": 0.05}, {**ctx, "soft_hedge_open": False})
    assert v["soft_hedge"] == 0.0                                                    # window closed = no hedge
    assert "soft_hedge" in PortfolioManager.output_schema


def test_optimiser_holds_exactly_the_soft_hedge_weight_outside_risk_off():
    rng = np.random.default_rng(3)
    n, T = 30, 300
    secs = ["Energy"] * 8 + ["Health Care"] * 6 + ["Financials"] * 6 + ["Industrials"] * 5 + ["Utilities"] * 5
    names = [f"S{i}" for i in range(n)]
    mkt = rng.normal(0.0004, 0.01, T)
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.array([0.8 * mkt + rng.normal(0, 0.012, T) for _ in names]).T, axis=0)), columns=names)
    px["SPY"] = 400 * np.exp(np.cumsum(mkt))
    px["SPXU"] = 50 * np.exp(np.cumsum(-3 * mkt))
    sectors = {**dict(zip(names, secs)), "SPXU": "Inverse ETF"}
    kinds = {**{t: "stock" for t in names}, "SPXU": "inverse_etf"}
    conv = {t: 0.5 for t in names[:20]} | {"SPXU": 0.5}
    w0 = analytics.optimise(conv, px, sectors, kinds, 0.98, "neutral", 18.0, 15)
    assert "SPXU" not in w0                                                           # no soft hedge asked: nothing changes
    w = analytics.optimise(conv, px, sectors, kinds, 0.98, "neutral", 18.0, 15, soft_inverse=0.06)
    assert abs(w.get("SPXU", 0) - 0.06) < 0.004, w.get("SPXU")
    assert sum(w.values()) >= 0.95 - 1e-3 and all(v <= 0.20 + 1e-6 for t, v in w.items() if t != "SPXU")


def test_screener_only_adds_spxu_in_the_soft_window():
    src = open(os.path.join(config.BASE_DIR, "screener.py")).read()
    assert "with_inverse" in src and "SOFT_HEDGE" in src


# ---- entry bar, trailing stop, snapshot z-log ----------------------------------------------------------------------- #
def test_entry_bar_and_stop_settings():
    assert config.ENTRY_COMPOSITE == 6.3 and config.TRAILING_STOP == -0.12 and config.STOP_LOSS == -0.10
    src = open(os.path.join(config.BASE_DIR, "orchestrator.py")).read()
    assert "comp[t][\"composite_10\"] >= config.ENTRY_COMPOSITE" in src and "weakest + 1" in src      # beat the weakest by >= 1 stays


def test_snapshot_z_log_is_written_once_per_day(tmp=None):
    from types import SimpleNamespace as NS
    idx = pd.bdate_range("2026-10-20", periods=3)
    strats = [NS(name="xs_momentum_12_1", tilt=False, history=True), NS(name="quality_gp", tilt=False, history=False)]
    panels = {s.name: pd.DataFrame(np.random.default_rng(0).normal(size=(3, 5)), index=idx, columns=list("ABCDE")) for s in strats}
    path = os.path.join(tempfile.mkdtemp(), "z.csv")
    old = config.SNAPSHOT_Z_LOG, os.environ.get("FUND_SNAPSHOT_Z_LOG")
    config.SNAPSHOT_Z_LOG = path; os.environ["FUND_SNAPSHOT_Z_LOG"] = path
    try:
        n1 = log_snapshot_z(panels, strats, "2026-10-23")
        n2 = log_snapshot_z(panels, strats, "2026-10-23")                              # same day again: replaced, not duplicated
        n3 = log_snapshot_z(panels, strats, "2026-10-24")
        df = pd.read_csv(path)
        assert n1 == n2 == n3 and set(df["strategy"]) <= {"quality_gp"}                # price-based strategies are never logged
        assert df["asof"].nunique() == 2 and len(df) == n1 + n3
    finally:
        config.SNAPSHOT_Z_LOG = old[0]
        os.environ.pop("FUND_SNAPSHOT_Z_LOG", None) if old[1] is None else os.environ.__setitem__("FUND_SNAPSHOT_Z_LOG", old[1])


# ---- end to end (mock committee run) -------------------------------------------------------------------------------- #
from test_handoff import _with_tmp          # noqa: E402
from test_operations import _last_day       # noqa: E402


@_with_tmp
def test_mock_run_with_one_risk_off_signal_opens_the_soft_hedge_end_to_end():
    import orchestrator
    import database as db
    real_regime, real_mock = analytics.macro_regime, PortfolioManager.mock

    def one_signal(macro):                      # exactly ONE C1.3 risk-off signal -> neutral regime, soft window open
        r = real_regime(macro)
        off = {k: (k == "VIX > 25") for k in r["risk_off_signals"]}
        return {**r, "raw_regime": "neutral", "risk_off_signals": off, "lev_buy_block": ["VIX > 25"]}

    def pm_mock(self, ctx):
        out = real_mock(self, ctx)
        out["soft_hedge"] = 0.05 if ctx.get("soft_hedge_open") else 0.0
        return out
    analytics.macro_regime, PortfolioManager.mock = one_signal, pm_mock
    try:
        res = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        analytics.macro_regime, PortfolioManager.mock = real_regime, real_mock
    prop = db.get_proposal(res["run_id"])
    spxu = [o for o in res["orders"] if o["ticker"] == "SPXU"]
    assert spxu and spxu[0]["side"] == "BUY", [o["ticker"] for o in res["orders"]]
    w = spxu[0]["shares"] * prop["last_px"]["SPXU"] / 1_000_000
    assert 0.03 < w < 0.07, w                                                   # the PM's 5%
    pm = next(m for m in db.get_messages(res["run_id"]) if m["agent"] == "Portfolio Manager")["content"]
    assert pm["soft_hedge"] == 0.05
    assert not [b for b in (res.get("risk") or {}).get("hard_breaches", [])]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
