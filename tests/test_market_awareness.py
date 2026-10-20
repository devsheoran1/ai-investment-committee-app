"""Stage 5 (Oct 20): screener weights, PEAD, regime rules, Devil's Advocate, Morning Brief fallback, fast track.
Run:  /opt/miniconda3/bin/python3 tests/test_market_awareness.py"""
import os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import analytics, config, screener


def test_screener_effective_weights_match_the_decision():
    s1, s2 = config.SCREEN_STAGE1["weights"], config.SCREEN_STAGE2["weights"]
    assert abs(sum(s1.values()) - 1) < 1e-9 and abs(sum(s2.values()) - 1) < 1e-9
    eff = {k: round(v * s2["stage1"], 2) for k, v in s1.items()}
    assert eff == {"mom_12_1_volscaled": 0.22, "industry_mom_1m": 0.14, "trend": 0.14, "reversal_1w": 0.07}
    assert eff["mom_12_1_volscaled"] + eff["trend"] + eff["industry_mom_1m"] <= 0.50 + 1e-9   # momentum/trend cap


def test_missing_stage2_input_reweights_pro_rata_not_zero():
    df = pd.DataFrame({"stage1": [1.0, -1.0, 0.0, 0.5], "revisions_drift": [np.nan, np.nan, np.nan, np.nan],
                       "quality": [0.0, 0.0, 0.0, 0.0]}, index=list("ABCD"))
    sc = screener._blend(df, config.SCREEN_STAGE2["weights"], None)
    assert sc.idxmax() == "A" and sc.idxmin() == "B" and (sc > -9).all()            # NaN input never scores as 0/-inf


def test_pead_sign_decay_and_no_lookahead():
    idx = pd.bdate_range(end="2026-10-16", periods=40)
    close = pd.DataFrame({"SPY": np.full(40, 100.0), "UP": np.full(40, 50.0), "MIX": np.full(40, 50.0)}, index=idx)
    rep = idx[30]
    close.loc[idx[31]:, "UP"] = 55.0                                               # beat + positive reaction
    close.loc[idx[31]:, "MIX"] = 45.0                                              # beat but sold off -> mixed
    e = {"UP": {"last4": [{"date": str(rep.date()), "surprise_pct": 8.0, "time_et": "07:00"}]},
         "MIX": {"last4": [{"date": str(rep.date()), "surprise_pct": 8.0, "time_et": "07:00"}]}}
    p = analytics.pead(e, close)
    assert p["UP"]["drift_score"] > 0 and p["MIX"]["drift_score"] == 0
    assert abs(p["UP"]["drift_score"] - (1 - p["UP"]["days_since_report"] / config.PEAD_WINDOW_DAYS)) < 1e-3
    early = analytics.pead(e, close, asof=idx[29])                                 # before the report: nothing
    assert early == {}
    wrecked = close.copy(); wrecked.loc[idx[36]:, "UP"] = 1.0
    assert analytics.pead(e, wrecked, asof=idx[35])["UP"] == analytics.pead(e, close, asof=idx[35])["UP"]


def test_regime_rules_c13_exactly():
    base = {"vix_level": 18, "spy_vs_50dma_pct": 1, "spy_vs_200dma_pct": 3, "hyg_lqd_vs_20dma_pct": 0.2, "vix_1d_change_pct": 2}
    assert analytics.macro_regime(base)["raw_regime"] == "risk_on"
    assert analytics.macro_regime({**base, "vix_level": 21})["raw_regime"] == "neutral"            # not all 3
    one = {**base, "vix_level": 26}
    assert analytics.macro_regime(one)["raw_regime"] == "neutral"                                   # only 1 of 4
    two = {**base, "vix_level": 26, "hyg_lqd_vs_20dma_pct": -1.5}
    assert analytics.macro_regime(two)["raw_regime"] == "risk_off"                                  # 2 of 4
    assert analytics.apply_hysteresis("risk_off", "risk_on", "risk_on")[0] == "risk_off"            # downgrade now
    assert analytics.apply_hysteresis("risk_on", "neutral", "neutral")[0] == "neutral"              # 1st day: wait
    assert analytics.apply_hysteresis("risk_on", "neutral", "risk_on")[0] == "risk_on"              # 2nd day: up


def test_devils_advocate_and_pm_must_answer():
    from committee.devils_advocate import DevilsAdvocate
    from committee.portfolio_manager import PortfolioManager
    ideas = [{"ticker": "AAA", "side": "BUY", "target_weight": 0.12}]
    out = DevilsAdvocate().validate({"objections": {"AAA": {"bear_case": "RSI 82", "evidence": "rsi", "verdict": "nuke"}}},
                                    {"ideas": ideas})
    assert out["objections"]["AAA"]["verdict"] == "proceed"                        # invalid verdict -> safe default
    ctx = {"composite": {"AAA": {"composite": 0.5, "insufficient_data": False}}, "current_weights": {}, "mode": "init",
           "eligible_new": [], "forced_exits": {}, "macro_view": {"equity_exposure_target": 0.98},
           "da_objections": {"AAA": {"verdict": "drop"}}}
    pm = PortfolioManager().validate({"convictions": {"AAA": 0.8}}, ctx)
    assert "NOT ANSWERED" in pm["da_responses"]["AAA"]


def test_morning_brief_never_blocks_a_run():
    import morning_brief
    old = config.MORNING_BRIEF_PROVIDER
    try:
        b = morning_brief.build({}, {"KO": {"reports_soon": True, "next_date": "2026-10-27"}}, [], provider="mock")
        assert b["text"] and "KO" in b["earnings_this_week"] and b["model"] == "mock"
    finally:
        config.MORNING_BRIEF_PROVIDER = old
    ev = morning_brief.macro_events(dt.date(2026, 10, 26), days=7)
    assert any("FOMC" in e["event"] for e in ev) and not any("CPI" in e["event"] for e in ev)


def test_fast_track_cap_applies_to_stocks_in_optimiser():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-10-16", periods=300)
    names = [f"S{i}" for i in range(14)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (300, 14)), axis=0)), index=idx, columns=names)
    px["SPY"] = 400 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 300)))
    conv = {t: 0.2 for t in names}; conv["S0"] = 1.0
    w = analytics.optimise(conv, px, {t: f"sec{i % 5}" for i, t in enumerate(names)}, {t: "stock" for t in names},
                           0.98, "neutral", 18, 14, ensure_min=True, caps={"S0": 0.10}, min_weight=0.026)
    assert w["S0"] <= 0.10 + 1e-6, w["S0"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
