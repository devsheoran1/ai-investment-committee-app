"""Stage 1 (Oct 20): every hard risk rule, the soft/warning split, the C11 drawdown ladder, vol-targeted
ETF caps, the A7 holdings floor and the A12 cash upsizing on the EXECUTABLE book.
Run:  /opt/miniconda3/bin/python3 tests/test_risk_limits.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import analytics, config, data
from committee.risk_manager import RiskManager, classify

SECS = ["Energy", "Health Care", "Financials", "Industrials", "Utilities", "Materials"]


def _px(n=30, T=300, seed=1):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, T)
    names = [f"S{i}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.array([0.8 * mkt + rng.normal(0, 0.012, T) for _ in names]).T, axis=0)),
                      columns=names, index=pd.bdate_range(end="2026-10-16", periods=T))
    px["SPY"] = 400 * np.exp(np.cumsum(mkt))
    return px, {t: SECS[i % len(SECS)] for i, t in enumerate(names)}, {t: "stock" for t in names}


def _rr(**over):
    base = {"regime": "neutral", "n_positions": 12, "cash_weight": 0.01, "weights": {f"S{i}": 0.0825 for i in range(12)},
            "sector_weights": {"Energy": 0.2, "Financials": 0.2}, "leveraged_etf_weight": 0.0, "leveraged_by_etf": {},
            "inverse_etf_weight": 0.0, "inverse_by_etf": {}, "portfolio_vol_ann": 0.15, "var_95_1d": 0.015,
            "effective_n_positions": 11.0, "top3_max_pair_corr": 0.5, "top3_holdings": ["S0", "S1", "S2"],
            "adv_fraction": {"S0": 0.01}, "beta_vs_spy": 1.0, "max_pairwise_corr": 0.6, "dd_state": {"level": "ok"},
            "current_weights": {}, "etf_effective_caps": {}}
    base.update(over)
    return classify(base)


def test_each_hard_limit_breaches():
    cases = {
        "holdings": _rr(n_positions=9),
        "cash": _rr(cash_weight=0.051),
        "name": _rr(weights={"S0": 0.21, "S1": 0.1}),
        "sector": _rr(sector_weights={"Energy": 0.36}),
        "lev total": _rr(regime="risk_on", leveraged_etf_weight=0.31, leveraged_by_etf={"TQQQ": 0.15, "SOXL": 0.16}),
        "lev each": _rr(regime="risk_on", leveraged_etf_weight=0.16, leveraged_by_etf={"TQQQ": 0.16}),
        "lev regime": _rr(regime="neutral", leveraged_etf_weight=0.11, leveraged_by_etf={"TQQQ": 0.11}),
        "inverse": _rr(regime="risk_off", inverse_etf_weight=0.26, inverse_by_etf={"SQQQ": 0.26}),
        "inverse outside risk-off": _rr(regime="neutral", inverse_etf_weight=0.05, inverse_by_etf={"SQQQ": 0.05}),
        "vol": _rr(portfolio_vol_ann=0.36),
        "var": _rr(var_95_1d=0.051),
        "c11 warn": _rr(regime="risk_on", leveraged_etf_weight=0.12, leveraged_by_etf={"TQQQ": 0.12},
                        dd_state={"level": "warn", "lev_cap": 0.10, "drawdown": -0.08}, current_weights={"TQQQ": 0.12}),
        "c11 hard": _rr(regime="risk_off", inverse_etf_weight=0.10, inverse_by_etf={"SQQQ": 0.10},
                        dd_state={"level": "hard", "lev_cap": 0.0, "drawdown": -0.11, "inverse_min_risk_off": 0.15}),
    }
    for name, rr in cases.items():
        assert rr["breaches"], f"{name}: expected a hard breach"
    assert not _rr()["breaches"], _rr()["breaches"]


def test_soft_rejects_vs_warnings():
    assert _rr(effective_n_positions=3.5)["soft_rejects"]
    assert _rr(top3_max_pair_corr=0.95)["soft_rejects"]
    assert _rr(adv_fraction={"S0": 0.30})["soft_rejects"]
    r = _rr(max_pairwise_corr=0.85, beta_vs_spy=0.4, effective_n_positions=6)    # old rules: now warnings only
    assert not r["soft_rejects"] and not r["breaches"] and len(r["warnings"]) >= 2


def test_llm_approve_on_breach_is_overridden_and_logged():
    rr = _rr(cash_weight=0.08)
    out = RiskManager().validate({"decision": "approve", "rationale": "looks fine"}, {"risk_report": rr, "round": 1})
    assert out["decision"] == "reject" and out["override"] and out["hard_breaches"]
    assert isinstance(out["required_changes"], list) and out["required_changes"]
    out = RiskManager().validate({"decision": "reject"}, {"risk_report": _rr(beta_vs_spy=0.3), "round": 1})
    assert out["decision"] == "approve" and "warnings only" in out["override"]


def test_vol_doubling_halves_etf_cap():
    T = 300
    base = np.array([0.01, -0.01] * (T // 2))
    spike = base.copy(); spike[-20:] *= 2
    idx = pd.bdate_range(end="2026-10-16", periods=T + 1)
    px = pd.DataFrame({"NORM": 100 * np.exp(np.r_[0, np.cumsum(base)]), "SPIKE": 100 * np.exp(np.r_[0, np.cumsum(spike)])}, index=idx)
    caps = analytics.etf_caps(px, ["NORM", "SPIKE"], {"NORM": "leveraged_etf", "SPIKE": "leveraged_etf"})
    assert abs(caps["NORM"] - config.LEV_BASE_CAP_EACH) < 1e-3, caps
    assert abs(caps["SPIKE"] - config.LEV_BASE_CAP_EACH / 2) < 0.005, caps


def test_c11_ladder_and_recovery():
    c = config.STARTING_CAPITAL
    assert analytics.drawdown_state([c], c * 0.95)["level"] == "ok"
    s = analytics.drawdown_state([c], c * 0.925)
    assert s["level"] == "warn" and s["lev_cap"] == config.DD_WARN_LEV_CAP and s["no_new_leveraged"]
    s = analytics.drawdown_state([c], c * 0.89)
    assert s["level"] == "hard" and s["lev_cap"] == 0.0 and s["inverse_min_risk_off"] == config.DD_HARD_INVERSE_MIN
    assert analytics.drawdown_state([c, c * 0.92], c * 0.95)["level"] == "warn"          # 1 day back above -7%
    assert analytics.drawdown_state([c, c * 0.92, c * 0.95], c * 0.95)["level"] == "ok"   # 2 days back: lifted


def _book(n, px_row, nav=1_000_000, cash_frac=0.01):
    names = [f"S{i}" for i in range(n)]
    per = (1 - cash_frac) * nav / n
    return {t: {"shares": int(per / px_row[t]), "avg_cost": float(px_row[t]), "opened_date": "2026-10-01"} for t in names}


def test_a_discretionary_exit_on_10_name_book_without_paired_buy_is_blocked():
    px, sectors, kinds = _px()
    last = px.iloc[-1].to_dict()
    pos = _book(10, last)
    nav = sum(p["shares"] * last[t] for t, p in pos.items()) + 10_000
    target = {t: p["shares"] * last[t] / nav for t, p in pos.items()}
    target["S0"] = 0.0                                                     # PM wants out, no buy proposed
    orders, skipped = analytics.plan_orders(target, pos, last, nav, 10_000, {}, False, {}, {},
                                            sectors=sectors, asset_types=kinds)
    assert not any(o["ticker"] == "S0" and o["side"] == "SELL" for o in orders), orders
    assert any(s["ticker"] == "S0" and "paired buy" in s["reason"] for s in skipped), skipped


def test_b_two_stop_losses_execute_and_next_run_replaces_first():
    px, sectors, kinds = _px()
    last = px.iloc[-1].to_dict()
    pos = _book(10, last)
    nav = sum(p["shares"] * last[t] for t, p in pos.items()) + 10_000
    target = {t: p["shares"] * last[t] / nav for t, p in pos.items()}
    forced = {"S0": "stop-loss", "S1": "stop-loss"}
    orders, _ = analytics.plan_orders(target, pos, last, nav, 10_000, forced, False, {}, {},
                                      sectors=sectors, asset_types=kinds)
    sold = {o["ticker"] for o in orders if o["side"] == "SELL"}
    assert {"S0", "S1"} <= sold, orders                                    # forced exits always go through
    # next run: 8 names left; target adds 3 replacements AND big resizes of existing names
    pos2 = {t: p for t, p in pos.items() if t not in ("S0", "S1")}
    cash2 = 10_000 + sum(pos[t]["shares"] * last[t] for t in ("S0", "S1"))
    tgt2 = {t: 0.16 for t in pos2}                                         # 8 big resizes compete for 5 slots
    tgt2.update({"S20": 0.05, "S21": 0.05, "S22": 0.05})                  # ... with 3 affordable replacements
    orders2, _ = analytics.plan_orders(tgt2, pos2, last, nav, cash2, {}, False, {}, {},
                                       sectors=sectors, asset_types=kinds)
    new = [o["ticker"] for o in orders2 if o["side"] == "BUY" and o.get("is_new")]
    assert {"S20", "S21", "S22"} <= set(new), orders2
    assert all(o.get("replacement") for o in orders2 if o.get("is_new"))


def test_c_init_always_produces_12_to_15_names():
    px, sectors, kinds = _px(n=40)
    for conv_pattern in ("all_positive", "mostly_avoid"):
        conv = {t: 0.6 for t in kinds} if conv_pattern == "all_positive" else \
            {t: (0.8 if i < 5 else -0.9) for i, t in enumerate(kinds)}
        w = analytics.optimise(conv, px, sectors, kinds, 0.98, "neutral", 18.0, config.TARGET_HOLDINGS_INIT,
                               ensure_min=True, min_weight=0.026)
        assert config.TARGET_HOLDINGS[0] <= len(w) <= config.TARGET_HOLDINGS[1], (conv_pattern, len(w))
        assert sum(w.values()) >= 1 - config.RISK_LIMITS["max_cash_weight"] - 1e-3


def test_cash_capped_by_upsizing_buys_within_the_trade_cap():
    px, sectors, kinds = _px()
    last = px.iloc[-1].to_dict()
    pos = _book(12, last, cash_frac=0.12)                                 # 12% cash after yesterday's exits
    cash = 0.12 * 1_000_000
    nav = sum(p["shares"] * last[t] for t, p in pos.items()) + cash
    target = {t: p["shares"] * last[t] / nav for t, p in pos.items()}
    target.update({"S20": 0.03, "S21": 0.03})                             # only 6% of new buys planned
    orders, skipped = analytics.plan_orders(target, pos, last, nav, cash, {}, False, {}, {},
                                            sectors=sectors, asset_types=kinds)
    assert len(orders) <= config.MAX_TRADES_PER_DAY
    spent = sum(o["notional"] for o in orders if o["side"] == "BUY") - sum(o["notional"] for o in orders if o["side"] == "SELL")
    assert (cash - spent) / nav <= config.RISK_LIMITS["max_cash_weight"] + 1e-6, (cash - spent) / nav
    assert any(o.get("upsized") for o in orders)


def test_missing_price_for_a_held_name_never_crashes():
    px, sectors, kinds = _px()
    last = px.iloc[-1].to_dict()
    pos = _book(12, last)
    nav = sum(p["shares"] * last[t] for t, p in pos.items()) + 10_000
    target = {t: p["shares"] * last[t] / nav for t, p in pos.items()}
    del last["S3"]                                                         # Yahoo dropped a held name
    orders, skipped = analytics.plan_orders(target, pos, last, nav, 10_000, {}, False, {}, {},
                                            sectors=sectors, asset_types=kinds)
    assert any(s["ticker"] == "S3" and "untradeable" in s["reason"] for s in skipped)
    fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    old = config.PRICES_CSV; config.PRICES_CSV = path
    pd.DataFrame({"S3": [10.0, 11.0, np.nan]}, index=pd.Index(["d1", "d2", "d3"], name="date")).to_csv(path)
    try:
        assert data.last_known_prices(["S3"])["S3"] == 11.0
    finally:
        config.PRICES_CSV = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
