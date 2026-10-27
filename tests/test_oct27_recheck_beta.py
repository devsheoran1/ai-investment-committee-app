"""Oct 27 (2): option C after a failed CIO-edit re-check, and the soft neutral beta band.
Run:  /opt/miniconda3/bin/python3 tests/test_oct27_recheck_beta.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import analytics, config, orchestrator


def _o(t, side, sh, px, forced=False):
    return {"ticker": t, "side": side, "shares": sh, "price": px, "notional": round(sh * px, 2), "forced": forced}


# ---- upsizing ---------------------------------------------------------------------------------------- #
def test_upsize_brings_cash_under_the_cap_inside_name_and_sector_caps():
    nav, px = 1_000_000.0, {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    sectors, at = {"AAA": "X", "BBB": "Y", "CCC": "Z"}, {t: "stock" for t in px}
    # book is all cash except one trimmed buy pair: 100k + 100k of buys -> cash 80%
    orders = [_o("AAA", "BUY", 1000, 100.0), _o("BBB", "BUY", 1000, 100.0)]
    out = analytics.upsize_buys_for_cash(orders, {}, px, nav, nav, sectors, at, {})
    spent = sum(o["notional"] for o in out)
    w = {o["ticker"]: o["notional"] / nav for o in out}
    assert all(v <= config.RISK_LIMITS["max_single_weight"] + 1e-9 for v in w.values())          # 20% name cap holds
    assert spent > 200_000 and all(o.get("upsized") for o in out)
    assert abs(w["AAA"] - w["BBB"]) < 0.002                                                       # pro rata to size
    assert [o["shares"] for o in orders] == [1000, 1000]                                          # input untouched


def test_upsize_is_a_noop_when_cash_is_already_fine_and_respects_sector_cap():
    nav, px = 1_000_000.0, {"AAA": 100.0, "BBB": 100.0}
    big = [_o("AAA", "BUY", 4900, 100.0), _o("BBB", "BUY", 4900, 100.0)]                          # 98% invested
    assert analytics.upsize_buys_for_cash(big, {}, px, nav, nav, {}, {"AAA": "stock", "BBB": "stock"}, {}) == big
    same = analytics.upsize_buys_for_cash([_o("AAA", "BUY", 1000, 100.0), _o("BBB", "BUY", 1000, 100.0)], {}, px, nav, nav,
                                          {"AAA": "X", "BBB": "X"}, {"AAA": "stock", "BBB": "stock"}, {})
    assert sum(o["notional"] for o in same) <= 0.35 * nav + 1                                      # one sector: 35% cap binds


# ---- recovery paths ------------------------------------------------------------------------------------ #
def test_recovery_keeps_cio_edits_when_upsizing_fixes_the_breach():
    pre = [_o("A", "BUY", 10, 10.0), _o("B", "BUY", 10, 10.0)]
    edited = [_o("A", "BUY", 10, 10.0), _o("B", "BUY", 5, 10.0)]
    up = lambda b: [dict(o, shares=o["shares"] + 5) for o in b]
    calls = []
    out, path, left = orchestrator.recover_cio_edits(pre, edited, up, lambda b: calls.append(1) or [])
    assert path == "cio_edits_kept_upsized" and out == up(edited) and left == [] and len(calls) == 1   # re-checked exactly once


def test_recovery_reverts_to_the_risk_approved_list_when_it_still_fails_or_nothing_upsizes():
    pre = [_o("A", "BUY", 10, 10.0), _o("STOP", "SELL", 3, 10.0, True)]
    edited = [_o("STOP", "SELL", 3, 10.0, True)]
    out, path, left = orchestrator.recover_cio_edits(pre, edited, lambda b: [dict(o, shares=o["shares"] + 1) for o in b],
                                                     lambda b: ["cash 6.5% > 5% (IPS A12)"])
    assert path == "reverted_to_risk_approved" and out == pre and left == ["cash 6.5% > 5% (IPS A12)"]
    out, path, _ = orchestrator.recover_cio_edits(pre, edited, lambda b: b, lambda b: ["x"])       # upsize found no room
    assert path == "reverted_to_risk_approved" and out == pre


# ---- soft beta band ------------------------------------------------------------------------------------ #
def _panel(seed=3, n=9, T=300):
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, T)
    cols = {"SPY": m}
    for i in range(n):
        cols[f"S{i}"] = 0.95 * m + rng.normal(0, 0.008, T)
    cols["NEG"] = -0.8 * m + rng.normal(0, 0.008, T)                                            # an ERX-like negative-beta name
    df = pd.DataFrame(cols)
    return pd.DataFrame(100 * np.exp(df.cumsum().values), index=pd.bdate_range("2025-01-01", periods=T), columns=df.columns)


def test_neutral_band_is_soft_with_lower_bound_half_and_other_regimes_stay_hard():
    assert config.REGIME_RULES["neutral"]["beta"] == (0.5, 1.2)
    assert config.BETA_BAND_SOFT == {"risk_on": False, "neutral": True, "risk_off": False}
    assert config.REGIME_RULES["risk_on"]["beta"] == (1.2, 1.6) and config.REGIME_RULES["risk_off"]["beta"] == (0.3, 0.7)


def test_soft_band_lets_a_high_conviction_negative_beta_name_into_the_book():
    px = _panel()
    names = [c for c in px.columns if c != "SPY"]
    sec = {t: f"Sec{i}" for i, t in enumerate(names)}
    at = {t: "stock" for t in names}
    conv = {t: 0.3 for t in names}
    conv["NEG"] = 1.0
    kw = dict(prices=px, sectors=sec, asset_types=at, equity_exposure=0.98, regime="neutral", vix=17.0, max_names=10,
              min_weight=0.005, convictions=conv)
    hard = analytics.optimise(**kw, beta_band=(0.9, 1.2), beta_soft=False)
    soft = analytics.optimise(**kw)                                                              # config default: neutral = soft (0.5, 1.2)
    assert soft.get("NEG", 0) > hard.get("NEG", 0) + 0.05, (soft.get("NEG"), hard.get("NEG"))
    r = np.log(px).diff().dropna()
    beta = lambda w: sum(x * np.cov(r[t], r["SPY"])[0, 1] / r["SPY"].var() for t, x in w.items())
    assert beta(soft) >= 0.45                                                                    # lower bound 0.5 still pulls the book up


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
