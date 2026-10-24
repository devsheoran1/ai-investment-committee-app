"""Oct 23: profit-taking without cash (B4: drift trims paired with the best unheld idea, B5: trailing stop) + vol-managed sizing.
Run:  /opt/miniconda3/bin/python3 tests/test_profit_taking.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import analytics, config
from orchestrator import _trailing_stop

NAV = 1_000_000.0


def _book(weights, px=None):
    px = px or {}
    pos = {t: {"shares": int(w * NAV / px.get(t, 100.0)), "avg_cost": px.get(t, 100.0), "opened_date": "2026-10-01"}
           for t, w in weights.items()}
    return pos


def _plan(target, pos, px, ideas, cash=0.0, composite=None, sectors=None, **kw):
    sectors = sectors or {t: "S" + t[0] for t in set(target) | set(pos) | set(ideas)}
    comp = composite or {t: {"composite": 0.5} for t in px}
    at = {t: "stock" for t in px}
    return analytics.plan_orders(target, pos, px, NAV, cash, kw.pop("forced", {}), False, comp, kw.pop("untradeable", {}),
                                 sectors=sectors, asset_types=at, buy_ideas=ideas, **kw)


def _scenario():
    """A is 14% above its 8% target after a rally (a 6pp drift trim); the rest of the book is on target."""
    w = {"A": 0.14, **{t: 0.085 for t in "BCDEFGHIJK"}}
    px = {t: 100.0 for t in list(w) + ["N1", "N2", "N3"]}
    pos = _book(w, px)
    cash = NAV - sum(p["shares"] * 100 for p in pos.values())
    target = {**{t: 0.09 for t in w}, "A": 0.08}               # others a hair under target: inside the drift band
    return w, px, pos, cash, target


def test_a_drift_trim_is_paired_with_the_best_unheld_idea_sized_to_the_proceeds():
    w, px, pos, cash, target = _scenario()
    orders, skipped = _plan(target, pos, px, ["N1", "N2"], cash)
    trim = next(o for o in orders if o["ticker"] == "A")
    buy = next(o for o in orders if o.get("pair_of") == "A")
    assert trim["side"] == "SELL" and not trim["full_exit"] and trim["pair_buy"] == "N1"
    assert buy["ticker"] == "N1" and buy["side"] == "BUY" and buy["is_new"]
    assert abs(buy["notional"] - trim["notional"]) <= 100                      # proceeds go to the new idea, not to cash
    assert len(orders) <= config.MAX_TRADES_PER_DAY
    assert orders.index(trim) < orders.index(buy)                              # sells first


def test_pairing_off_or_no_ideas_leaves_the_old_behaviour():
    w, px, pos, cash, target = _scenario()
    config.PAIR_TRIMS = False
    try:
        orders, _ = _plan(target, pos, px, ["N1"], cash)
    finally:
        config.PAIR_TRIMS = True
    assert not any(o.get("pair_of") for o in orders)
    orders, _ = _plan(target, pos, px, [], cash)
    assert not any(o.get("pair_of") for o in orders)


def test_ideas_that_are_held_untradeable_etfs_or_forced_are_skipped():
    w, px, pos, cash, target = _scenario()
    orders, _ = _plan(target, pos, px, ["B", "N2", "N3"], cash, untradeable={"N2": "halted"})
    buy = next(o for o in orders if o.get("pair_of") == "A")
    assert buy["ticker"] == "N3"                                               # B is already held, N2 is halted
    px["ETF1"] = 50.0
    at = {t: ("leveraged_etf" if t == "ETF1" else "stock") for t in px}
    o2, _ = analytics.plan_orders(target, pos, px, NAV, cash, {}, False, {t: {"composite": 0.5} for t in px}, {},
                                  sectors={t: "S" + t[0] for t in px}, asset_types=at, buy_ideas=["ETF1", "N1"])
    assert next(o for o in o2 if o.get("pair_of") == "A")["ticker"] == "N1"


def test_buys_the_plan_already_has_absorb_the_proceeds_so_no_second_buy_is_added():
    w, px, pos, cash, target = _scenario()
    target = {**target, "N2": 0.06}                                            # the PM already wants a new name
    orders, _ = _plan(target, pos, px, ["N1"], cash)
    assert not any(o.get("pair_of") for o in orders)
    assert any(o["ticker"] == "N2" and o["side"] == "BUY" for o in orders)


def test_pair_respects_name_and_sector_caps():
    w, px, pos, cash, target = _scenario()
    # sector SA = A+B+C+D = 39.5% now, 33.5% after the trim; N1 (also SA) would break the 35% cap -> the next idea is used
    sectors = {t: "SA" if t in ("A", "B", "C", "D") else "S" + t for t in px}
    sectors.update({"N1": "SA", "N2": "SX"})
    orders, _ = _plan(target, pos, px, ["N1", "N2"], cash, sectors=sectors)
    buy = next(o for o in orders if o.get("pair_of") == "A")
    assert buy["ticker"] == "N2"
    assert buy["notional"] / NAV <= config.RISK_LIMITS["max_single_weight"]


def test_a_pair_needs_two_slots_otherwise_the_trim_goes_alone():
    w, px, pos, cash, target = _scenario()
    # four forced exits already use 4 of the 5 daily slots -> only one slot is left for the trim
    forced = {t: "stop-loss" for t in ("B", "C", "D", "E")}
    target2 = {t: v for t, v in target.items() if t not in forced}
    orders, skipped = _plan(target2, pos, px, ["N1", "N2"], cash, forced=forced)
    assert len(orders) <= config.MAX_TRADES_PER_DAY
    assert [o["ticker"] for o in orders if o["side"] == "SELL"] == ["B", "C", "D", "E", "A"] or \
        {o["ticker"] for o in orders} == {"B", "C", "D", "E", "A"}
    assert not any(o.get("pair_of") for o in orders)                           # 4 forced + the trim = 5: no slot for the buy
    assert any("paired buy" in s["reason"] and s["ticker"] == "N1" for s in skipped)


def test_pairs_compete_for_the_five_slots_by_priority_with_other_trades():
    w, px, pos, cash, target = _scenario()
    w2 = dict(w); w2["B"] = 0.145
    pos = _book(w2, px)
    cash = NAV - sum(p["shares"] * 100 for p in pos.values())
    target = {**target, "B": 0.08}
    orders, skipped = _plan(target, pos, px, ["N1", "N2", "N3"], cash)
    assert len(orders) <= config.MAX_TRADES_PER_DAY
    sells = [o for o in orders if o["side"] == "SELL"]
    pairs = [o for o in orders if o.get("pair_of")]
    assert len(pairs) <= len(sells) and all(p["pair_of"] in {s["ticker"] for s in sells} for p in pairs)
    assert len({p["ticker"] for p in pairs}) == len(pairs)                     # one distinct idea per trim


def test_init_day_never_pairs():
    w, px, pos, cash, target = _scenario()
    orders, _ = analytics.plan_orders(target, pos, px, NAV, cash, {}, True, {t: {"composite": 0.5} for t in px}, {},
                                      sectors={t: "S" + t[0] for t in px}, asset_types={t: "stock" for t in px}, buy_ideas=["N1"])
    assert not any(o.get("pair_of") for o in orders)


# ---------------------------------------------------------------------------------------------- trailing stop
def test_trailing_drawdown_peak_is_cost_closes_since_entry_or_live_price():
    dd, peak = analytics.trailing_drawdown(100.0, 92.0, np.array([101.0, 108.0, 104.0]))
    assert peak == 108.0 and abs(dd - (92 / 108 - 1)) < 1e-12
    dd, peak = analytics.trailing_drawdown(100.0, 95.0, np.array([]))
    assert peak == 100.0 and abs(dd + 0.05) < 1e-12
    dd, peak = analytics.trailing_drawdown(100.0, 130.0, np.array([110.0]))
    assert peak == 130.0 and dd == 0.0                                          # live price above every close: no drawdown


def test_trailing_stop_fires_at_minus_8_percent_from_the_peak_since_entry_only():
    idx = pd.bdate_range("2026-09-01", periods=30)
    s = pd.Series(100.0, index=idx)
    s.iloc[5] = 140.0                                                          # a peak BEFORE we bought: must not count
    s.iloc[15] = 112.0                                                         # the peak since entry
    pos = {"shares": 10, "avg_cost": 100.0, "opened_date": str(idx[10].date())}
    assert _trailing_stop(pos, 112.0 * 0.881, s) is None                       # -11.9% from the peak
    why = _trailing_stop(pos, 112.0 * 0.879, s)                                # -12.1%
    assert why and "trailing stop" in why and "-12" in why
    assert _trailing_stop({**pos, "opened_date": str(idx[0].date())}, 112.0 * 0.919, s)    # peak 140 if bought earlier
    assert config.TRAILING_STOP == -0.12 and config.STOP_LOSS == -0.10          # the IPS A17 stop is untouched


# ---------------------------------------------------------------------------------------------- vol-managed sizing
def _vol_prices():
    rng = np.random.default_rng(2)
    vols = {"CALM": 0.006, "MID1": 0.012, "MID2": 0.012, "MID3": 0.012, "WILD": 0.04}
    px = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, v, 120))) for t, v in vols.items()})
    px["ETF"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.05, 120)))
    return px


def test_vol_managed_scales_positive_stock_convictions_inside_the_clip():
    px = _vol_prices()
    conv = {"CALM": 0.5, "MID1": 0.5, "MID2": 0.5, "MID3": 0.5, "WILD": 0.5, "ETF": 0.5}
    at = {t: "stock" for t in conv}; at["ETF"] = "leveraged_etf"
    out = analytics.vol_managed(conv, px, at)
    cfg = config.VOL_MANAGED
    assert out["CALM"] > conv["CALM"] and out["WILD"] < conv["WILD"]
    assert out["WILD"] >= conv["WILD"] * cfg["floor"] - 1e-3 and out["CALM"] <= min(1.0, conv["CALM"] * cfg["cap"]) + 1e-3
    assert out["ETF"] == conv["ETF"]                                           # ETFs have their own vol-targeted caps
    neg = analytics.vol_managed({**conv, "WILD": -0.6}, px, at)
    assert neg["WILD"] == -0.6                                                 # negatives untouched
    assert all(v <= 1.0 for v in out.values())


def test_vol_managed_can_be_switched_off_and_needs_enough_names():
    px = _vol_prices()
    conv = {"CALM": 0.5, "WILD": 0.5}
    at = {t: "stock" for t in conv}
    assert analytics.vol_managed(conv, px, at) == conv                        # < 3 names: left alone
    config.VOL_MANAGED["enabled"] = False
    try:
        c5 = {t: 0.5 for t in px.columns if t != "ETF"}
        assert analytics.vol_managed(c5, px, {t: "stock" for t in c5}) == c5
    finally:
        config.VOL_MANAGED["enabled"] = True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
