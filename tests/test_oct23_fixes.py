"""Oct 23 P0 + ensemble decisions: marking rule, SPY same-date, label fallback, blend/priors, politicians off, sector
quotas, DST labels, live tracker. No network, no LLM.   Run:  /opt/miniconda3/bin/python3 tests/test_oct23_fixes.py"""
import datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PUBLISH"] = "0"
import numpy as np
import pandas as pd
import config, data, valuation, tracker, screener, nyse_calendar as cal
from strategies import ensemble as En
from committee.sentiment_analyst import SentimentAnalyst

SNAP_TS = "2026-10-23T00:03:04+02:00"          # the live init: 18:03 ET on Oct 22 = after the Oct 22 close


def _book():
    pos = {"VLO": {"shares": 302, "avg_cost": 426.28}}
    trades = pd.DataFrame([{"date": "2026-10-23", "hour": "00:03:03", "symbol": "VLO", "price": 426.28, "id": 1}])
    snaps = pd.DataFrame([{"date": "2026-10-23", "ts": SNAP_TS, "total_equity": 1e6, "benchmark_px": 763.96}])
    return pos, trades, snaps


def _close(last, vlo, spy):
    idx = pd.to_datetime(["2026-10-20", "2026-10-21"] + (["2026-10-22"] if last >= 22 else []) + (["2026-10-23"] if last >= 23 else []))
    n = len(idx)
    return pd.DataFrame({"VLO": ([420.0, 421.0] + [vlo, vlo + 1])[:n], "SPY": ([769.05, 773.09] + [763.96, spy])[:n]}, index=idx)


# ---- P0.1 valuation: never a close dated before the fill session ------------------------------------------------ #
def test_fill_session_is_the_ny_session_of_the_fill():
    assert valuation.session_of("2026-10-23 00:03:03") == dt.date(2026, 10, 22)      # 18:03 ET: same-day close
    assert valuation.session_of("2026-10-23 14:30:00") == dt.date(2026, 10, 22)      # 08:30 ET: pre-open -> yesterday's close
    assert valuation.session_of("2026-10-23 16:00:00") == dt.date(2026, 10, 23)      # 10:00 ET: open
    assert valuation.session_of("2026-10-24 12:00:00") == dt.date(2026, 10, 23)      # Saturday -> Friday


def test_position_bought_at_oct22_close_is_not_marked_at_oct21():
    pos, trades, snaps = _book()
    m = valuation.mark(pos, _close(21, 0, 0), trades, snaps)                         # prices.csv ends Oct 21
    assert m.px["VLO"] == 426.28 and "VLO" in m.stale                                # fill price + flagged
    nav = (1e6 - 302 * 426.28) + 302 * m.px["VLO"]
    assert abs(nav / 1e6 - 1) < 1e-6                                                 # no fake -0.20%
    m2 = valuation.mark(pos, _close(22, 430.0, 0), trades, snaps)                    # Oct 22 close arrived
    assert m2.px["VLO"] == 430.0 and not m2.stale


# ---- P0.2 SPY on the same date as the fund ----------------------------------------------------------------------- #
def test_spy_is_measured_from_inception_to_the_fund_mark_date():
    pos, trades, snaps = _book()
    old_bug = 773.09 / 763.96 - 1                                                    # Oct 21 close / Oct 22 inception
    assert old_bug > 0.011                                                           # = the fake +1.20%
    m = valuation.mark(pos, _close(21, 0, 0), trades, snaps)
    assert m.spy_return == 0.0 and m.mark_date == dt.date(2026, 10, 22)              # data older than inception: flat, not +1.2%
    m = valuation.mark(pos, _close(22, 430.0, 0), trades, snaps)
    assert abs(m.spy_return) < 1e-12 and m.mark_date == dt.date(2026, 10, 22)
    m = valuation.mark(pos, _close(23, 430.0, 770.0), trades, snaps)
    assert abs(m.spy_return - (770.0 / 763.96 - 1)) < 1e-12 and m.mark_date == dt.date(2026, 10, 23)


def test_pct_never_shows_negative_zero():
    assert valuation.pct(-1e-7) == "0.00%" and valuation.pct(0.012) == "+1.20%" and valuation.pct(-0.0035) == "-0.35%"
    assert valuation.pct(-0.000004) == "0.00%" and valuation.pct(-0.0006) == "-0.06%"


# ---- P0.3 names ---------------------------------------------------------------------------------------------------- #
def test_name_survives_a_failed_universe_read_and_falls_back_to_the_trade_name():
    saved, loaded, csv = dict(data._NAMES), data._NAMES_LOADED[0], config.UNIVERSE_CSV
    try:
        data._NAMES.clear(); data._NAMES_LOADED[0] = False
        config.UNIVERSE_CSV = "/nonexistent.csv"
        assert data.label("VLO") == "VLO" and data.label("VLO", "Valero Energy") == "VLO · Valero Energy"
        config.UNIVERSE_CSV = csv                                                    # the file is back: names load next call
        assert data.label("VLO") == "VLO · Valero Energy"
    finally:
        data._NAMES.clear(); data._NAMES.update(saved); data._NAMES_LOADED[0] = loaded; config.UNIVERSE_CSV = csv


def test_refresh_when_latest_close_is_missing():
    c = pd.DataFrame({"SPY": [1.0, 2.0]}, index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    real = cal.last_completed_session
    try:
        cal.last_completed_session = lambda now=None: dt.date(2026, 10, 22)
        assert data.close_missing(c)
        cal.last_completed_session = lambda now=None: dt.date(2026, 10, 21)
        assert not data.close_missing(c)
    finally:
        cal.last_completed_session = real


# ---- ensemble decisions ------------------------------------------------------------------------------------------- #
def test_blend_is_75_25():
    assert config.ENSEMBLE["blend"] == 0.75


def test_snapshot_only_prior_is_half_of_equal_capped_at_5pct_until_ic_exists():
    names = [f"s{i}" for i in range(17)]
    hist = {n: (i < 9) for i, n in enumerate(names)}                                  # 8 snapshot-only, 9 price-based
    table = {n: {"n": 0, "ic": None, "t": None, "shrunk_ic": 0.0, "validated": False} for n in names}
    w, _ = En.weights_from_table(table, hist)
    snap = [n for n in names if not hist[n]]
    assert all(abs(w[n] - 0.5 / 17) < 1e-4 for n in snap) and all(w[n] <= 0.05 for n in snap)
    assert all(abs(w[n] - 1 / 17) < 1e-4 for n in names if hist[n])
    few = {n: False for n in names[:6]}                                               # 0.5/6 = 8.3% -> capped at 5%
    w2, _ = En.weights_from_table({n: table[n] for n in few}, few)
    assert all(abs(v - 0.05) < 1e-4 for v in w2.values())
    table["s10"] = {"n": 60, "ic": 0.05, "t": 3.0, "shrunk_ic": 0.04, "validated": True}   # measured IC replaces the prior
    w3, _ = En.weights_from_table(table, hist)
    assert w3["s10"] > 0.05 and w3["s10"] != w["s10"]


def test_politician_signal_is_off_and_not_in_the_sentiment_prompt():
    assert config.POLITICIAN_SIGNAL is False
    ctx = {"tickers": ["NVDA"], "news": {"NVDA": [{"title": "NVDA beats", "ts": "2026-10-22 09:00", "source": "Reuters", "finbert": 0.4}]},
           "earnings": {}, "politician_buys": {"NVDA": {"buyers": ["Nancy Pelosi"], "n_buys": 2, "last": "2026-10-13", "amounts": []}}}
    p = SentimentAnalyst().build_prompt(ctx)
    assert "CONGRESS" not in p and "Pelosi" not in p
    assert "politician" not in SentimentAnalyst.role_prompt.lower()


# ---- screener sector quotas ----------------------------------------------------------------------------------------- #
def _sectors(sizes):
    rng = np.random.default_rng(1)
    sec = {f"{s}{i}": s for s, n in sizes.items() for i in range(n)}
    return pd.Series(rng.normal(size=len(sec)).copy(), index=list(sec)), pd.Series(sec)


def test_stage1_quota_top5_per_sector_plus_5_best_overall():
    sizes = {f"S{k}": 40 for k in range(10)}
    sizes["Tiny"] = 3                                                                  # a sparse sector keeps all its names
    sc, sec = _sectors(sizes)
    pick = screener.quota_pick(sc, sec, 60, floor=5, overall=5)
    assert len(pick) == 60
    cnt = sec[pick].value_counts()
    assert set(cnt.index) == set(sizes) and (cnt >= 3).all() and cnt["Tiny"] == 3
    assert all(cnt[k] >= 5 for k in sizes if k != "Tiny")
    best5 = [t for t in sc.sort_values(ascending=False).index if t not in
             [x for _, g in sc.groupby(sec) for x in g.sort_values(ascending=False).index[:5]]][:5]
    assert set(best5) <= set(pick)


def test_stage2_quota_1_to_5_per_sector_then_by_score():
    sizes = {f"S{k}": 6 for k in range(11)}
    sizes["Energy"] = 1                                                                # only 1 name: still represented
    sc, sec = _sectors(sizes)
    sc = sc.copy(); sc["Energy0"] = -9.0                                                               # worst score, kept by the floor
    pick = screener.quota_pick(sc, sec, 30, floor=1, cap=5)
    cnt = sec[pick].value_counts()
    assert len(pick) == 30 and set(cnt.index) == set(sizes) and cnt.max() <= 5 and cnt.min() >= 1 and "Energy0" in pick
    # a sector with nothing but strong names is capped at 5
    sc2, sec2 = _sectors({"A": 20, "B": 4, "C": 4, "D": 4, "E": 4, "F": 4})
    sc2 = sc2 + (sec2 == "A") * 10.0
    p2 = screener.quota_pick(sc2, sec2, 24, floor=1, cap=5)
    assert len(p2) == 24 and sec2[p2].value_counts()["A"] >= 5


# ---- DST labels (item 10) -------------------------------------------------------------------------------------------- #
def test_session_labels_follow_the_ny_clock_across_the_us_eu_dst_gap():
    assert cal.session_labels("2026-10-22") == {"pre_open": "15:30", "close": "22:00", "mark": "21:45", "mark_et": "15:45"}
    for d in ("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"):   # EU on CET, US still on EDT
        assert cal.session_labels(d) == {"pre_open": "14:30", "close": "21:00", "mark": "20:45", "mark_et": "15:45"}, d
    assert cal.session_labels("2026-11-02")["pre_open"] == "15:30" and cal.session_labels("2026-11-02")["close"] == "22:00"
    assert cal.session_labels("2026-11-27")["close"] == "19:00"                        # half-day: 13:00 ET


# ---- live tracker (item 5) -------------------------------------------------------------------------------------------- #
def test_tracker_ranks_the_three_variants_against_forward_returns():
    tick = [f"T{i}" for i in range(12)]
    fwd = {t: 0.01 * i for i, t in enumerate(tick)}                                    # T11 best
    close = pd.DataFrame({t: [100.0, 100.0 * (1 + fwd[t])] for t in tick}, index=pd.to_datetime(["2026-10-22", "2026-10-23"]))
    sig = {t: {"score": 0.1 * i, "breakdown": {"ensemble": 0.1 * i, "legacy": -0.1 * i}} for i, t in enumerate(tick)}
    res = tracker.compute([{"started_at": SNAP_TS, "signals": sig}], close)
    v = res["variants"]
    assert res["n_runs"] == 1 and v["ensemble"]["ic"] > 0.99 and v["legacy"]["ic"] < -0.99 and v["deployed"]["top_excess"] > 0
    assert v["legacy"]["top_excess"] < 0 and "ensemble IC" in tracker.line(res)
    assert tracker.compute([{"started_at": SNAP_TS, "signals": sig}], close.iloc[:1]) is None       # nothing after the run yet
    assert "starts once" in tracker.line(None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
