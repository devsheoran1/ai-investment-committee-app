"""Oct 23: the 17 strategies (A1). Each: shape, sector-neutral z in +-3, one-line reason, NO look-ahead, graceful when data is absent.
Run:  /opt/miniconda3/bin/python3 tests/test_strategies.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import strategies
from strategies import MODULES, registry
from strategies.base import sector_neutral_z
import synth

PRICE_ONLY = ["xs_momentum_12_1", "ts_trend", "industry_momentum", "residual_momentum", "high_52w", "st_reversal_1w",
              "low_beta", "high_volume_premium", "sector_rotation", "etf_trend_200dma", "turn_of_month"]


def test_seventeen_modules_one_file_each():
    assert len(MODULES) == 17 and len(set(MODULES)) == 17
    for m in MODULES:
        assert os.path.exists(os.path.join(os.path.dirname(strategies.__file__), m + ".py")), m
    for s in registry():
        assert s.name and s.desc and "\n" not in s.desc


def test_price_strategies_shape_zscore_bounds_and_sector_neutrality():
    d, (close, *_rest) = synth.data()
    for s in registry(PRICE_ONLY):
        raw = s.raw(d)
        assert raw.index.equals(close.index), s.name
        if s.tilt:
            continue
        z = s.z(d, raw)
        assert float(z.abs().max().max()) <= 3.0 + 1e-9, s.name                      # winsorised
        assert z.iloc[-1].notna().sum() >= 8, s.name
        if s.universe == "stock" and s.neutral and s.name != "sector_rotation":      # mean z ~ 0 inside each sector
            last = z.iloc[-1].dropna()
            for sec in synth.SECTORS:
                names = [t for t in last.index if d.sector.get(t) == sec]
                assert abs(last[names].mean()) < 0.35, (s.name, sec, last[names].mean())
        t = z.iloc[-1].dropna().index[0]
        why = s.reason(t, raw.iloc[-1][t], float(z.iloc[-1][t]))
        assert why.startswith(s.name + ":") and "\n" not in why and "(z " in why


def test_etf_strategy_only_touches_leveraged_etfs_and_is_absolute():
    d, (close, vol, opn, uni, at, stocks, etfs) = synth.data()
    s = registry(["etf_trend_200dma"])[0]
    z = s.z(d)
    assert set(z.columns) == set(etfs)
    # absolute: a ETF 20% above its 200dma scores +2 (distance / 10%), not a relative rank
    raw = s.raw(d)
    assert np.allclose(z.iloc[-1].dropna(), (raw.iloc[-1].dropna() / 0.10).clip(-3, 3))


def test_no_lookahead_signal_at_k_ignores_everything_after_k():
    d, (close, vol, opn, uni, at, stocks, etfs) = synth.data(T=420)
    k = 330
    d_cut = strategies.build_data(close.iloc[:k + 1], vol.iloc[:k + 1], uni, at)
    for s in registry(PRICE_ONLY):
        if s.tilt:
            continue
        full, cut = s.z(d).iloc[k], s.z(d_cut).iloc[-1]
        both = full.dropna().index.intersection(cut.dropna().index)
        assert len(both) > 5, s.name
        assert np.allclose(full[both], cut[both], atol=1e-9), f"{s.name} changes when future prices are removed"


def test_pead_sue_uses_next_bar_and_decays_over_sixty_days():
    d, (close, *_r) = synth.data()
    day = close.index[200]
    d.earn_events = {"EN0": [{"date": str(day.date()), "surprise_pct": 10.0}], "EN1": [{"date": str(day.date()), "surprise_pct": 400.0}]}
    s = registry(["pead_sue"])[0]
    raw = s.raw(d)
    assert pd.isna(raw["EN0"].iloc[200])                                   # same day: not yet known
    assert abs(raw["EN0"].iloc[201] - 10.0) < 1e-9                         # first bar after the report
    assert abs(raw["EN0"].iloc[231] - 5.0) < 1e-9                          # halfway through the 60 bars
    assert raw["EN0"].iloc[262] == 0.0 or pd.isna(raw["EN0"].iloc[262]) or abs(raw["EN0"].iloc[262]) < 1e-9
    assert raw["EN1"].iloc[201] == 50.0                                    # surprise clipped at +-50%


def test_earnings_premium_five_days_into_the_report_and_unknown_is_nan():
    d, (close, *_r) = synth.data()
    ev = close.index[300]
    d.earn_events = {"EN0": [{"date": str(ev.date()), "surprise_pct": 1.0}]}
    raw = registry(["earnings_premium"])[0].raw(d)
    assert abs(raw["EN0"].iloc[299] - 5 / 6) < 1e-9                       # the day before the report
    assert abs(raw["EN0"].iloc[295] - 1 / 6) < 1e-9 and raw["EN0"].iloc[294] == 0
    assert raw["EN0"].iloc[301] == 0
    assert raw["EN7"].isna().all()                                         # no calendar for this name -> NaN, not 0
    # a report scheduled after the last bar counts business days ahead
    nxt = (close.index[-1] + pd.offsets.BDay(3)).date()
    d.next_earnings = {"EN2": str(nxt)}
    assert abs(registry(["earnings_premium"])[0].raw(d)["EN2"].iloc[-1] - 3 / 6) < 1e-9


def test_snapshot_strategies_fill_only_the_last_row():
    d, _ = synth.data()
    d.revisions = {"EN0": {"rev_30d_0q": 0.10, "rev_30d_1y": 0.05}, "EN1": {"rev_30d_0q": None, "rev_30d_1y": None}}
    d.fundamentals = {"EN0": {"gross_margin": 0.6, "roa": 0.1}, "EN1": {"gross_margin": 0.2, "roa": 0.02}, "EN2": {}}
    d.news = {"EN0": [{"finbert": 0.6}, {"finbert": 0.2}], "EN1": [], "EN2": [{"finbert": -0.5}]}
    d.insiders = {"EN0": {"buyers": 2, "buy_usd": 500000, "sell_usd": 0}, "EN1": {"buyers": 0, "buy_usd": 0, "sell_usd": 100000}}
    for name in ("eps_revisions", "quality_gp", "news_surprise", "insider_buying"):
        s = registry([name])[0]
        assert not s.history
        raw = s.raw(d)
        assert raw.iloc[:-1].isna().all().all(), name                      # no history is ever invented
    assert abs(registry(["eps_revisions"])[0].raw(d)["EN0"].iloc[-1] - 0.075) < 1e-9
    q = registry(["quality_gp"])[0].raw(d).iloc[-1]
    assert q["EN0"] > q["EN1"] and pd.isna(q["EN2"])
    n = registry(["news_surprise"])[0].raw(d).iloc[-1]
    assert abs(n["EN0"] - 0.4) < 1e-9 and n["EN2"] < 0 and pd.isna(n["EN1"])
    ib = registry(["insider_buying"])[0].raw(d).iloc[-1]
    assert ib["EN0"] > ib["EN1"] == 0.0                                     # sales alone are not a buy signal


def test_missing_inputs_make_the_signal_absent_not_zero():
    d, _ = synth.data()
    for name in ("eps_revisions", "quality_gp", "news_surprise", "insider_buying", "pead_sue", "earnings_premium"):
        raw = registry([name])[0].raw(d)
        assert raw.isna().all().all(), name


def test_turn_of_month_flags_last_day_and_first_three():
    from strategies.turn_of_month import flags, in_window
    idx = pd.bdate_range("2025-03-24", "2025-04-10")
    f = flags(idx)
    on = [str(x.date()) for x in f.index[f == 1]]
    assert on == ["2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03"], on
    assert in_window("2025-04-02") and not in_window("2025-04-15") and in_window("2025-04-30") and not in_window("2025-04-05")
    s = registry(["turn_of_month"])[0]
    assert s.tilt


def test_insider_and_news_run_skips_a_strategy_with_no_data():
    d, (close, *_r) = synth.data()
    d.insiders = None                                                       # EDGAR down
    r = strategies.run(d, ["EN0", "EN1"])
    assert r is not None and "insider_buying" in r["skipped"]
    assert "insider_buying" not in r["scores"]["EN0"]["strategies"]
    assert "insider_buying" not in r["weights"]


def test_sector_neutral_z_falls_back_for_tiny_sectors():
    raw = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0]],
                       columns=list("ABCDEFGHIJK"))
    sec = pd.Series(["x"] * 4 + ["y"] * 6 + ["z"], index=list("ABCDEFGHIJK"))       # x has 4 (< 5), z has 1
    z = sector_neutral_z(raw, sec)
    assert abs(z[list("EFGHIJ")].iloc[0].mean()) < 1e-9                    # y (6 names) is z-scored on its own
    assert z["K"].iloc[0] > 0 and abs(z.iloc[0]).max() <= 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
