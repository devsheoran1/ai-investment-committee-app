"""Oct 27: leveraged ETFs are scored on their underlying index (no saturation at +-1.0); composite treats n/a as absent, not 0.
Run:  /opt/miniconda3/bin/python3 tests/test_etf_scoring.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analytics, config
from committee.quant_analyst import QuantAnalyst
import test_quant_models as tq


def _ctx():
    ctx = tq._ctx(n=10)
    ctx["tickers"] = ctx["tickers"] + ["ERX", "SOXL", "NAIL", "SPXU"]
    for t, mom, tr in (("ERX", 900.0, 1), ("SOXL", 3000.0, -1), ("NAIL", -900.0, -1), ("SPXU", -80.0, -1)):
        ctx["technicals"][t] = {"mom_12_1_pct": mom, "px_vs_50dma_pct": 20.0 * tr, "px_vs_200dma_pct": 30.0 * tr, "rsi_14": 55,
                                "macd_hist_pct": 0.5 * tr, "vol_3m_ann_pct": 90.0, "ret_5d_pct": 1.0}
        ctx["forecast"]["forecasts"][t] = {"pred_5d_return_pct": 0.0, "p_up": 0.5}
    ctx["underlying_tech"] = {t: {"mom_12_1_pct": m, "px_vs_50dma_pct": 3.0, "px_vs_200dma_pct": 6.0, "rsi_14": 55,
                                  "macd_hist_pct": 0.2, "vol_3m_ann_pct": 18.0, "underlying": u}
                              for t, m, u in (("ERX", 12.0, "XLE"), ("SOXL", 25.0, "SOXX"), ("NAIL", -5.0, "ITB"), ("SPXU", 12.0, "SPY"))}
    return ctx


def test_no_etf_sits_at_plus_minus_one_even_with_extreme_own_series_and_extreme_ensemble():
    q = QuantAnalyst()
    ctx = _ctx()
    ctx["ensemble"] = {"scores": {t: {"score": s, "votes_long": 4, "votes_short": 0, "n_strategies": 4, "min_votes": 3, "strategies": {}}
                                  for t, s in (("ERX", 1.0), ("SOXL", -1.0), ("NAIL", -1.0), ("SPXU", 1.0))},
                       "allocated": 1.0, "tom_tilt": 0.0, "weights": {}, "mode": "x", "asof": "d"}
    bd = q.breakdown(ctx)
    for t in ("ERX", "SOXL", "NAIL", "SPXU"):
        assert abs(bd[t]["rule_score"]) <= QuantAnalyst.ETF_CAP < 1.0, (t, bd[t])
        assert bd[t]["breakdown"]["scored_on"]
        assert abs(bd[t]["breakdown"]["momentum_12_1"]) < 1.0
    out = q.validate({"signals": {t: {"score": -1.0 if t != "ERX" else 1.0} for t in ctx["tickers"]}}, ctx)["signals"]
    assert all(abs(out[t]["score"]) <= QuantAnalyst.ETF_CAP for t in ("ERX", "SOXL", "NAIL", "SPXU"))   # an LLM +-1.0 is capped too


def test_etf_is_scored_on_the_underlying_not_its_own_3x_series():
    q = QuantAnalyst()
    ctx = _ctx()
    a = q.breakdown(ctx)["SOXL"]["breakdown"]
    assert a["trend"] == 1.0 and a["macd"] == 0.5          # SOXL's own series is below its 200dma; SOXX is above
    ctx["underlying_tech"] = {}
    assert q.breakdown(ctx)["SOXL"]["breakdown"]["trend"] == -1.0   # no underlying data: old behaviour, not a crash


def test_inverse_etf_gets_the_flipped_index_signal():
    bd = QuantAnalyst().breakdown(_ctx())
    assert bd["SPXU"]["breakdown"]["trend"] == -1.0 and bd["SPXU"]["rule_score"] < 0


def test_composite_redistributes_na_weights_instead_of_scoring_them_zero():
    W = config.SIGNAL_WEIGHTS
    comp = analytics.composite_scores(["ERX"], {}, {"ERX": "leveraged_etf"}, {}, {"ERX": {"score": 0.8}},
                                      {"ERX": {"score": None}}, {}, "neutral")["ERX"]
    assert set(comp["na"]) == {"sentiment", "research"}
    exp = (W["quant"] * 0.8 + W["macro"] * 0.0) / (W["quant"] + W["macro"])
    assert abs(comp["composite"] - exp) < 1e-3 and abs(sum(comp["weights_used"].values()) - 1) < 0.01


def test_underlying_technicals_maps_etfs_only():
    import pandas as pd
    idx = pd.bdate_range("2024-01-01", periods=300)
    px = pd.DataFrame({"XLE": np.linspace(50, 80, 300), "ERX": np.linspace(10, 40, 300), "MU": np.linspace(1, 2, 300)}, index=idx)
    u = analytics.underlying_technicals(px, ["ERX", "MU"], {"ERX": "leveraged_etf", "MU": "stock"})
    assert list(u) == ["ERX"] and u["ERX"]["underlying"] == "XLE"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
