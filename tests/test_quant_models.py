"""Stage 4 (Oct 20): C3.4 trust tiers + t floor, C3.3 RSI rules, C3.7 zero-score escalation, LSTM leakage fix.
Run:  /opt/miniconda3/bin/python3 tests/test_quant_models.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import forecaster
from forecasters.horse_race import trust_weight
from committee.quant_analyst import QuantAnalyst


def test_c34_tiers_baseline_and_t_floor():
    assert trust_weight(0.51, 0.50, 500, 2.0) == 0.0          # under 52%
    assert trust_weight(0.53, 0.54, 500, 2.0) == 0.0          # doesn't beat always-up
    assert trust_weight(0.53, 0.50, 500, 2.0) == 0.25
    assert trust_weight(0.57, 0.50, 500, 2.0) == 0.50         # cap
    assert trust_weight(0.57, 0.50, 500, 0.8) == 0.0          # t < 1 floor
    assert trust_weight(0.57, 0.50, 60, 2.0) == 0.0           # < 100 predictions


def _ctx(rsi=50, above200=True, n=8):
    tick = [f"T{i}" for i in range(n)]
    tech = {t: {"mom_12_1_pct": 10.0 * i, "px_vs_50dma_pct": 2.0, "px_vs_200dma_pct": 5.0 if above200 else -5.0,
                "rsi_14": rsi, "macd_hist_pct": 0.1, "vol_3m_ann_pct": 25.0, "ret_5d_pct": 1.0} for i, t in enumerate(tick)}
    return {"tickers": tick, "technicals": tech,
            "forecast": {"forecasts": {t: {"pred_5d_return_pct": 0.1, "p_up": 0.55} for t in tick},
                         "oos_directional_hit_rate": 0.51},
            "race": {"weights": {"lstm": 0.0, "kronos": 0.0, "lgbm": 0.0}, "live": {}, "scoreboard_line": "x"}}


def test_rsi_rules_in_validate():
    q = QuantAnalyst()
    ctx = _ctx(rsi=80)
    out = q.validate({"signals": {t: {"score": 0.8} for t in ctx["tickers"]}}, ctx)["signals"]
    assert all(v["score"] == 0.4 and "overbought" in v["flags"] for v in out.values())     # RSI > 75 cuts the score
    ctx = _ctx(rsi=20, above200=False)
    out = q.validate({"signals": {t: {"score": 0.6} for t in ctx["tickers"]}}, ctx)["signals"]
    assert all(v["score"] <= 0 and "oversold_watch" in v["flags"] and "below_200dma" in v["flags"] for v in out.values())
    ctx = _ctx(rsi=20, above200=True)                                                       # above 200dma: allowed
    out = q.validate({"signals": {t: {"score": 0.6} for t in ctx["tickers"]}}, ctx)["signals"]
    assert all(v["score"] == 0.6 for v in out.values())
    assert set(next(iter(out.values()))["breakdown"]) >= {"momentum_12_1", "trend", "macd", "rsi", "models"}


class _LazyLLM:
    provider = "groq"
    def __init__(self, lazy_until): self.calls, self.lazy_until = [], lazy_until
    def chat_json(self, system, user, tier="fast", agent_name=None, mock_fn=None):
        self.calls.append(tier)
        lazy = len(self.calls) <= self.lazy_until
        import re, json
        ticks = re.findall(r'"(T\d+)":\[', user)
        return {"signals": {t: {"score": 0.0 if lazy else 0.3} for t in ticks}}, f"m{len(self.calls)}"


def test_c37_escalation_split_then_senior_then_rule_based():
    q = QuantAnalyst()
    out, model = q.run(_ctx(), _LazyLLM(lazy_until=1))          # fixed by splitting the batch
    assert out.get("quant_fallback", "").startswith("split") and "split" in model
    out, model = q.run(_ctx(), _LazyLLM(lazy_until=3))          # 20b, both halves lazy -> 120b fixes it
    assert "120b" in out["quant_fallback"]
    out, model = q.run(_ctx(), _LazyLLM(lazy_until=99))         # everything lazy -> rule-based, flagged
    assert "rule-based" in out["quant_fallback"] and all("[rule-based" in v["comment"] for v in out["signals"].values())


def test_lstm_scaling_uses_training_window_only():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (300, 2)); R[250:] *= 10               # a vol explosion after the training window
    dv, vol20, mkt = np.zeros_like(R), np.ones_like(R) * 0.01, rng.normal(0, 0.01, 300)
    *_, sd_train = forecaster._samples(R, dv, vol20, mkt, 60, 5, train_end=250)
    *_, sd_full = forecaster._samples(R, dv, vol20, mkt, 60, 5, train_end=300)
    assert np.all(sd_train < 0.02) and np.all(sd_full > sd_train * 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
