import numpy as np

from .base import Agent, compact, clip


class QuantAnalyst(Agent):
    name = "Quant Analyst"
    title = "Quantitative Analyst (technicals + deep learning)"
    tier = "fast"
    role_prompt = """You score every shortlisted stock AND leveraged ETF on price behaviour plus
the fund's LSTM forecast. Strategy bias is momentum (IPS C3.3): buy strength
(12-1m momentum, 3m and 5-day returns, price above 50/200dma); RSI > 75 is an
overbought warning, not a sell. LSTM trust rule (IPS C3.4), from its out-of-sample
hit rate: < 52% -> ignore it; 52-55% -> 25% weight; > 55% -> 50% weight.
Leveraged ETFs: judge trend strength and volatility; they decay in choppy markets.
Score -1..+1 (0 = no edge). Use the range: the PM needs to see who is strongest."""
    output_schema = """{
  "signals": {"<TICKER>": {"score": number -1..1, "comment": "max 20 words"}},
  "model_reliability": "low" | "medium" | "high",
  "note": "max 40 words on how much you relied on the LSTM and why"
}"""

    def build_prompt(self, ctx):
        fc = ctx["forecast"]
        model_card = {k: fc[k] for k in ("model", "train_samples", "test_samples",
                                         "oos_directional_hit_rate", "oos_information_coefficient")}
        rows = {t: {**ctx["technicals"][t], **fc["forecasts"][t]} for t in ctx["tickers"]}
        return (f"LSTM MODEL CARD (out-of-sample):\n{compact(model_card)}\n\n"
                f"PER-STOCK SIGNALS (pred_5d_return_pct is the LSTM forecast):\n{compact(rows)}")

    def mock(self, ctx):
        tick = ctx["tickers"]
        fc = ctx["forecast"]
        mom = np.array([ctx["technicals"][t]["mom_12_1_pct"] or 0 for t in tick], float)
        pred = np.array([fc["forecasts"][t]["pred_5d_return_pct"] for t in tick], float)
        hr = fc["oos_directional_hit_rate"]
        trust = 0.0 if hr < 0.52 else 0.25 if hr <= 0.55 else 0.5   # IPS C3.4
        zm = (mom - mom.mean()) / (mom.std() + 1e-9)
        zp = (pred - pred.mean()) / (pred.std() + 1e-9)
        sc = np.clip((1 - trust) * zm + trust * zp, -2, 2) / 2
        rel = "high" if trust >= 0.5 else "medium" if trust > 0 else "low"
        return {"signals": {t: {"score": round(float(s), 2),
                                "comment": f"[mock] 12-1 mom {m:+.0f}%, LSTM {p:+.2f}%"}
                            for t, s, m, p in zip(tick, sc, mom, pred)},
                "model_reliability": rel,
                "note": f"[mock] LSTM hit rate {fc['oos_directional_hit_rate']:.0%}; weight {trust:.0%} on model."}

    def validate(self, out, ctx):
        src = out.get("signals") or {}
        sig = {t: {"score": round(clip((src.get(t) or {}).get("score")), 2),
                   "comment": str((src.get(t) or {}).get("comment", ""))[:160]} for t in ctx["tickers"]}
        rel = str(out.get("model_reliability", "low")).lower()
        return {"signals": sig, "model_reliability": rel if rel in ("low", "medium", "high") else "low",
                "note": str(out.get("note", ""))[:400]}
