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
        race = ctx.get("race")
        has_k = bool(race) and any((race["live"].get(t) or {}).get("kronos_ret") is not None for t in ctx["tickers"])
        tech_keys = list(next(iter(ctx["technicals"].values())).keys())
        cols = tech_keys + ["pred_5d_return_pct"] + (["kronos_ret", "kronos_p_up", "agree"] if has_k else [])
        rows = {}
        for t in ctx["tickers"]:
            vals = [ctx["technicals"][t][k] for k in tech_keys] + [fc["forecasts"][t]["pred_5d_return_pct"]]
            if has_k:
                lv = race["live"].get(t) or {}
                vals += [lv.get("kronos_ret"), lv.get("kronos_p_up"),
                         None if lv.get("agree") is None else ("yes" if lv["agree"] else "no")]
            rows[t] = vals
        txt = (f"LSTM MODEL CARD (out-of-sample):\n{compact(model_card)}\n\n"
               f"PER-STOCK SIGNALS: each ticker maps to a list in this column order {compact(cols)} "
               f"(pred_5d_return_pct = LSTM forecast; kronos_ret = Kronos-mini 5d forecast in %, kronos_p_up = share "
               f"of its sampled paths ending up; null = not available):\n{compact(rows)}")
        if race:
            txt += (f"\n\nMODEL SCOREBOARD (same out-of-sample origins; weight a model only as far as its IC "
                    f"justifies, a model with IC <= 0 gets no weight):\n{race['scoreboard_line']}")
        return txt

    def mock(self, ctx):
        tick = ctx["tickers"]
        fc = ctx["forecast"]
        mom = np.array([ctx["technicals"][t]["mom_12_1_pct"] or 0 for t in tick], float)
        pred = np.array([fc["forecasts"][t]["pred_5d_return_pct"] for t in tick], float)
        hr = fc["oos_directional_hit_rate"]
        trust = 0.0 if hr < 0.52 else 0.25 if hr <= 0.55 else 0.5   # IPS C3.4
        race = ctx.get("race")
        if race:   # blend weights from OOS rank IC, floored at 0 (no skill = no weight)
            w = race["weights"]
            trust = 0.0 if sum(w.values()) == 0 else trust
        zm = (mom - mom.mean()) / (mom.std() + 1e-9)
        zp = (pred - pred.mean()) / (pred.std() + 1e-9)
        if race and sum(race["weights"].values()) > 0:
            kr = np.array([(race["live"].get(t) or {}).get("kronos_ret") or 0.0 for t in tick], float)
            zk = (kr - kr.mean()) / (kr.std() + 1e-9)
            wl, wk = race["weights"].get("lstm", 0.0), race["weights"].get("kronos", 0.0)
            zp = wl * zp + wk * zk
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
        fallback = False
        if src and all(abs(v["score"]) < 1e-9 for v in sig.values()) and len(sig) > 5:
            # parseable reply but every score 0.0 and no comments: the small model went lazy (seen on Groq 20b).
            # Use the rule-based score instead of feeding the PM a blank, and say so.
            sig = {t: {"score": v["score"], "comment": "[rule-based fallback] " + v["comment"]}
                   for t, v in self.mock(ctx)["signals"].items()}
            fallback = True
        race = ctx.get("race")
        for t in sig:   # python-filled optional fields
            lv = ((race or {}).get("live") or {}).get(t) or {}
            if lv.get("kronos_ret") is not None:
                sig[t]["kronos_ret"], sig[t]["kronos_p_up"] = lv["kronos_ret"], lv["kronos_p_up"]
        res = {"signals": sig, "model_reliability": rel if rel in ("low", "medium", "high") else "low",
               "note": ("LLM returned all-zero scores: rule-based momentum/LSTM blend used instead. " if fallback else "")
               + str(out.get("note", ""))[:400]}
        if race:
            res["model_scoreboard"] = race["scoreboard_line"]
        return res
