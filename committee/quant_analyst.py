import numpy as np

import config

from .base import Agent, as_map, compact, clip, entry


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

    # How to read the model scoreboard (same wording as lab_summary.py, the Model Lab and the daily report):
    #   hit = how often a model called the 5-day direction right on unseen data; always-up = the lazy "stocks usually go
    #   up" guess, which a model must beat; IC = does it rank future winners above losers (0 none, +0.05 slight, +0.10 good);
    #   t = IC divided by its noise, below 1 it could be luck. C3.4: a model gets weight only if it beats always-up, clears
    #   52% (25% weight; above 55% = 50%) and has t >= 1; otherwise Quant uses price signals only.
    # ---- Python-computed signals (C3.2): the LLM reads them, never computes them ---------------------- #
    SUB_WEIGHTS = {"momentum_12_1": 0.40, "trend": 0.30, "macd": 0.10, "rsi": 0.20}
    ZERO_SHARE = 0.30                                                    # C3.7: > 30% exact 0.0 = lazy reply

    @staticmethod
    def _z(vals: dict) -> dict:
        v = np.array([x for x in vals.values() if x is not None], float)
        if len(v) < 2 or v.std() == 0:
            return {k: 0.0 for k in vals}
        return {k: (0.0 if x is None else float(np.clip((x - v.mean()) / v.std() / 2, -1, 1))) for k, x in vals.items()}

    def breakdown(self, ctx) -> dict[str, dict]:
        """Sub-score per signal, -1..+1, plus the model blend and the flags. 3m vol risk-adjusts the total."""
        T = ctx["technicals"]
        mom = self._z({t: T[t].get("mom_12_1_pct") for t in ctx["tickers"]})
        race = ctx.get("race") or {}
        w = race.get("weights") or {}
        live = race.get("live") or {}
        model_z = {}
        for m, key in (("lstm", "lstm"), ("kronos", "kronos_ret"), ("lgbm", "lgbm")):
            if w.get(m):
                model_z[m] = self._z({t: (live.get(t) or {}).get(key) for t in ctx["tickers"]})
        ens = ctx.get("ensemble") or {}
        e_scores = ens.get("scores") or {}
        b_eff = config.ENSEMBLE["blend"] * float(ens.get("allocated") or 0.0)     # thin IC evidence = legacy score keeps its say
        out = {}
        for t in ctx["tickers"]:
            x = T[t]
            above50, above200 = (x.get("px_vs_50dma_pct") or 0) > 0, (x.get("px_vs_200dma_pct") or 0) > 0
            rsi = x.get("rsi_14") or 50
            b = {"momentum_12_1": round(mom[t], 2),
                 "trend": round((0.5 if above50 else -0.5) + (0.5 if above200 else -0.5), 2),
                 "macd": 0.5 if (x.get("macd_hist_pct") or 0) > 0 else -0.5,
                 "rsi": -1.0 if rsi > 75 else (0.3 if 50 <= rsi <= 70 else 0.0)}
            tech = sum(self.SUB_WEIGHTS[k] * v for k, v in b.items())
            wm = sum(w.get(m, 0) for m in model_z)
            models = sum(w[m] * model_z[m][t] for m in model_z) / wm if wm else 0.0
            raw = (1 - wm) * tech + wm * models
            e = (e_scores.get(t) or {}).get("score")
            if e is not None:                                              # A2/A3: multi-strategy ensemble (strategies/)
                raw = (1 - b_eff) * raw + b_eff * e
            if ens:
                raw += ens.get("tom_tilt") or 0.0                          # turn-of-month: tilt only, never a vote
            vol = (x.get("vol_3m_ann_pct") or 30) / 100
            score = raw / (1 + max(0.0, vol - 0.40) / 0.40)                # vol risk-adjusts, never a buy signal
            flags = [f for f, ok in (("overbought", rsi > 75), ("oversold_watch", rsi < 25),
                                     ("below_200dma", not above200)) if ok]
            out[t] = {"breakdown": {**b, "models": round(models, 2)}, "rule_score": round(float(np.clip(score, -1, 1)), 2),
                      "flags": flags}
            if e is not None:
                out[t]["breakdown"]["ensemble"] = round(e, 2)
        return out

    def build_prompt(self, ctx):
        fc = ctx["forecast"]
        model_card = {k: fc.get(k) for k in ("model", "walk_forward_days", "oos_directional_hit_rate", "always_up_hit_rate",
                                             "oos_information_coefficient")}
        race = ctx.get("race")
        has_k = bool(race) and any((race["live"].get(t) or {}).get("kronos_ret") is not None for t in ctx["tickers"])
        tech_keys = list(next(iter(ctx["technicals"].values())).keys())
        pead = ctx.get("pead") or {}
        bd = self.breakdown(ctx)
        cols = tech_keys + ["pred_5d_return_pct", "lstm_p_up"] + (["kronos_ret", "kronos_p_up"] if has_k else []) + \
            (["lgbm_pred", "agree"] if race else []) + (["pead_drift"] if pead else []) + ["rule_score", "flags"]
        rows = {}
        for t in ctx["tickers"]:
            f = fc["forecasts"].get(t) or {}
            vals = [ctx["technicals"][t][k] for k in tech_keys] + [f.get("pred_5d_return_pct"), f.get("p_up")]
            lv = (race or {}).get("live", {}).get(t) or {}
            if has_k:
                vals += [lv.get("kronos_ret"), lv.get("kronos_p_up")]
            if race:
                vals += [lv.get("lgbm"), None if lv.get("agree") is None else ("yes" if lv["agree"] else "no")]
            if pead:
                vals.append((pead.get(t) or {}).get("drift_score"))
            vals += [bd[t]["rule_score"], bd[t]["flags"] or None]
            rows[t] = vals
        txt = (f"LSTM MODEL CARD (walk-forward, out-of-sample):\n{compact(model_card)}\n\n"
               f"PER-STOCK SIGNALS: each ticker maps to a list in this column order {compact(cols)} "
               f"(pred_5d_return_pct / lstm_p_up = LSTM forecast and P(up); kronos_ret = Kronos-mini 5d forecast in %; "
               f"lgbm_pred = LightGBM 5d forecast in %; pead_drift = post-earnings drift -1..+1; rule_score = Python's "
               f"rule-based score from the C3.2 signals and the C3.4 model weights; null = not available):\n{compact(rows)}")
        if race:
            txt += (f"\n\nMODEL WEIGHTS THIS CYCLE (C3.4, computed in Python): {compact(race.get('weights'))}\n"
                    f"MODEL SCOREBOARD: {race['scoreboard_line']}")
        return txt

    def mock(self, ctx):
        bd = self.breakdown(ctx)
        fc = ctx["forecast"]
        return {"signals": {t: {"score": bd[t]["rule_score"],
                                "comment": f"[mock] 12-1 mom {ctx['technicals'][t].get('mom_12_1_pct')}%, "
                                           f"LSTM {(fc['forecasts'].get(t) or {}).get('pred_5d_return_pct')}%"}
                            for t in ctx["tickers"]},
                "model_reliability": "low" if not sum(((ctx.get("race") or {}).get("weights") or {}).values()) else "medium",
                "note": "[mock] rule-based score from the C3.2 signals and C3.4 model weights"}

    def _lazy(self, out) -> bool:
        sc = [v["score"] for v in out["signals"].values()]
        return len(sc) > 5 and sum(1 for x in sc if x == 0.0) / len(sc) > self.ZERO_SHARE

    def run(self, ctx, llm):
        """C3.7 zero-score guard: > 30% exact 0.0 -> split the batch in two and retry -> gpt-oss-120b -> rule-based."""
        out, model = super().run(ctx, llm)
        if llm.provider == "mock" or not self._lazy(out):
            return out, model
        tick = ctx["tickers"]
        merged, models = {}, []
        for half in (tick[:len(tick) // 2], tick[len(tick) // 2:]):
            o, m = super().run({**ctx, "tickers": half}, llm)
            merged.update(o["signals"]); models.append(m)
        out2 = {**out, "signals": {t: merged.get(t, out["signals"][t]) for t in tick}}
        if not self._lazy(out2):
            out2["quant_fallback"] = "split batch (C3.7)"
            return out2, f"{model} -> split ({'/'.join(models)})"
        raw, m3 = llm.chat_json(self.system_prompt, self.build_prompt(ctx), tier="smart", agent_name=self.name,
                                mock_fn=lambda: self.mock(ctx))
        out3 = self.validate(raw if isinstance(raw, dict) else {}, ctx)
        if not self._lazy(out3):
            out3["quant_fallback"] = "retried on gpt-oss-120b (C3.7)"
            return out3, f"{model} -> {m3}"
        rb = self.validate(self.mock(ctx), ctx)
        rb["quant_fallback"] = "rule-based score (C3.7: 20b, split batch and 120b all lazy)"
        for v in rb["signals"].values():
            v["comment"] = "[rule-based fallback, C3.7] " + v["comment"]
        return rb, f"{model} -> rule-based fallback"

    def validate(self, out, ctx):
        src = as_map(out.get("signals"))        # tickers outside the shortlist are simply never read
        bd = self.breakdown(ctx)
        race = ctx.get("race") or {}
        sig = {}
        for t in ctx["tickers"]:
            r = entry(src, t)
            fallback = clip(r.get("score"), default=None) is None if r else True
            # missing / invalid LLM score -> the rules score already computed, flagged; never a silent 0.0
            score = round(bd[t]["rule_score"] if fallback else clip(r.get("score")), 2)
            x = ctx["technicals"][t]
            rsi, above200 = x.get("rsi_14") or 50, (x.get("px_vs_200dma_pct") or 0) > 0
            if rsi > 75 and score > 0:                                   # C3.3: RSI > 75 cuts the score
                score = round(score * 0.5, 2)
            if rsi < 25 and not above200:                                # C3.3: no falling knives
                score = min(score, 0.0)
            sig[t] = {"score": score, "comment": "Quant: rule-based fallback" if fallback else str(r.get("comment", ""))[:160],
                      "fallback": fallback,
                      "breakdown": bd[t]["breakdown"], "flags": bd[t]["flags"], "rule_score": bd[t]["rule_score"]}
            es = ((ctx.get("ensemble") or {}).get("scores") or {}).get(t)
            if es:
                sig[t]["votes"] = {"long": es["votes_long"], "short": es["votes_short"], "of": es["n_strategies"],
                                   "needed": es["min_votes"]}
                sig[t]["strategies"] = {n: {"z": v["z"], "w": v["w"], "why": v["reason"]} for n, v in es["strategies"].items()}
            lv = (race.get("live") or {}).get(t) or {}
            if lv.get("kronos_ret") is not None:
                sig[t]["kronos_ret"], sig[t]["kronos_p_up"] = lv["kronos_ret"], lv["kronos_p_up"]
        rel = str(out.get("model_reliability", "low")).lower()
        res = {"signals": sig, "model_reliability": rel if rel in ("low", "medium", "high") else "low",
               "model_weights": race.get("weights") or {"lstm": 0.0, "kronos": 0.0, "lgbm": 0.0},
               "note": str(out.get("note", ""))[:400]}
        n_fb = sum(1 for v in sig.values() if v["fallback"])
        if out.get("quant_fallback"):
            res["quant_fallback"] = out["quant_fallback"]
        elif n_fb:
            res["quant_fallback"] = f"Quant: rule-based fallback for {n_fb}/{len(sig)} names"
        if race:
            res["model_scoreboard"] = race["scoreboard_line"]
        ens = ctx.get("ensemble")
        if ens:
            res["strategy_ensemble"] = {k: ens.get(k) for k in ("mode", "allocated", "weights", "ic_table", "skipped", "tom_tilt", "asof")}
        return res
