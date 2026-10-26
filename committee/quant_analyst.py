import numpy as np

import config
from strategies import regime as strat_regime

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
  "note": "max 40 words on how much you relied on the LSTM and why",
  "family_tilt": {"family": "momentum"|"earnings"|"reversal"|"defensive"|"", "pct": number -0.2..0.2, "reason": "max 25 words"},
  "hedge_proposal": {"pct": number 0..0.10, "reason": "max 25 words"}
}
(family_tilt and hedge_proposal are optional. family_tilt scales ONE strategy family's ensemble weight by 1+pct, needs a
written reason and is logged. hedge_proposal is only read when "soft_hedge" is open in the brief; the PM decides.)"""

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

    ETF_CAP = 0.95                                                       # an ETF score never sits at +-1.0 by construction

    @staticmethod
    def _stats(vals: dict):
        v = np.array([x for x in vals.values() if x is not None], float)
        return (float(v.mean()), float(v.std())) if len(v) >= 2 and v.std() > 0 else None

    def breakdown(self, ctx) -> dict[str, dict]:
        """Sub-score per signal, -1..+1, plus the model blend and the flags. 3m vol risk-adjusts the total."""
        T = ctx["technicals"]
        mom = self._z({t: T[t].get("mom_12_1_pct") for t in ctx["tickers"]})
        U = ctx.get("underlying_tech") or {}
        mom_st = self._stats({t: T[t].get("mom_12_1_pct") for t in ctx["tickers"]})
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
            etf = t in U
            sgn = -1.0 if config.LEVERAGE_FACTOR.get(t, 1.0) < 0 else 1.0        # inverse ETF: the index signal, flipped
            if etf:                                                              # leveraged ETF -> its underlying index's signals
                x = U[t]
                m_u = x.get("mom_12_1_pct")
                mom[t] = 0.0 if (m_u is None or not mom_st) else float(np.tanh((m_u - mom_st[0]) / mom_st[1] / 2))
            above50, above200 = (x.get("px_vs_50dma_pct") or 0) > 0, (x.get("px_vs_200dma_pct") or 0) > 0
            rsi = x.get("rsi_14") or 50
            b = {"momentum_12_1": round(mom[t], 2),
                 "trend": round((0.5 if above50 else -0.5) + (0.5 if above200 else -0.5), 2),
                 "macd": 0.5 if (x.get("macd_hist_pct") or 0) > 0 else -0.5,
                 "rsi": -1.0 if rsi > 75 else (0.3 if 50 <= rsi <= 70 else 0.0)}
            if etf and sgn < 0:
                b = {k: round(-v, 2) for k, v in b.items()}
            tech = sum(self.SUB_WEIGHTS[k] * v for k, v in b.items())
            wm = sum(w.get(m, 0) for m in model_z)
            models = sum(w[m] * model_z[m][t] for m in model_z) / wm if wm else 0.0
            raw = (1 - wm) * tech + wm * models
            e = (e_scores.get(t) or {}).get("score")
            if etf and e is not None:
                e = float(np.tanh(e))                                          # the ETF's own z-scores saturate at +-1: squash
            volpen = 1 + max(0.0, ((x.get("vol_3m_ann_pct") or 30) / 100) - 0.40) / 0.40
            legacy = float(np.clip(raw / volpen, -1, 1))                  # pre-blend score, kept for the live tracker
            if e is not None:                                              # A2/A3: multi-strategy ensemble (strategies/)
                raw = (1 - b_eff) * raw + b_eff * e
            if ens:
                raw += ens.get("tom_tilt") or 0.0                          # turn-of-month: tilt only, never a vote
            vol = (x.get("vol_3m_ann_pct") or 30) / 100
            score = raw / (1 + max(0.0, vol - 0.40) / 0.40)                # vol risk-adjusts, never a buy signal
            flags = [f for f, ok in (("overbought", rsi > 75), ("oversold_watch", rsi < 25),
                                     ("below_200dma", not above200)) if ok]
            lim = self.ETF_CAP if etf else 1.0
            out[t] = {"breakdown": {**b, "models": round(models, 2)}, "rule_score": round(float(np.clip(score, -lim, lim)), 2),
                      "flags": flags}
            if etf:
                out[t]["breakdown"]["scored_on"] = x["underlying"]
            out[t]["breakdown"]["legacy"] = round(legacy, 3)             # additive (live tracker): legacy-only score
            if e is not None:
                out[t]["breakdown"]["ensemble"] = round(e, 3)
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
        ens = ctx.get("ensemble") or {}
        if ens.get("weights"):
            fam = {}
            for k, w in ens["weights"].items():
                f = strat_regime.family_of(k) or "other"
                fam[f] = round(fam.get(f, 0.0) + w, 3)
            txt += (f"\n\nENSEMBLE REGIME (Python, SPY-based): {ens.get('regime_state')}; family weights after the regime allocator: "
                    f"{compact(fam)}. You MAY tilt ONE family by up to +-{config.LLM_TILT_MAX:.0%} via family_tilt, only with a written reason.")
        if ctx.get("soft_hedge_open"):
            txt += (f"\n\nSOFT HEDGE OPEN: risk-off signal(s) {ctx.get('soft_hedge_signals')}. You may propose 0-"
                    f"{config.SOFT_HEDGE['max']:.0%} {config.SOFT_HEDGE['ticker']} via hedge_proposal; the PM decides.")
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
        tilt = self._family_tilt(out, ctx)
        for t in ctx["tickers"]:
            r = entry(src, t)
            fallback = clip(r.get("score"), default=None) is None if r else True
            # missing / invalid LLM score -> the rules score already computed, flagged; never a silent 0.0
            score = round(bd[t]["rule_score"] if fallback else clip(r.get("score")), 2)
            if tilt and tilt.get("applied") and t in tilt["_delta"]:      # bounded family tilt: shift by the change in the ensemble score
                score = round(float(np.clip(score + tilt["_delta"][t], -1, 1)), 2)
            x = ctx["technicals"][t]
            if t in (ctx.get("underlying_tech") or {}):
                score = round(float(np.clip(score, -self.ETF_CAP, self.ETF_CAP)), 2)
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
        hp = as_map(out.get("hedge_proposal"))
        res = {"signals": sig, "model_reliability": rel if rel in ("low", "medium", "high") else "low",
               "model_weights": race.get("weights") or {"lstm": 0.0, "kronos": 0.0, "lgbm": 0.0},
               "note": str(out.get("note", ""))[:400]}
        if tilt:
            res["family_tilt"] = {k: v for k, v in tilt.items() if not k.startswith("_")}
        if ctx.get("soft_hedge_open") and hp:
            res["hedge_proposal"] = {"pct": round(clip(hp.get("pct"), 0.0, config.SOFT_HEDGE["max"], 0.0), 3),
                                     "reason": str(hp.get("reason", ""))[:200]}
        n_fb = sum(1 for v in sig.values() if v["fallback"])
        if out.get("quant_fallback"):
            res["quant_fallback"] = out["quant_fallback"]
        elif n_fb:
            res["quant_fallback"] = f"Quant: rule-based fallback for {n_fb}/{len(sig)} names"
        if race:
            res["model_scoreboard"] = race["scoreboard_line"]
        ens = ctx.get("ensemble")
        if ens:
            res["strategy_ensemble"] = {k: ens.get(k) for k in ("mode", "allocated", "weights", "weights_pre_regime", "regime_state", "regime_mult", "ic_table", "skipped", "tom_tilt", "asof")}
            if tilt:
                res["strategy_ensemble"]["tilted_weights"] = tilt.get("weights_after")
        return res

    @staticmethod
    def _family_tilt(out, ctx) -> dict | None:
        """Oct 24: the LLM may scale ONE family's ensemble weight by 1 +- 20% with a written reason. Bounded, logged,
        recomputed in Python from the stored per-strategy z; it never touches Risk limits."""
        ft = as_map(out.get("family_tilt"))
        fam = str(ft.get("family") or "").lower()
        ens = ctx.get("ensemble") or {}
        if not ft or fam not in config.STRATEGY_FAMILIES or not ens.get("weights"):
            return None
        pct = float(np.clip(clip(ft.get("pct"), -1.0, 1.0, 0.0), -config.LLM_TILT_MAX, config.LLM_TILT_MAX))
        reason = str(ft.get("reason") or "").strip()[:200]
        rec = {"family": fam, "pct": round(pct, 3), "reason": reason, "applied": False, "_delta": {}}
        if not reason or pct == 0.0:
            rec["note"] = "ignored: " + ("no written reason" if not reason else "zero tilt")
            return rec
        w0 = ens["weights"]
        w1 = strat_regime.tilt(w0, fam, pct, ens.get("has_history") or {k: True for k in w0})
        for t, e in (ens.get("scores") or {}).items():
            if e.get("score") is None:
                continue
            new = strat_regime.rescore(e.get("strategies") or {}, w1)
            if new is not None:
                rec["_delta"][t] = config.ENSEMBLE["blend"] * (new - e["score"])
        rec.update(applied=True, weights_before=w0, weights_after=w1)
        return rec
