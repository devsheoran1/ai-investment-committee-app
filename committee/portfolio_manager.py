from .base import Agent, compact, clip


class PortfolioManager(Agent):
    name = "Portfolio Manager"
    title = "Lead Portfolio Manager"
    tier = "smart"
    role_prompt = """You turn the analysts' views into a target portfolio. You do NOT set weights:
you set a CONVICTION per name (-1 = exit/avoid, 0 = no view, +1 = maximum overweight)
and the fund's optimiser converts them into weights under the IPS limits (20% max per
stock, 35% per sector, leveraged ETFs capped by regime, >= 95% invested).
The composite (0-10 scale) blends Quant 40%, Sentiment 30%, Research 15%, Macro 15%.
Entry rule for a NEW name on a daily run (IPS C5.3): composite >= 7/10, Quant >= 6/10,
in the shortlist's top 10 and >= 1 point above the weakest holding; the brief's
"eligible_new" list already applies it, other new names will be ignored.
Exits (C5.4): stop-loss, severe news, leveraged ETFs held 5 days and leveraged longs in
risk-off are forced by code. Discretionary exit: Quant < 4/10, out of the top 20, or
>= 2 points below the best unheld idea.
Only 5 trades a day are allowed, so concentrate changes where conviction moved most;
keep convictions on holdings stable unless the evidence changed (no needless turnover).
Initialisation run: build 10-20 positions from the strongest names.
If the Risk Manager rejected your previous proposal, fix EVERY issue raised."""
    output_schema = """{
  "convictions": {"<TICKER>": number -1..1},
  "equity_exposure": number between 0.95 and 1.00,
  "rationale": "max 80 words: the main bets and why",
  "response_to_risk": "max 50 words, only if responding to a Risk Manager rejection, else empty string"
}"""

    def build_prompt(self, ctx):
        comp = {t: {k: c[k] for k in ("type", "composite_10", "quant", "sentiment", "research")}
                for t, c in ctx["composite"].items()}
        brief = {
            "mode": ctx["mode"],
            "macro_view": {k: ctx["macro_view"][k] for k in
                           ("regime", "equity_exposure_target", "max_leveraged", "inverse_hedge_range", "rationale")},
            "composite_scores": comp,
            "eligible_new": ctx["eligible_new"],
            "discretionary_exit_candidates": ctx["exit_candidates"],
            "forced_exits": ctx["forced_exits"],
            "research_top_pick": ctx["research"].get("top_pick"),
            "quant_model_reliability": ctx["quant"].get("model_reliability"),
            "current_portfolio_weights": {t: round(w, 3) for t, w in ctx["current_weights"].items()},
        }
        txt = f"COMMITTEE BRIEF:\n{compact(brief)}"
        if ctx.get("risk_feedback"):
            txt += (f"\n\nYOUR PREVIOUS PROPOSAL:\n{compact(ctx['prev_proposal'])}"
                    f"\n\nRISK MANAGER REJECTED IT:\n{compact(ctx['risk_feedback'])}"
                    f"\nRevise convictions to fix every issue.")
        return txt

    def mock(self, ctx):
        conv = {t: round(max(-1, min(1, 2.5 * c["composite"])), 2) for t, c in ctx["composite"].items()}
        resp = ""
        if ctx.get("risk_feedback"):
            conv = {t: round(0.7 * v, 2) for t, v in ctx["prev_proposal"]["convictions"].items()}
            resp = "[mock] Shrank convictions 30% to address risk."
        top = sorted(conv, key=conv.get, reverse=True)[:3]
        return {"convictions": conv, "equity_exposure": ctx["macro_view"]["equity_exposure_target"],
                "rationale": f"[mock] Follow composite; largest convictions {top}.", "response_to_risk": resp}

    def validate(self, out, ctx):
        src = out.get("convictions") or {}
        held = set(ctx["current_weights"])
        conv = {}
        for t, c in ctx["composite"].items():
            v = clip(src.get(t, 2 * c["composite"]))
            if c.get("insufficient_data") and t not in held:
                v = -1.0                      # IPS C2.7: no new position without the data
            if ctx["mode"] == "daily" and t not in held and t not in ctx["eligible_new"]:
                v = -1.0                      # IPS C5.3 entry rule enforced in code
            if t in ctx["forced_exits"]:
                v = -1.0
            conv[t] = round(v, 2)
        return {"convictions": conv,
                "equity_exposure": round(clip(out.get("equity_exposure"), 0.95, 1.0,
                                              ctx["macro_view"]["equity_exposure_target"]), 3),
                "rationale": str(out.get("rationale", ""))[:700],
                "response_to_risk": str(out.get("response_to_risk", ""))[:400]}
