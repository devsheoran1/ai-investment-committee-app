"""
Devil's Advocate (IPS C0.1, added Oct 20). Runs after the PM, before Risk, on the PM's top 3 ideas: one
gpt-oss-120b call returning, per idea, the strongest bear case, the evidence it rests on, and a verdict
(proceed / downsize / drop). The PM must answer each objection; the CIO sees both. Advisory, no veto.
"""
from .base import Agent, compact

VERDICTS = ("proceed", "downsize", "drop")


class DevilsAdvocate(Agent):
    name = "Devil's Advocate"
    title = "Devil's Advocate (argues against the top 3 ideas)"
    tier = "smart"
    role_prompt = """Your job is to argue AGAINST the PM's three biggest ideas today, like a sceptical PM at the
morning meeting. For each idea give the single strongest bear case, the specific evidence from the data you were
given (a stretched RSI, a negative news item, an earnings date inside the holding window, a crowded sector,
a weak research flag, a model that disagrees, the Morning Brief), and a verdict: "proceed" (the bull case survives),
"downsize" (real risk, smaller position), or "drop" (the case is broken). Use ONLY the data given; if the evidence
against an idea is thin, say so and verdict "proceed". You have no veto: the PM must answer you and the CIO weighs both."""
    output_schema = """{
  "objections": {"<TICKER>": {"bear_case": "max 40 words", "evidence": "max 30 words citing given data",
                              "verdict": "proceed" | "downsize" | "drop"}}
}"""

    def build_prompt(self, ctx):
        ideas = {}
        for o in ctx["ideas"]:
            t = o["ticker"]
            c = ctx["composite"].get(t, {})
            ideas[t] = {"name": ctx.get("names", {}).get(t, t), "side": o["side"], "target_weight": o["target_weight"],
                        "composite_10": c.get("composite_10"), "quant": c.get("quant"), "sentiment": c.get("sentiment"),
                        "research": c.get("research"),
                        "technicals": {k: ctx["technicals"].get(t, {}).get(k) for k in ("rsi_14", "ret_1m_pct", "px_vs_200dma_pct", "vol_3m_ann_pct")},
                        "news": (ctx["sentiment"]["sentiment"].get(t) or {}).get("key_driver"),
                        "research_flags": ctx.get("research_flags", {}).get(t),
                        "earnings": (ctx.get("earnings") or {}).get(t, {}).get("next_date"),
                        "pm_thesis": ((ctx.get("pm") or {}).get("trade_notes") or {}).get(t, {}).get("thesis")}
        return f"PM'S TOP 3 IDEAS TODAY:\n{compact(ideas)}"

    def mock(self, ctx):
        out = {}
        for o in ctx["ideas"]:
            t = o["ticker"]
            rsi = ctx["technicals"].get(t, {}).get("rsi_14") or 50
            v = "downsize" if rsi > 70 else "proceed"
            out[t] = {"bear_case": f"[mock] momentum may be stretched (RSI {rsi})", "evidence": f"RSI-14 {rsi}", "verdict": v}
        return {"objections": out}

    def validate(self, out, ctx):
        src = out.get("objections") or {}
        res = {}
        for o in ctx["ideas"]:
            r = src.get(o["ticker"]) or {}
            v = str(r.get("verdict", "proceed")).lower()
            res[o["ticker"]] = {"bear_case": str(r.get("bear_case", "no objection raised"))[:300],
                                "evidence": str(r.get("evidence", ""))[:240],
                                "verdict": v if v in VERDICTS else "proceed"}
        return {"objections": res}
