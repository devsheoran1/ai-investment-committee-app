from .base import Agent, compact


class ReflectionAnalyst(Agent):
    """Post-trade reviewer. One batched call per run, only when past calls have come due.
    Not part of ROSTER: it grades the committee, it doesn't sit on it."""
    name = "Reflection Analyst"
    title = "Post-trade reviewer (committee memory)"
    tier = "fast"
    role_prompt = """You review the committee's own past calls after their 5-day horizon. For each call you get the
thesis, what would have invalidated it, the signals behind it, the regime, and the realised return vs SPY.
Write ONE specific lesson (max 35 words) per call: which signal misled us or held up, and in which regime
or sector it applies. No generic advice ("be careful"), no invented causes: use only the figures given.
A 'flat' verdict means the call was neither right nor wrong: say what we can't conclude. For SELL calls,
'right' means the stock then lagged SPY."""
    output_schema = """{
  "lessons": {"<call id>": {"lesson": "max 35 words, specific", "tags": ["max 3 short tags, e.g. momentum, energy, risk_on"]}}
}"""

    def build_prompt(self, ctx):
        rows = {}
        for g in ctx["reflect_due"]:
            rows[str(g["id"])] = {"ticker": g["ticker"], "sector": g.get("sector"), "action": g["action"],
                                  "regime": g.get("regime"), "thesis": (g.get("thesis") or "")[:140],
                                  "invalidation": (g.get("invalidation") or "")[:100],
                                  "signals": g.get("signals"), "ret_pct": round(g["ret"] * 100, 1),
                                  "spy_pct": round(g["spy_ret"] * 100, 1),
                                  "excess_pct": round(g["excess_ret"] * 100, 1), "verdict": g["verdict"]}
        return f"PAST CALLS TO REVIEW (horizon elapsed):\n{compact(rows)}"

    def mock(self, ctx):
        import reflection
        return {"lessons": {str(g["id"]): {"lesson": reflection.deterministic_lesson(g),
                                           "tags": reflection.default_tags(g)} for g in ctx["reflect_due"]}}

    def validate(self, out, ctx):
        src = out.get("lessons") or {}
        res = {}
        for g in ctx["reflect_due"]:
            r = src.get(str(g["id"])) or {}
            lesson = str(r.get("lesson", "")).strip()[:300]
            res[str(g["id"])] = {"lesson": lesson, "tags": [str(t)[:24] for t in (r.get("tags") or [])][:3]}
        return {"lessons": res}
