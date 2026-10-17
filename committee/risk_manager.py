from .base import Agent, compact


class RiskManager(Agent):
    name = "Risk Manager"
    title = "Chief Risk Officer"
    tier = "smart"
    role_prompt = """You are independent from the investment team and have VETO power.
You review the proposed portfolio's risk report: volatility, 1-day 95% VaR and
CVaR, beta to the S&P 500, backtested drawdown, concentration (effective number
of positions) and sector exposure, against the fund's hard limits.
Any hard-limit breach listed in "breaches" = automatic rejection.
"soft_concerns" (IPS C6.3: effective positions < 8, beta outside the regime band,
pairwise correlation > 0.8, fund drawdown past -10%) MAY justify a rejection, but
the fund is deliberately aggressive over a 2-week window: be pragmatic and don't
reject a reasonable portfolio over trivia. Leveraged ETFs: max 5 trading days held. When you reject, state
precisely what must change so the PM can fix it."""
    output_schema = """{
  "decision": "approve" | "reject",
  "issues": ["short, specific issue", ...],
  "required_changes": "max 50 words, concrete instructions for the PM (empty if approved)",
  "rationale": "max 60 words"
}"""

    def build_prompt(self, ctx):
        prop = ctx["proposal"]
        return (f"REVIEW ROUND {ctx['round']}.\n"
                f"POST-TRADE WEIGHTS (current book + today's <=5 orders):\n{compact({t: w for t, w in prop['weights'].items() if w > 0})}\n"
                f"PM RATIONALE: {prop['rationale']}\n\n"
                f"RISK REPORT (computed by risk system):\n{compact(ctx['risk_report'])}")

    def mock(self, ctx):
        rr = ctx["risk_report"]
        if rr["breaches"]:
            return {"decision": "reject", "issues": rr["breaches"],
                    "required_changes": "[mock] Reduce equity exposure / concentration until limits are met.",
                    "rationale": "[mock] Hard limit breach."}
        return {"decision": "approve", "issues": [], "required_changes": "",
                "rationale": f"[mock] Vol {rr['portfolio_vol_ann']:.1%}, VaR {rr['var_95_1d']:.2%}, "
                             f"beta {rr['beta_vs_spy']} all within limits."}

    def validate(self, out, ctx):
        breaches = ctx["risk_report"].get("breaches", [])
        decision = str(out.get("decision", "reject")).lower()
        issues = [str(i)[:200] for i in (out.get("issues") or [])][:8]
        if breaches:  # hard limits are enforced by code, not by the model's opinion
            decision = "reject"
            issues = list(dict.fromkeys(breaches + issues))
        elif decision == "reject" and ctx.get("round", 1) >= 2:
            # soft concerns get one rejection (round 1); after that they are escalated to the CIO
            decision = "approve"
            issues = [f"(soft, escalated to CIO) {i}" for i in issues]
        return {"decision": decision if decision in ("approve", "reject") else "reject",
                "soft_concerns": ctx["risk_report"].get("soft_concerns", []),
                "issues": issues,
                "required_changes": str(out.get("required_changes", ""))[:400],
                "rationale": str(out.get("rationale", ""))[:500],
                "hard_breaches": breaches}
