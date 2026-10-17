from .base import Agent, compact


class ChiefInvestmentOfficer(Agent):
    name = "CIO"
    title = "Chief Investment Officer — chairs the Investment Committee"
    tier = "smart"
    role_prompt = """You chair the committee and decide which trades we finally do (IPS C7).
Think like an aggressive hedge-fund PM on a 2-week clock: momentum, earnings and news
that can move a stock fast matter; long-term growth doesn't. Take risk when the
opportunity is real, always inside the limits. Reject if the analysts disagree and
there is no clear reason to invest, if the risk is too large for the expected profit,
or if the trades are churn without a real opportunity. A rejection keeps the current
portfolio (forced risk exits such as stop-losses still execute).
Hard limits are already enforced by code: if the Risk verdict is "approve", no hard limit
is breached. Soft concerns (beta band, correlation, concentration) are judgment calls for
you, not limit breaches; never describe them as hard-limit breaches.
You see the actual order list (max 5 trades/day). Write the minutes a CIO would circulate."""
    output_schema = """{
  "decision": "approve" | "reject",
  "committee_summary": "max 100 words: what we decided and why",
  "key_risks": ["max 3 short items"],
  "dissent_noted": "max 40 words on where the analysts disagreed, or empty",
  "message_to_team": "max 30 words"
}"""

    def build_prompt(self, ctx):
        brief = {
            "macro": {k: ctx["macro_view"][k] for k in ("regime", "rationale")},
            "research_picks": {"top": ctx["research"].get("top_pick"), "least": ctx["research"].get("least_preferred")},
            "quant_note": ctx["quant"].get("note"),
            "sentiment_mood": ctx["sentiment"].get("market_mood"),
            "composite_scores": {t: c["composite"] for t, c in ctx["composite"].items()},
            "final_proposal": {"weights": {t: w for t, w in ctx["proposal"]["weights"].items() if w > 0},
                               "cash": ctx["risk_report"].get("cash_weight"),
                               "pm_rationale": ctx["proposal"]["rationale"]},
            "risk_verdict": {k: ctx["risk_decision"][k] for k in ("decision", "issues", "rationale")},
            "risk_metrics": {k: ctx["risk_report"].get(k) for k in
                             ("portfolio_vol_ann", "var_95_1d", "beta_vs_spy", "effective_n_positions")},
            "negotiation_rounds": ctx["round"],
            "turnover_pct_of_nav": ctx["turnover"],
            "orders": [{k: o[k] for k in ("ticker", "side", "shares", "notional", "forced_reason")}
                       for o in ctx.get("orders", [])],
            "mode": ctx.get("mode"),
        }
        return f"INVESTMENT COMMITTEE PACK:\n{compact(brief)}"

    def mock(self, ctx):
        ok = ctx["risk_decision"]["decision"] == "approve"
        return {"decision": "approve" if ok else "reject",
                "committee_summary": f"[mock] Committee {'approves' if ok else 'rejects'} the PM's portfolio "
                                     f"after {ctx['round']} risk round(s); turnover {ctx['turnover']:.0%} of NAV.",
                "key_risks": ["[mock] model risk", "[mock] macro regime shift"],
                "dissent_noted": "", "message_to_team": "[mock] Good work."}

    def validate(self, out, ctx):
        d = str(out.get("decision", "reject")).lower()
        if ctx["risk_decision"]["decision"] != "approve" or ctx.get("no_new_trades"):
            d = "reject"  # CIO can't overrule a Risk veto
        return {"decision": d if d in ("approve", "reject") else "reject",
                "committee_summary": str(out.get("committee_summary", ""))[:900],
                "key_risks": [str(x)[:150] for x in (out.get("key_risks") or [])][:3],
                "dissent_noted": str(out.get("dissent_noted", ""))[:400],
                "message_to_team": str(out.get("message_to_team", ""))[:300]}
