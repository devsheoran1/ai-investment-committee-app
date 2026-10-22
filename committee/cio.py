from .base import Agent, as_map, compact


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
On the INITIALISATION run (mode "init") the fund has no book yet and must deploy: you cannot decline a
risk-approved proposal. You MAY drop names or trim buys you dislike ("drop" / "trim"), and record concerns as dissent.
On any day, "drop" / "trim" only cut BUY orders; every change goes back to the Risk Manager. "lessons" are the committee's graded past calls: weigh them, don't over-fit. You see the actual order list (max 5 trades/day). Write the minutes a CIO would circulate."""
    output_schema = """{
  "decision": "approve" | "reject",
  "committee_summary": "max 100 words: what we decided and why",
  "key_risks": ["max 3 short items"],
  "dissent_noted": "max 40 words on where the analysts disagreed, or empty",
  "message_to_team": "max 30 words",
  "drop": ["TICKER", "... BUY orders to cut entirely (optional)"],
  "trim": {"TICKER": 0.5}
}
"trim" = fraction of the BUY order to KEEP (0-1). Omit drop/trim if you change nothing."""

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
            "lessons": ctx.get("lessons") or None,
            "devils_advocate": ctx.get("da_objections") or None,
            "pm_answers_to_devils_advocate": (ctx.get("proposal") or {}).get("da_responses") or None,
            "committee_track_record": ctx.get("track_record") if ctx.get("track_record", {}).get("calls") else None,
            "invalidated_holdings": [i["ticker"] for i in ctx.get("invalidated", [])] or None,
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
        llm_d = str(out.get("decision", "reject")).lower()
        llm_d = llm_d if llm_d in ("approve", "reject") else "reject"
        d, why = llm_d, ""
        if ctx.get("mode") == "init" and ctx["risk_decision"]["decision"] == "approve":
            d = "approve"   # Brief §1: the fund must start fully invested; the CIO can trim/drop and dissent, not decline
            why = "init day: the fund must deploy; the CIO may trim or drop names but cannot decline"
        if ctx["risk_decision"]["decision"] != "approve" or ctx.get("no_new_trades"):
            d = "reject"    # CIO can't overrule a Risk veto
            why = "Risk veto / insufficient information: the CIO cannot approve"
        summary = str(out.get("committee_summary", ""))[:900]
        dissent = str(out.get("dissent_noted", ""))[:400]
        if d != llm_d:      # minutes must match the decision: the LLM's text becomes recorded dissent
            dissent = (f"CIO recommended {llm_d.upper()} (overridden: {why}). {summary} {dissent}").strip()[:600]
            summary = f"Decision: {d.upper()} ({why})."
        else:
            summary = f"Decision: {d.upper()}. {summary}".strip()
        buys = {o["ticker"] for o in ctx.get("orders", []) if o.get("side") == "BUY" and not o.get("forced")}
        drop = [t for t in (out.get("drop") if isinstance(out.get("drop"), list) else []) if t in buys]
        trim = {}
        for t, f in as_map(out.get("trim")).items():
            try:
                f = float(f)
            except (TypeError, ValueError):
                continue
            if t in buys and t not in drop and 0 < f < 1:
                trim[t] = round(f, 2)
        key_risks = out.get("key_risks") if isinstance(out.get("key_risks"), list) else []
        return {"decision": d, "llm_decision": llm_d, "committee_summary": summary,
                "key_risks": [str(x)[:150] for x in key_risks][:3],
                "dissent_noted": dissent,
                "message_to_team": str(out.get("message_to_team", ""))[:300],
                "drop": drop if d == "approve" else [], "trim": trim if d == "approve" else {}}
