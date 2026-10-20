"""
Risk Manager (IPS C6). Prof. Manero's rule 3 (Oct 20): leverage and inverse hedges are allowed only because
THIS agent enforces the limits. So the hard-limit verdict lives here (check_book / classify), is applied to the
post-trade book, and is what the transcript and report show as the Risk Manager's verdict.

    hard (always reject, Python)  holdings < 10, cash > 3%, name > 20%, sector > 35%, leveraged > 30% total or 15% each,
                                  inverse > 25% or outside risk-off, vol > 25%, VaR > 5%, the C11 drawdown rule
    soft (may reject, IPS C6.3)   effective positions < 4, corr > 0.9 among the top 3, position > 25% of 20d ADV
    warnings (never reject)       everything else: beta band, concentration, correlation, vol-target caps ...
"""
import analytics
import config

from .base import Agent, compact

EPS = 1e-4


def _row(rule, value, limit, ok, kind, fmt="{:.1%}"):
    f = (lambda v: fmt.format(v)) if isinstance(value, (int, float)) else str
    return {"rule": rule, "value": f(value), "limit": limit, "status": "pass" if ok else "fail", "kind": kind}


def classify(rr: dict) -> dict:
    """Turn a risk report into the Risk Manager's verdict lists + a pass/fail table (IPS E3)."""
    L, R = config.RISK_LIMITS, config.REGIME_RULES[rr["regime"]]
    S = config.SOFT_REJECT
    rows, hard, soft, warn = [], [], [], []

    def add(rule, value, limit, ok, kind, msg, fmt="{:.1%}"):
        rows.append(_row(rule, value, limit, ok, kind, fmt))
        if not ok:
            (hard if kind == "hard" else soft if kind == "soft" else warn).append(msg)

    n, cash = rr.get("n_positions", 0), rr.get("cash_weight", 1.0)
    add("Holdings (A7)", n, f"{config.MIN_HOLDINGS}-{config.MAX_HOLDINGS}", config.MIN_HOLDINGS <= n <= config.MAX_HOLDINGS,
        "hard", f"{n} holdings outside {config.MIN_HOLDINGS}-{config.MAX_HOLDINGS} (IPS A7 floor 10)", "{}")
    add("Cash max (A12)", cash, f"<= {L['max_cash_weight']:.0%}", cash <= L["max_cash_weight"] + EPS, "hard",
        f"cash {cash:.1%} > {L['max_cash_weight']:.0%} (IPS A12)")
    add("Cash >= 0 (Brief)", cash, ">= 0", cash >= -EPS, "hard", f"cash {cash:.2%} < 0")
    wts = rr.get("weights", {})
    stocks = {t: x for t, x in wts.items() if t not in rr.get("leveraged_by_etf", {}) and t not in rr.get("inverse_by_etf", {})}
    top_name = max(stocks.items(), key=lambda kv: kv[1]) if stocks else ("-", 0.0)
    add("Max single name (A8)", top_name[1], f"<= {L['max_single_weight']:.0%}", top_name[1] <= L["max_single_weight"] + EPS,
        "hard", f"{top_name[0]} {top_name[1]:.1%} > {L['max_single_weight']:.0%} (IPS A8)")
    secs = {k: v for k, v in rr.get("sector_weights", {}).items() if k not in ("Leveraged ETF", "Inverse ETF")}
    top_sec = max(secs.items(), key=lambda kv: kv[1]) if secs else ("-", 0.0)
    add("Max sector (A9)", top_sec[1], f"<= {L['max_sector_weight']:.0%}", top_sec[1] <= L["max_sector_weight"] + EPS,
        "hard", f"{top_sec[0]} sector {top_sec[1]:.1%} > {L['max_sector_weight']:.0%} (IPS A9)")
    lev_total, lev_each = rr.get("leveraged_etf_weight", 0.0), rr.get("leveraged_by_etf", {})
    lev_cap = min(L["max_leveraged_total"], R["max_leveraged"])
    add("Leveraged total (A10/C1.4)", lev_total, f"<= {lev_cap:.0%} in {rr['regime']}", lev_total <= lev_cap + EPS, "hard",
        f"leveraged ETFs {lev_total:.1%} > {lev_cap:.0%} allowed in {rr['regime']} (IPS A10/C1.4)")
    big_lev = max(lev_each.items(), key=lambda kv: kv[1]) if lev_each else ("-", 0.0)
    add("Leveraged per ETF (A10)", big_lev[1], f"<= {L['max_leveraged_each']:.0%}", big_lev[1] <= L["max_leveraged_each"] + EPS,
        "hard", f"{big_lev[0]} {big_lev[1]:.1%} > {L['max_leveraged_each']:.0%} per leveraged ETF (IPS A10)")
    inv_total = rr.get("inverse_etf_weight", 0.0)
    add("Inverse total (A11)", inv_total, f"<= {L['max_inverse_total']:.0%}", inv_total <= L["max_inverse_total"] + EPS,
        "hard", f"inverse ETFs {inv_total:.1%} > {L['max_inverse_total']:.0%} (IPS A11)")
    add("Inverse only in risk-off (A11)", "yes" if inv_total > 0 else "none", "risk-off only",
        inv_total <= EPS or rr["regime"] == "risk_off", "hard", "inverse ETFs held outside risk-off (IPS A11)")
    vol, var = rr.get("portfolio_vol_ann", 0.0), rr.get("var_95_1d", 0.0)
    add("Portfolio vol (A3)", vol, f"<= {L['max_portfolio_vol']:.0%}", vol <= L["max_portfolio_vol"] + EPS, "hard",
        f"volatility {vol:.1%} > {L['max_portfolio_vol']:.0%} (IPS A3)")
    add("1d VaR 95% (A5)", var, f"<= {L['max_var_95_1d']:.0%}", var <= L["max_var_95_1d"] + EPS, "hard",
        f"1d VaR95 {var:.2%} > {L['max_var_95_1d']:.0%} (IPS A5)", "{:.2%}")
    # C11 graduated drawdown rule
    dd = rr.get("dd_state") or {}
    lvl = dd.get("level", "ok")
    c11_ok, c11_msg = True, ""
    if lvl in ("warn", "hard"):
        if lev_total > (dd.get("lev_cap") or 0.0) + EPS:
            c11_ok, c11_msg = False, f"C11 {lvl}: leveraged sleeve {lev_total:.1%} > {dd.get('lev_cap', 0):.0%}"
        cur = rr.get("current_weights", {})
        added = [t for t, x in lev_each.items() if x > cur.get(t, 0.0) + 0.005]
        if added:
            c11_ok, c11_msg = False, f"C11 {lvl}: no new leveraged buys allowed ({', '.join(added)})"
        if lvl == "hard" and rr["regime"] == "risk_off" and inv_total + EPS < (dd.get("inverse_min_risk_off") or 0):
            c11_ok, c11_msg = False, f"C11 hard: inverse hedge {inv_total:.1%} < {dd['inverse_min_risk_off']:.0%} in risk-off"
    add("Drawdown rule (A4/C11)", dd.get("drawdown", 0.0), f"level {lvl}", c11_ok, "hard", c11_msg)
    # soft rejects (IPS C6.3)
    eff = rr.get("effective_n_positions", 0.0)
    add("Effective positions (C6.3)", eff, f">= {S['min_effective_positions']}", eff >= S["min_effective_positions"],
        "soft", f"effective positions {eff:.1f} < {S['min_effective_positions']}", "{:.1f}")
    t3 = rr.get("top3_max_pair_corr", 0.0)
    add("Top-3 pairwise corr (C6.3)", t3, f"<= {S['max_top3_pair_corr']}", t3 <= S["max_top3_pair_corr"], "soft",
        f"correlation {t3:.2f} > {S['max_top3_pair_corr']} among the top 3 holdings {rr.get('top3_holdings')}", "{:.2f}")
    advf = rr.get("adv_fraction", {})
    worst = max(advf.items(), key=lambda kv: kv[1]) if advf else ("-", 0.0)
    add("Liquidity, % of 20d ADV (C6.3)", worst[1], f"<= {S['max_adv_fraction']:.0%}", worst[1] <= S["max_adv_fraction"],
        "soft", f"{worst[0]} position = {worst[1]:.0%} of 20d ADV (> {S['max_adv_fraction']:.0%})")
    # warnings
    beta, (lo, hi) = rr.get("beta_vs_spy", 0.0), R["beta"]
    add("Beta band (C1.4, target)", beta, f"{lo}-{hi}", lo - 0.005 <= beta <= hi + 0.005, "warn",
        f"beta {beta:.2f} outside the {rr['regime']} band {lo}-{hi} (optimiser target, warning only)", "{:.2f}")
    if rr.get("max_pairwise_corr", 0) > 0.8:
        warn.append(f"max pairwise correlation {rr['max_pairwise_corr']:.2f} > 0.8 (warning)")
    for t, x in lev_each.items():
        cap = rr.get("etf_effective_caps", {}).get(t)
        if cap is not None and x > cap + EPS:
            warn.append(f"{t} {x:.1%} above its vol-targeted cap {cap:.1%} (within IPS)")
    return {**rr, "limit_checks": rows, "breaches": hard, "soft_rejects": soft, "warnings": warn,
            "soft_concerns": soft + warn}


def check_book(weights, prices, sectors, asset_types, regime, **kw) -> dict:
    """The Risk Manager's check of a (post-trade) book: metrics from analytics, verdict from classify()."""
    return classify(analytics.risk_report(weights, prices, sectors, asset_types, regime, **kw))


def fix_hints(rr: dict) -> list[str]:
    hints = []
    for b in rr.get("breaches", []):
        if "holdings" in b:
            hints.append("add names (or pair every exit with a buy) to keep >= 10 holdings")
        elif "cash" in b and ">" in b:
            hints.append("upsize today's buys so cash <= 3%")
        elif "sector" in b:
            hints.append("trim the named sector below 35% and redistribute")
        elif "leveraged" in b or "C11" in b:
            hints.append("cut the leveraged sleeve to its cap / no new leveraged buys")
        elif "volatility" in b or "VaR" in b:
            hints.append("reduce leverage and the highest-vol names")
        elif "inverse" in b:
            hints.append("inverse ETFs only in risk-off, max 25%")
        else:
            hints.append(f"fix: {b}")
    return list(dict.fromkeys(hints))


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

    check_book = staticmethod(check_book)

    def build_prompt(self, ctx):
        prop, rr = ctx["proposal"], ctx["risk_report"]
        metrics = {k: rr.get(k) for k in ("portfolio_vol_ann", "var_95_1d", "cvar_95_1d", "beta_vs_spy", "n_positions",
                                          "effective_n_positions", "cash_weight", "leveraged_etf_weight",
                                          "inverse_etf_weight", "sector_weights", "top3_max_pair_corr",
                                          "fund_drawdown_from_peak", "etf_effective_caps", "regime")}
        return (f"REVIEW ROUND {ctx['round']} of {config.MAX_RISK_ROUNDS}.\n"
                f"POST-TRADE WEIGHTS (current book + today's <=5 orders):\n{compact({t: w for t, w in prop['weights'].items() if w > 0})}\n"
                f"PM RATIONALE: {prop['rationale']}\n\n"
                f"RISK METRICS:\n{compact(metrics)}\n"
                f"HARD BREACHES (Python, automatic reject): {compact(rr.get('breaches', []))}\n"
                f"SOFT CONCERNS THAT MAY JUSTIFY A REJECTION (IPS C6.3): {compact(rr.get('soft_rejects', []))}\n"
                f"WARNINGS ONLY (never a reason to reject): {compact(rr.get('warnings', []))}")

    def mock(self, ctx):
        rr = ctx["risk_report"]
        if rr["breaches"] or rr.get("soft_rejects"):
            return {"decision": "reject", "issues": rr["breaches"] + rr.get("soft_rejects", []),
                    "required_changes": "[mock] " + "; ".join(fix_hints(rr) or ["address the soft concerns"]),
                    "rationale": "[mock] Limit breach."}
        return {"decision": "approve", "issues": [], "required_changes": "",
                "rationale": f"[mock] Vol {rr['portfolio_vol_ann']:.1%}, VaR {rr['var_95_1d']:.2%}, "
                             f"beta {rr['beta_vs_spy']} all within limits."}

    def validate(self, out, ctx):
        rr = ctx["risk_report"]
        hard, soft, warn = rr.get("breaches", []), rr.get("soft_rejects", []), rr.get("warnings", [])
        llm_decision = str(out.get("decision", "reject")).lower()
        decision, override = llm_decision if llm_decision in ("approve", "reject") else "reject", None
        if hard and decision == "approve":
            decision, override = "reject", "LLM approved a book with hard breaches: overridden to reject (IPS C6.1)"
        elif hard:
            decision = "reject"
        elif decision == "reject" and not soft:
            decision, override = "approve", "LLM rejected on warnings only (not allowed by IPS C6.3): overridden to approve"
        llm_changes = str(out.get("required_changes", "")).strip()
        changes = fix_hints(rr) + ([llm_changes[:300]] if llm_changes else [])
        issues = list(dict.fromkeys(hard + soft + [str(i)[:200] for i in (out.get("issues") or [])][:8]))
        return {"decision": decision, "llm_decision": llm_decision, "override": override,
                "hard_breaches": hard, "soft_rejects": soft, "warnings": warn,
                "required_changes": changes if decision == "reject" else [],
                "issues": issues, "soft_concerns": soft + warn,
                "rationale": str(out.get("rationale", ""))[:500],
                "limit_checks": rr.get("limit_checks", [])}
