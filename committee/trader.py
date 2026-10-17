"""
Execution Agent (Brief §8.2) — deterministic rules, no LLM (IPS C8.1).
Every judgment was made upstream; the Trader only validates, sequences and logs.

Check order matters: per-order validity first (tradeable price, minimum size,
positive shares), THEN the daily cap, so an invalid order is rejected for its
own reason instead of being hidden behind "cap full".
"""
import math

import config

from .base import Agent


class Trader(Agent):
    name = "Trader"
    title = "Head Trader & Compliance (rule-based)"
    tier = "rules"
    role_prompt = ("Deterministic code, not an LLM. Validates each order, enforces the 5-trade daily cap, "
                   "keeps cash >= 0, sequences sells before buys (largest first) and logs every rejection.")
    output_schema = """{"execute": bool, "approved_orders": [...], "rejected": [{"ticker","reason"}], "execution_notes": str, "flags": [...]}"""

    def run(self, ctx, llm=None):
        orders, rejected, flags = [], [], []
        used = ctx["trades_today"]
        cash = ctx["cash"]
        for o in ctx["orders"]:
            px = o.get("price")
            if px is None or not math.isfinite(px) or px <= 0:
                rejected.append({"ticker": o["ticker"], "reason": "no valid live price"}); continue
            if int(o.get("shares", 0)) <= 0:
                rejected.append({"ticker": o["ticker"], "reason": "non-positive share count"}); continue
            if o["notional"] < config.MIN_TRADE_NOTIONAL and not (o["side"] == "SELL" and o.get("full_exit")):
                rejected.append({"ticker": o["ticker"], "reason": f"below ${config.MIN_TRADE_NOTIONAL:,} minimum"}); continue
            orders.append(o)

        if not ctx["init_mode"]:
            room = max(0, config.MAX_TRADES_PER_DAY - used)
            for o in orders[room:]:
                rejected.append({"ticker": o["ticker"], "reason": f"daily cap: {used} trade(s) already done today"})
            orders = orders[:room]

        sells = sorted([o for o in orders if o["side"] == "SELL"], key=lambda o: -o["notional"])
        buys = sorted([o for o in orders if o["side"] == "BUY"], key=lambda o: -o["notional"])
        avail = cash + sum(o["notional"] for o in sells)
        ok_buys = []
        for o in buys:
            if o["notional"] > avail + 1e-6:
                rejected.append({"ticker": o["ticker"], "reason": "would take cash below zero"}); continue
            avail -= o["notional"]
            ok_buys.append(o)
        final = sells + ok_buys

        if ctx.get("nav_jump_halt"):
            flags.append(ctx["nav_jump_halt"])
            return ({"execute": False, "approved_orders": [], "rejected": rejected,
                     "execution_notes": "HALTED: " + ctx["nav_jump_halt"], "flags": flags}, "rules")
        notes = (f"{len(final)} order(s) approved ({len(sells)} sells first, then {len(ok_buys)} buys); "
                 f"{len(rejected)} rejected. Trades used today after execution: "
                 f"{used + len(final)}{'' if ctx['init_mode'] else '/' + str(config.MAX_TRADES_PER_DAY)}"
                 f"{' (initialisation: cap exempt)' if ctx['init_mode'] else ''}.")
        return ({"execute": bool(final), "approved_orders": final, "rejected": rejected,
                 "execution_notes": notes, "flags": flags}, "rules")
