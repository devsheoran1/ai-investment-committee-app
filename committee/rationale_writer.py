"""
Trade Rationale Writer (IPS C0.4 / E2, Oct 20). One gpt-oss-20b call after execution writes 2-3 plain-English
sentences PER TRADE, in the committee's voice, for someone who isn't watching the numbers: what the company is,
why now (the specific signal or news), what it does to the portfolio, and the exit trigger. Facts only from the
agent outputs it is given. The Risk verdict follows in one line, added by Python.

Guard: a rationale that names a ticker other than its own trade (the dry run pasted "add FAS" onto trades that
had nothing to do with FAS) or a price the facts don't contain falls back to the deterministic template.
Not part of ROSTER: it documents decisions, it doesn't make them.
"""
import re

from .base import Agent, compact


def template(f: dict) -> str:
    """Deterministic rationale (mock mode and fallback)."""
    who = f.get("name") or f["ticker"]
    what = f.get("sector") or ""
    if f["side"] == "BUY":
        why = f.get("why") or f"its composite score is {f.get('composite_10', 'n/a')}/10"
        s1 = f"Bought {who}{f' ({what})' if what else ''} because {why}."
        s2 = (f"It takes the position from {f['weight_before']:.1%} to {f['weight_after']:.1%} of the fund"
              f"{' as a replacement to keep at least 10 holdings' if f.get('replacement') else ''}.")
        s3 = f"We exit if it closes below ${f['invalidation_px']:.2f}." if f.get("invalidation_px") else ""
    else:
        why = f.get("forced_reason") or f.get("why") or "its scores fell behind the best alternatives"
        s1 = f"Sold {who}{f' ({what})' if what else ''}: {why}."
        s2 = f"The position goes from {f['weight_before']:.1%} to {f['weight_after']:.1%} of the fund."
        s3 = ""
    return " ".join(x for x in (s1, s2, s3) if x)


class TradeRationaleWriter(Agent):
    name = "Trade Rationale Writer"
    title = "Trade Rationale Writer (plain-English trade notes)"
    tier = "fast"
    role_prompt = """You write the trade log a client would read. For EACH trade, write 2-3 short plain-English
sentences in the committee's voice ("Bought ...", "Sold ..."): (1) what the company or ETF is and why now, citing
the specific signal or news given in its facts; (2) what the trade does to the portfolio (position size, risk,
replacing an exit); (3) for buys, the exit trigger: "We exit if it closes below $X" using its invalidation price.
Use ONLY the facts given for that trade. Never mention another ticker or company, never invent numbers, no
jargon like "composite" or "conviction" unless explained in plain words.
Example: "Bought Phillips 66, a US refiner, because refiners are leading the market on strong margins and our
momentum model ranks it in the top 3. It also lifts the fund's beta, which was too low. We exit if it closes
below $276." """
    output_schema = """{"rationales": {"<TICKER>": "2-3 sentences"}}"""

    def build_prompt(self, ctx):
        return f"EXECUTED TRADES AND THEIR FACTS:\n{compact({f['ticker']: f for f in ctx['facts']})}"

    def mock(self, ctx):
        return {"rationales": {f["ticker"]: template(f) for f in ctx["facts"]}}

    def validate(self, out, ctx):
        src = out.get("rationales") or {}
        known = set(ctx.get("all_tickers") or [])
        res, fallbacks = {}, []
        for f in ctx["facts"]:
            txt = str(src.get(f["ticker"], "")).strip()
            others = {w for w in re.findall(r"\b[A-Z]{2,5}\b", txt) if w in known and w != f["ticker"]}
            prices_ok = all(abs(float(p.replace(",", "")) - float(f.get("invalidation_px") or 0)) < 0.02 or
                            str(p) in compact(f) for p in re.findall(r"\$([\d,]+\.?\d*)", txt))
            if not txt or others or not prices_ok or len(txt) > 700:
                txt = template(f)
                fallbacks.append(f["ticker"])
            res[f["ticker"]] = txt
        return {"rationales": res, "fallback_used_for": fallbacks}
