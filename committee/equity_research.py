import numpy as np

from .base import Agent, compact, clip

RATINGS = ("buy", "hold", "sell", "insufficient_data")


class EquityResearchAnalyst(Agent):
    name = "Equity Research Analyst"
    title = "Senior Equity Research Analyst (fundamentals)"
    tier = "fast"
    role_prompt = """Your score is a fundamentals-based QUALITY / RISK screen, not a price call:
a 2-week horizon is too short for fundamentals to drive returns (IPS C2.1). A high
score supports a larger position; a low score flags company-specific risk to weigh
against the momentum case. Primary metrics: debt/equity, free cash flow, margins
(balance-sheet safety matters most on a short, aggressive book). P/E, EV/EBITDA and
P/B are secondary context. Compare with sector peers within the shortlist.
The "flags" are computed for you (negative FCF, D/E above sector median, negative
margin): incorporate them, don't re-derive them. Every figure you cite must be one
you were given. Names marked insufficient_data get no score.
Score -1 (high risk) to +1 (high quality). Use the full range."""
    output_schema = """{
  "ratings": {
    "<TICKER>": {"rating": "buy"|"hold"|"sell", "score": number -1..1,
                 "thesis": "max 25 words citing given figures"}
  },
  "top_pick": "<TICKER>",
  "least_preferred": "<TICKER>"
}"""

    def build_prompt(self, ctx):
        f = {t: {k: v for k, v in d.items() if k not in ("synthetic", "missing_core_metrics")}
             for t, d in ctx["fundamentals"].items() if not d.get("insufficient_data")}
        flags = {t: ctx["research_flags"].get(t, []) for t in f if ctx["research_flags"].get(t)}
        note = " (NOTE: fundamentals are synthetic demo data)" if any(
            d.get("synthetic") for d in ctx["fundamentals"].values()) else ""
        return (f"FUNDAMENTALS{note}. Units: ratios as decimals (0.25 = 25%), "
                f"mkt_cap/fcf in $bn, debt_to_equity in %.\n{compact(f)}\n\nFLAGS:\n{compact(flags)}\n\n"
                f"Rate every ticker: {list(f)}")

    def mock(self, ctx):
        f = {t: d for t, d in ctx["fundamentals"].items() if not d.get("insufficient_data")}
        tick = list(f)
        if not tick:
            return {"ratings": {}, "top_pick": None, "least_preferred": None}

        def z(key):
            v = np.array([float(f[t].get(key)) if f[t].get(key) is not None else np.nan for t in tick])
            v = np.where(np.isnan(v), np.nanmean(v) if not np.all(np.isnan(v)) else 0, v)
            return (v - v.mean()) / (v.std() + 1e-9)

        raw = (z("roe") + z("net_margin") + z("fcf_bn") - z("debt_to_equity") - 0.5 * z("pe_fwd")) / 4
        ratings = {}
        for t, s in zip(tick, raw):
            sc = float(np.clip(s, -1, 1))
            ratings[t] = {"rating": "buy" if sc > 0.25 else "sell" if sc < -0.25 else "hold",
                          "score": round(sc, 2),
                          "thesis": f"[mock] D/E {f[t].get('debt_to_equity')}, FCF {f[t].get('fcf_bn')}bn, "
                                    f"margin {f[t].get('net_margin')}"}
        return {"ratings": ratings, "top_pick": max(ratings, key=lambda t: ratings[t]["score"]),
                "least_preferred": min(ratings, key=lambda t: ratings[t]["score"])}

    def validate(self, out, ctx):
        ratings = {}
        src = out.get("ratings") or {}
        for t, d in ctx["fundamentals"].items():
            if d.get("insufficient_data"):   # IPS C2.7: excluded this cycle, not neutral
                ratings[t] = {"rating": "insufficient_data", "score": 0.0,
                              "thesis": f"insufficient data ({d.get('missing_core_metrics')} core metrics missing)"}
                continue
            r = src.get(t) or {}
            rating = str(r.get("rating", "hold")).lower()
            ratings[t] = {"rating": rating if rating in RATINGS[:3] else "hold", "score": round(clip(r.get("score")), 2),
                          "thesis": str(r.get("thesis", "no view"))[:200],
                          "flags": ctx["research_flags"].get(t, [])}
        return {"ratings": ratings, "top_pick": out.get("top_pick"),
                "least_preferred": out.get("least_preferred")}
