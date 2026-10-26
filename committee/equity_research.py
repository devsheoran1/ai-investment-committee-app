import numpy as np

import daycache

from .base import Agent, as_map, compact, clip, entry, split_by_size

RATINGS = ("buy", "hold", "sell", "insufficient_data")
MAX_REQUEST_CHARS = 14000     # whole request (system + brief + fundamentals); same budget as Sentiment (Groq 413 on 46 names, Oct 26)


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
Score -1 (high risk) to +1 (high quality). Use the full range.
EARNINGS PREVIEW (C2.8): for names reporting within 5 trading days you get the last 4 quarters' EPS surprises, the
current estimate and the recommendation trend; add one line on whether the setup favours a beat given expectations."""
    output_schema = """{
  "ratings": {
    "<TICKER>": {"rating": "buy"|"hold"|"sell", "score": number -1..1,
                 "thesis": "max 25 words citing given figures",
                 "earnings_preview": "only for names reporting soon: max 25 words, else omit"}
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
        earn = {t: {k: e.get(k) for k in ("next_date", "last4", "current_estimate", "rec_trend")}
                for t, e in (ctx.get("earnings") or {}).items() if t in f and e.get("reports_soon")}
        return (f"FUNDAMENTALS{note}. Units: ratios as decimals (0.25 = 25%), "
                f"mkt_cap/fcf in $bn, debt_to_equity in %.\n{compact(f)}\n\nFLAGS:\n{compact(flags)}\n\n"
                + (f"EARNINGS PREVIEW INPUTS (reporting within 5 trading days):\n{compact(earn)}\n\n" if earn else "")
                + f"Rate every ticker: {list(f)}")

    def run(self, ctx, llm):
        """Oct 27: (1) one request never exceeds MAX_REQUEST_CHARS: names are halved until each batch fits;
        (2) a name's rating is reused for the rest of the day while its fundamentals row is unchanged."""
        fund = ctx["fundamentals"]
        live = [t for t, d in fund.items() if not d.get("insufficient_data")]
        keys = {t: daycache.h(t, fund[t], ctx["research_flags"].get(t)) for t in live}
        hits = {t: g for t in live if daycache.active(ctx) and (g := daycache.get("research", keys[t]))}
        todo = [t for t in live if t not in hits]
        dead = {t: d for t, d in fund.items() if d.get("insufficient_data")}

        def sub(ch):
            return {**ctx, "fundamentals": {**{t: fund[t] for t in ch}, **dead}}

        def size(ch):
            return len(self.system_prompt) + len(self.build_prompt(sub(ch))) + min(900, len(ctx.get("morning_brief") or ""))
        chunks = split_by_size(todo, size, MAX_REQUEST_CHARS) if todo else []
        ratings, models, picks = {}, [], []
        for ch in chunks:
            o, m = super().run(sub(ch), llm)
            ratings.update(o["ratings"]); models.append(m); picks.append(o)
            if daycache.active(ctx):
                for t in ch:
                    if t in o["ratings"]:
                        daycache.put("research", keys[t], ctx.get("run_id"), o["ratings"][t])
        for t, (r, _) in hits.items():
            ratings[t] = r
        for t in dead:
            ratings.setdefault(t, self.validate({}, sub([]))["ratings"][t])
        ratings = {t: ratings[t] for t in fund if t in ratings}
        scored = {t: r["score"] for t, r in ratings.items() if r["rating"] != "insufficient_data"}
        if len(picks) == 1 and not hits:
            top, least = picks[0]["top_pick"], picks[0]["least_preferred"]
        else:                                              # merged batches / cache: best and worst by score
            top = max(scored, key=scored.get) if scored else None
            least = min(scored, key=scored.get) if scored else None
        out = {"ratings": ratings, "top_pick": top, "least_preferred": least}
        label = " + ".join(dict.fromkeys(models))
        if len(chunks) > 1:
            out["batches"] = len(chunks)
            label += f" ({len(chunks)} batches)"
        if hits:
            runs = sorted({r for _, r in hits.values() if r})
            out["reused"] = {"names": len(hits), "of": len(live), "from_runs": runs}
            label = (label + " + " if label else "") + f"reused {len(hits)}/{len(live)} names from run {', '.join(map(str, runs))}"
        return out, label or "no names to rate"

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
        src = as_map(out.get("ratings"))
        for t, d in ctx["fundamentals"].items():
            if d.get("insufficient_data"):   # IPS C2.7: excluded this cycle, not neutral
                ratings[t] = {"rating": "insufficient_data", "score": 0.0,
                              "thesis": f"insufficient data ({d.get('missing_core_metrics')} core metrics missing)"}
                continue
            r = entry(src, t)
            rating = str(r.get("rating", "hold")).lower()
            ratings[t] = {"rating": rating if rating in RATINGS[:3] else "hold", "score": round(clip(r.get("score")), 2),
                          "thesis": str(r.get("thesis", "no view"))[:200],
                          "flags": ctx["research_flags"].get(t, [])}
            if r.get("earnings_preview") and ((ctx.get("earnings") or {}).get(t) or {}).get("reports_soon"):
                ratings[t]["earnings_preview"] = str(r["earnings_preview"])[:200]
        pick = lambda x: x if isinstance(x, str) and x in ratings else None     # never a ticker outside the shortlist
        return {"ratings": ratings, "top_pick": pick(out.get("top_pick")),
                "least_preferred": pick(out.get("least_preferred"))}
