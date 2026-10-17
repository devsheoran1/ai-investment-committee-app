from .base import Agent, compact, clip

POS = ("beat", "surge", "record", "upgrade", "raises", "growth", "strong", "rally", "wins", "soar", "buy")
NEG = ("miss", "falls", "cut", "downgrade", "lawsuit", "probe", "weak", "plunge", "recall", "slump", "fine")


class SentimentAnalyst(Agent):
    name = "Sentiment Analyst"
    title = "News & Sentiment Analyst"
    tier = "fast"
    role_prompt = """You read the last 3 days of headlines per name (Yahoo Finance / Google News,
dated [MM-DD]) and flag catalysts that can move the price in the next days (IPS C4.1):
earnings vs expectations, guidance changes, M&A, regulation, big announcements.
What matters is not whether news is good or bad but whether the market already
expected it (C4.3). Ignore opinion pieces, "stocks to watch" lists and listicles.
Analyst upgrades/downgrades, price-target changes, earnings previews for a report due
in the next days, and sharp moves with a stated cause ARE price-relevant: score them
(typically 0.2-0.5 in absolute value). No headlines or only noise = score 0. Score -1..+1; keep most scores modest
(|score| < 0.5) unless the news is material. A score <= -0.8 on a held name
(fraud, very bad earnings, big guidance cut) forces a sell proposal (C4.4),
so use it only for genuinely severe news."""
    output_schema = """{
  "sentiment": {"<TICKER>": {"score": number -1..1, "key_driver": "max 15 words or 'no material news'"}},
  "market_mood": "max 25 words"
}"""

    def build_prompt(self, ctx):
        return f"RECENT HEADLINES PER STOCK:\n{compact(ctx['news'])}"

    def mock(self, ctx):
        out = {}
        for t in ctx["tickers"]:
            heads = ctx["news"].get(t, [])
            text = " ".join(heads).lower()
            s = sum(w in text for w in POS) - sum(w in text for w in NEG)
            out[t] = {"score": max(-0.6, min(0.6, 0.15 * s)),
                      "key_driver": f"[mock] {heads[0][:70]}" if heads else "no material news"}
        return {"sentiment": out, "market_mood": "[mock] keyword-based sentiment"}

    def validate(self, out, ctx):
        src = out.get("sentiment") or {}
        sent = {}
        for t in ctx["tickers"]:
            r = src.get(t) or {}
            score = clip(r.get("score")) if ctx["news"].get(t) else 0.0   # no news -> no view
            sent[t] = {"score": round(score, 2), "key_driver": str(r.get("key_driver", "no material news"))[:150]}
        return {"sentiment": sent, "market_mood": str(out.get("market_mood", ""))[:300]}
