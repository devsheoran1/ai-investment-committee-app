"""
Sentiment / News Analyst (IPS C4.1-C4.6, owner Mer).

    inputs   up to 10 headlines per name over 5 days (Yahoo RSS, Google News incl. Reuters/Bloomberg/WSJ, SEC 8-K),
             each with source + Madrid timestamp + a FinBERT score (deterministic cross-check), plus earnings context
    judgment the LLM scores SURPRISE vs expectation (C4.3), citing the headline(s) it rests on
    no news  -> score null, shown "n/a" (never 0.0); its 30% weight is redistributed pro rata (C5.2)
    guard    > 70% of names WITH headlines scored exactly 0.0 -> one retry on gpt-oss-120b (C4.6)
"""
from .base import Agent, compact, clip

POS = ("beat", "surge", "record", "upgrade", "raises", "growth", "strong", "rally", "wins", "soar", "buy")
NEG = ("miss", "falls", "cut", "downgrade", "lawsuit", "probe", "weak", "plunge", "recall", "slump", "fine")
LAZY_SHARE = 0.70
MAX_HEADS_TO_LLM = 6          # FinBERT scores all 10; the LLM reads the 6 newest (Groq 8K tokens/min request size)
MAX_REQUEST_CHARS = 14000     # ~3.5K input tokens; Groq counts input + max_tokens (3000) against 8K/min


class SentimentAnalyst(Agent):
    name = "Sentiment Analyst"
    title = "News & Sentiment Analyst"
    tier = "fast"
    role_prompt = """You find news that can move a price in the next days (IPS C4.1): earnings vs expectations,
guidance changes, M&A, regulation, big announcements, analyst rating changes. What matters most is not whether
news is good or bad but whether the market already expected it (C4.3): a beat that was priced in is ~0, a surprise
is not. Each headline comes with its source, Madrid timestamp and a FinBERT tone score (-1..+1); FinBERT reads tone
only, so use it as a cross-check and overrule it when the surprise is clearly different. Check more than one
source before a big score. Ignore opinion pieces, "stocks to watch" lists and listicles.
Score every ticker that HAS headlines from -1 to +1 and name the headline(s) your score rests on. Most scores
should be modest; |score| >= 0.5 needs a material, unexpected event. A score <= -0.8 on a held name (fraud, very
bad earnings, big guidance cut) forces a sell proposal, so use it only for genuinely severe news.
If a ticker's headlines are all noise, score it 0 and say "noise only". Tickers with no headlines are not sent."""
    output_schema = """{
  "sentiment": {"<TICKER>": {"score": number -1..1, "key_driver": "max 15 words", "based_on": ["headline #ids it rests on"]}},
  "market_mood": "max 25 words"
}"""

    @staticmethod
    def _rows(ctx):
        rows = {}
        for t in ctx["tickers"]:
            heads = ctx["news"].get(t) or []
            if not heads:
                continue
            rows[t] = [{"id": i, "ts": h.get("ts", "")[5:], "src": h.get("source", "")[:20], "finbert": h.get("finbert"),
                        "h": h.get("title", "")[:110]} for i, h in enumerate(heads[:MAX_HEADS_TO_LLM])]
        return rows

    def build_prompt(self, ctx):
        rows = self._rows(ctx)
        txt = f"HEADLINES PER TICKER (Madrid time; finbert = tone score):\n{compact(rows)}"
        earn = {t: e for t, e in (ctx.get("earnings") or {}).items() if t in rows and e}
        if earn:
            txt += f"\n\nEARNINGS CONTEXT (B7):\n{compact(earn)}"
        return txt

    def mock(self, ctx):
        out = {}
        for t, heads in self._rows(ctx).items():
            fb = [h["finbert"] for h in heads if h.get("finbert") is not None]
            if fb:
                s = max(-0.6, min(0.6, sum(fb) / len(fb)))
            else:
                text = " ".join(h["h"] for h in heads).lower()
                s = max(-0.6, min(0.6, 0.15 * (sum(w in text for w in POS) - sum(w in text for w in NEG))))
            out[t] = {"score": round(s, 2), "key_driver": f"[mock] {heads[0]['h'][:60]}", "based_on": [0]}
        return {"sentiment": out, "market_mood": "[mock] FinBERT average per name"}

    def run(self, ctx, llm):
        """Batches the names so no single request exceeds Groq's request-size limit (Oct 20: 413 on 43 names)."""
        def size(ticks):            # the WHOLE request: system prompt + brief + headlines + earnings context
            c = {**ctx, "tickers": ticks}
            return len(self.system_prompt) + len(self.build_prompt(c)) + min(900, len(ctx.get("morning_brief") or ""))
        chunks, todo = [], [list(ctx["tickers"])]
        while todo:
            ch = todo.pop(0)
            if len(ch) > 1 and size(ch) > MAX_REQUEST_CHARS:
                todo[:0] = [ch[:len(ch) // 2], ch[len(ch) // 2:]]
            else:
                chunks.append(ch)
        if len(chunks) <= 1:
            return self._run_one(ctx, llm)
        merged, moods, models = {}, [], []
        for ch in chunks:
            o, m = self._run_one({**ctx, "tickers": ch}, llm)
            merged.update(o["sentiment"]); moods.append(o.get("market_mood", "")); models.append(m)
        return {"sentiment": {t: merged[t] for t in ctx["tickers"]}, "market_mood": next((x for x in moods if x), ""),
                "batches": len(chunks)}, " + ".join(dict.fromkeys(models)) + f" ({len(chunks)} batches)"

    def _run_one(self, ctx, llm):
        out, model = super().run(ctx, llm)
        scored = [v for v in out["sentiment"].values() if v["score"] is not None]
        if llm.provider != "mock" and len(scored) >= 5 and \
                sum(1 for v in scored if v["score"] == 0.0) / len(scored) > LAZY_SHARE:
            # C4.6 sanity guard: the small model went lazy -> one retry on the senior model
            raw, model2 = llm.chat_json(self.system_prompt, self.build_prompt(ctx), tier="smart",
                                        agent_name=self.name, mock_fn=lambda: self.mock(ctx))
            out = self.validate(raw if isinstance(raw, dict) else {}, ctx)
            out["retried_on"] = model2
            model = f"{model} -> retry {model2} (>{LAZY_SHARE:.0%} exact zeros)"
        return out, model

    def validate(self, out, ctx):
        src = out.get("sentiment") or {}
        sent = {}
        for t in ctx["tickers"]:
            heads = ctx["news"].get(t) or []
            fb = [h.get("finbert") for h in heads if h.get("finbert") is not None]
            fb_avg = round(sum(fb) / len(fb), 3) if fb else None
            if not heads:                                       # no news -> n/a, never 0.0 (C4.6)
                sent[t] = {"score": None, "key_driver": "no news", "finbert": None, "llm_score": None, "n_headlines": 0}
                continue
            r = src.get(t) or {}
            llm_score = None if r.get("score") is None else round(clip(r.get("score")), 2)
            ids = [i for i in (r.get("based_on") or []) if isinstance(i, int) and 0 <= i < len(heads)]
            sent[t] = {"score": llm_score,                       # final = LLM score (C4.6); FinBERT logged alongside
                       "key_driver": str(r.get("key_driver", "no answer" if llm_score is None else ""))[:150],
                       "finbert": fb_avg, "llm_score": llm_score, "n_headlines": len(heads),
                       "based_on": [f"{heads[i].get('source', '')}: {heads[i].get('title', '')[:100]}" for i in ids[:3]]}
        return {"sentiment": sent, "market_mood": str(out.get("market_mood", ""))[:300]}
