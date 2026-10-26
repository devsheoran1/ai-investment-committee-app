"""
What an "agent" is in this project:

    agent = role (system prompt) + inputs from Python tools + a JSON answer

Each agent:
  1. build_prompt(ctx)  -> turns the tool outputs it needs into a compact brief
  2. LLM call           -> the model reads the brief and answers in JSON
  3. validate(out, ctx) -> Python checks/clips the answer so a bad model reply
                           can never break the fund (e.g. weights > 100%)
  4. mock(ctx)          -> rule-based answer used when LLM_PROVIDER=mock

To build a new agent: subclass Agent, fill in those four methods. Done.
"""
from __future__ import annotations

import json

import config

HOUSE_RULES = f"""You are a member of the investment committee of "{config.FUND_NAME}",
an AI-run paper-trading fund in a 2-week class competition (IE MiF, Prof. Manero).
Mandate: ${config.STARTING_CAPITAL:,.0f}, S&P 500 stocks plus leveraged ETFs, long only
(downside protection only via inverse ETFs in a risk-off regime). Objective: maximise
absolute return over ~2 weeks within hard risk limits enforced by Python. Horizon is
days to weeks: ignore multi-year theses; momentum, earnings and news catalysts matter most.
At most {config.MAX_TRADES_PER_DAY} trades per day. "No trade" is a valid decision.
Committee rules:
- Use ONLY the data in the brief. Never invent numbers, news or events.
- If the data is missing or insufficient, say so and stay neutral (score 0).
- Be concise and specific, like a buy-side professional writing for a CIO.
- Reply with ONE valid JSON object that matches the requested schema exactly. No prose outside the JSON."""


def compact(obj) -> str:
    """JSON without whitespace = fewer tokens = fewer rate-limit hits."""
    return json.dumps(obj, separators=(",", ":"), default=str)


def as_map(x) -> dict:
    """Top-level LLM field that should be a dict; anything else (list, str, None) = empty. Never crashes."""
    return x if isinstance(x, dict) else {}


def entry(src: dict, t: str, key: str = "score") -> dict:
    """Per-ticker LLM entry, schema-drift tolerant: {"NUE": {"score": -0.1}} or flat {"NUE": -0.1} -> dict.
    Missing / malformed -> {} (the caller's fallback applies)."""
    v = as_map(src).get(t)
    if isinstance(v, dict):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return {key: v}
    if isinstance(v, str):
        try:
            return {key: float(v)}
        except ValueError:
            return {}
    return {}


def clip(x, lo=-1.0, hi=1.0, default=0.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


class Agent:
    name: str = "Agent"
    title: str = ""
    tier: str = "fast"              # "fast" (junior) or "smart" (senior) model
    role_prompt: str = ""           # who the agent is and how it thinks
    output_schema: str = ""         # the JSON shape we expect back

    @property
    def system_prompt(self) -> str:
        return f"{HOUSE_RULES}\n\nYOUR ROLE: {self.title}\n{self.role_prompt}\n\nOUTPUT JSON SCHEMA:\n{self.output_schema}"

    # --- override these ----------------------------------------------------- #
    def build_prompt(self, ctx: dict) -> str:
        raise NotImplementedError

    def mock(self, ctx: dict) -> dict:
        raise NotImplementedError

    def validate(self, out: dict, ctx: dict) -> dict:
        return out

    # --- shared ------------------------------------------------------------- #
    use_brief: bool = True          # the Morning Brief (C0.3) is prepended to every agent's prompt

    def run(self, ctx: dict, llm) -> tuple[dict, str]:
        prompt = self.build_prompt(ctx)
        if self.use_brief and ctx.get("morning_brief"):
            prompt = f"MORNING BRIEF (today, pre-open):\n{ctx['morning_brief'][:900]}\n\n{prompt}"
        raw, model = llm.chat_json(self.system_prompt, prompt, tier=self.tier,
                                   agent_name=self.name, mock_fn=lambda: self.mock(ctx))
        self.last_call = dict(getattr(llm, "last", {}) or {})          # Stage 8: what this agent saw
        return self.validate(raw if isinstance(raw, dict) else {}, ctx), model


def split_by_size(tickers, size_fn, limit: int) -> list[list]:
    """Halve a ticker list until every chunk's WHOLE request (size_fn(chunk) chars) fits `limit` (Groq 413/8K-per-min)."""
    chunks, todo = [], [list(tickers)]
    while todo:
        ch = todo.pop(0)
        if len(ch) > 1 and size_fn(ch) > limit:
            todo[:0] = [ch[:len(ch) // 2], ch[len(ch) // 2:]]
        else:
            chunks.append(ch)
    return chunks


_NOT_TICKERS = {"ON", "IT", "ALL", "ARE", "NOW", "BIG", "CEO", "USD", "ETF", "NAV", "VIX", "IPS", "RSI", "PM", "CIO", "AI",
                "EPS", "GDP", "FED", "CPI", "FOMC", "MACD", "LSTM", "SEC", "NYSE", "US", "UK", "EU", "DD", "OK", "KEY", "CAT"}


def scrub_refs(text: str, allowed: set, universe: set) -> tuple[str, list[str]]:
    """Oct 27 (CIO cited IEX/PSX/KO that were not part of the book): drop every sentence that names a known ticker which is
    neither held, proposed nor otherwise in `allowed`. Returns (clean text, dropped tickers) so the caller can log them."""
    import re
    bad, keep = [], []
    for sent in re.split(r"(?<=[.;!?])\s+", str(text or "")):
        toks = {t for t in re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", sent) if t in universe and t not in _NOT_TICKERS and len(t) >= 2}
        off = toks - set(allowed)
        if off:
            bad += sorted(off)
        else:
            keep.append(sent)
    return " ".join(keep).strip(), sorted(set(bad))
