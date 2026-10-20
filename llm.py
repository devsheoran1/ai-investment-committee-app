"""
One small wrapper around every LLM provider.

All agents call `LLMClient.chat_json(system, user)` and get back a Python dict.
The wrapper handles: picking the model, JSON parsing, and waiting politely
when the free tier says "slow down" (HTTP 429).
"""
from __future__ import annotations

import json
import os
import re
import time

import config


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```fences``` or add chatter. Dig it out."""
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        if self.provider not in config.PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'. "
                           f"Pick one of {list(config.PROVIDERS)}")
        self.cfg = config.PROVIDERS[self.provider]
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "rate_limited": 0}
        self._client = None
        if self.provider != "mock":
            from openai import OpenAI
            key_env = self.cfg["key_env"]
            api_key = os.getenv(key_env) if key_env else "ollama"
            if key_env and not api_key:
                raise LLMError(
                    f"{key_env} is not set. Copy .env.example to .env and paste your key, "
                    f"or set LLM_PROVIDER=mock to run without AI.")
            # max_retries=0: we do our own rate-limit-aware retries below
            self._client = OpenAI(base_url=self.cfg["base_url"], api_key=api_key,
                                  max_retries=0, timeout=120)

    def model_for(self, tier: str, agent_name: str | None = None) -> str:
        if agent_name and agent_name in config.MODEL_OVERRIDES:
            return config.MODEL_OVERRIDES[agent_name]
        return self.cfg[tier]

    # ------------------------------------------------------------------ #
    def chat_json(self, system: str, user: str, tier: str = "fast",
                  agent_name: str | None = None, mock_fn=None) -> tuple[dict, str]:
        """Returns (parsed_json, model_used)."""
        import time as _time
        self.last = {"system": system, "user": user, "raw": None, "tokens": 0, "latency_s": 0.0}
        if self.provider == "mock":
            if mock_fn is None:
                raise LLMError("mock provider needs a mock_fn")
            self.usage["calls"] += 1
            out = mock_fn()
            self.last["raw"] = json.dumps(out, default=str)[:20000]
            return out, "mock"
        _t0 = _time.time()

        model = self.model_for(tier, agent_name)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        use_json_mode = self.cfg.get("json_mode", False)
        last_err: Exception | None = None
        max_out = 3000   # Groq counts prompt + max_tokens against the per-minute limit: keep it modest

        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                kwargs = dict(model=model, messages=messages,
                              temperature=config.LLM_TEMPERATURE, max_tokens=max_out)
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.cfg.get("extra"):
                    kwargs["extra_body"] = self.cfg["extra"]
                resp = self._client.chat.completions.create(**kwargs)
                self.usage["calls"] += 1
                if resp.usage:
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
                content = resp.choices[0].message.content or ""
                self.last.update(raw=content[:20000], latency_s=round(_time.time() - _t0, 2),
                                 tokens=(resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0)
                try:
                    return _extract_json(content), model
                except (ValueError, json.JSONDecodeError) as e:
                    last_err = e
                    messages = messages + [
                        {"role": "assistant", "content": content[:4000]},
                        {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object, nothing else."},
                    ]
                    continue
            except Exception as e:  # noqa: BLE001 - provider SDK errors vary
                last_err = e
                status = getattr(e, "status_code", None)
                if status == 429:
                    self.usage["rate_limited"] += 1
                    n429 = getattr(self, "_n429", 0) + 1
                    self._n429 = n429
                    if (config.JUNIOR_GEMINI_FALLBACK and tier == "fast" and self.provider == "groq"
                            and n429 >= config.JUNIOR_GEMINI_FALLBACK_AFTER_429):
                        self._n429 = 0
                        self.usage["gemini_fallbacks"] = self.usage.get("gemini_fallbacks", 0) + 1
                        return LLMClient("gemini").chat_json(system, user, tier="fast", agent_name=agent_name, mock_fn=mock_fn)
                    wait = _retry_after(e) or min(60, 5 * 2 ** attempt)
                    time.sleep(wait)
                    continue
                if status == 413 and max_out > 1000:
                    max_out = max(1000, max_out // 2)    # "request too large": shrink the output budget and retry
                    continue
                if status == 400 and use_json_mode:
                    # some models/providers reject JSON mode -> fall back to plain text
                    use_json_mode = False
                    continue
                if status in (500, 502, 503, 504) or status is None:
                    time.sleep(min(30, 2 * 2 ** attempt))
                    continue
                raise LLMError(f"{self.provider}/{model} failed: {e}") from e
        raise LLMError(f"{self.provider}/{model} failed after retries: {last_err}")


def _retry_after(err) -> float | None:
    try:
        headers = err.response.headers
        val = headers.get("retry-after")
        return float(val) + 1 if val else None
    except Exception:  # noqa: BLE001
        return None
