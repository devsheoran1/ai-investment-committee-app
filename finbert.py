"""
FinBERT headline scorer (ProsusAI/finbert, local, MPS on the M2). IPS C4.6: every headline gets a deterministic
score as a cross-check; the Sentiment LLM then judges surprise vs expectation. Score = P(positive) - P(negative).
Results are cached per headline on disk, so re-runs and the intraday re-check cost nothing.
Never raises: if transformers/the model is unavailable, returns {} and Sentiment runs LLM-only.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle

import config

log = logging.getLogger("finbert")
MODEL_ID = "ProsusAI/finbert"
_CACHE_PATH = os.path.join(config.CACHE_DIR, "finbert_headlines.pkl")
_STATE: dict = {}


def _key(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if "cache" not in _STATE:
        try:
            with open(_CACHE_PATH, "rb") as f:
                _STATE["cache"] = pickle.load(f)
        except Exception:  # noqa: BLE001
            _STATE["cache"] = {}
    return _STATE["cache"]


def _model():
    if "model" not in _STATE:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(dev)
        labels = {v.lower(): k for k, v in mdl.config.id2label.items()}
        _STATE["model"] = (tok, mdl, dev, labels)
    return _STATE["model"]


def score(titles: list[str]) -> dict[str, float]:
    """{title: score in [-1, 1]}. Cached per headline."""
    cache = _load_cache()
    todo = list(dict.fromkeys(t for t in titles if t and _key(t) not in cache))
    if todo:
        try:
            import torch
            tok, mdl, dev, lab = _model()
            for i in range(0, len(todo), 32):
                batch = todo[i:i + 32]
                x = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
                with torch.no_grad():
                    p = mdl(**x).logits.softmax(-1).cpu().numpy()
                for t, row in zip(batch, p):
                    cache[_key(t)] = round(float(row[lab["positive"]] - row[lab["negative"]]), 3)
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:  # noqa: BLE001
            log.warning("FinBERT unavailable: %s", str(e)[:150])
    return {t: cache[_key(t)] for t in titles if t and _key(t) in cache}


def score_news(news: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Adds a 'finbert' field to every headline dict; returns the same structure."""
    allt = [h["title"] for items in news.values() for h in items]
    sc = score(allt) if allt else {}
    return {t: [{**h, "finbert": sc.get(h["title"])} for h in items] for t, items in news.items()}
