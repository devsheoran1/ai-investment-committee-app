"""Same-day agent-output cache (Oct 27). The brief allows 3 runs a day, so runs 2 and 3 should only pay for what changed:

    Morning Brief   once per day
    Macro           once per day (LLM text only; regime/tilts are re-validated from today's rules); re-run if VIX moved > 10%
    Research        per ticker, keyed on that ticker's fundamentals row
    Sentiment       per ticker, keyed on the hash of its headlines
    every other     reused when system prompt + user prompt are byte-identical to a same-day call
Stored in the fund DB (table day_cache, created on first use) so a --reset or a test DB starts clean. Mock runs never cache."""
from __future__ import annotations

import hashlib
import json

import config
import database as db


def _init(c) -> None:
    c.execute("CREATE TABLE IF NOT EXISTS day_cache (day TEXT, kind TEXT, key TEXT, run_id INTEGER, payload TEXT, "
              "PRIMARY KEY (day, kind, key))")


def h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def enabled(llm, replay: bool = False) -> bool:
    return bool(getattr(config, "DAY_CACHE", True)) and llm.provider != "mock" and not replay


def get(kind: str, key: str):
    """-> (payload, run_id) for today, or None."""
    with db.conn() as c:
        _init(c)
        r = c.execute("SELECT payload, run_id FROM day_cache WHERE day=? AND kind=? AND key=?", (db.today(), kind, key)).fetchone()
    return (json.loads(r["payload"]), r["run_id"]) if r else None


def put(kind: str, key: str, run_id, payload) -> None:
    with db.conn() as c:
        _init(c)
        c.execute("INSERT OR REPLACE INTO day_cache (day, kind, key, run_id, payload) VALUES (?,?,?,?,?)",
                  (db.today(), kind, key, run_id, json.dumps(payload, default=str)))


def active(ctx: dict) -> bool:
    return bool(ctx.get("day_cache"))
