"""Buys by well-known US politicians (STOCK Act disclosures), a context line for the SENTIMENT agent.

Why Sentiment and not Quant: it is an alternative-data / flow signal, not a price rule, and disclosures arrive up to
45 days late, so the evidence is weak and it must not move a score on its own. It is shown to the Sentiment analyst as
one extra context row per name (only names with a recent buy); nothing else in the pipeline reads it.
Source: Quiver Quant congress-trading API when QUIVER_API_KEY is set, else the public House/Senate Stock Watcher
JSON dumps. Any failure -> {} (no row). Cached daily.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import config

DAYS = 45                              # transaction date window; disclosure lag is up to 45 days
URLS = ("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json")
QUIVER = "https://api.quiverquant.com/beta/live/congresstrading"


def normalise(rows: list[dict]) -> list[dict]:
    """Different feeds, one shape: {member, ticker, date, type, amount_lo, amount_hi}."""
    out = []
    for r in rows:
        who = r.get("Representative") or r.get("representative") or r.get("senator") or r.get("Name") or ""
        tick = (r.get("Ticker") or r.get("ticker") or "").strip().upper()
        typ = str(r.get("Transaction") or r.get("type") or "").lower()
        day = str(r.get("TransactionDate") or r.get("transaction_date") or r.get("Traded") or "")[:10]
        if not tick or tick == "--" or not day:
            continue
        out.append({"member": who.strip(), "ticker": tick, "date": day,
                    "type": "buy" if "purchase" in typ or typ == "buy" else ("sell" if "sale" in typ or typ == "sell" else typ),
                    "amount": str(r.get("Range") or r.get("amount") or r.get("Amount") or "")})
    return out


def summarise(rows: list[dict], tickers, top: list[str], days: int = DAYS, today: date | None = None) -> dict[str, dict]:
    """Buys by the watch-listed members in the window -> {ticker: {"buyers": [...], "n_buys": n, "last": date}}."""
    today = today or date.today()
    cutoff = (today - timedelta(days=days)).isoformat()
    keys = [k.lower() for k in top]
    want = set(tickers)
    out: dict[str, dict] = {}
    for r in rows:
        if r["ticker"] not in want or r["type"] != "buy" or r["date"] < cutoff or r["date"] > today.isoformat():
            continue
        if not any(k in r["member"].lower() for k in keys):
            continue
        o = out.setdefault(r["ticker"], {"buyers": [], "n_buys": 0, "last": r["date"], "amounts": []})
        if r["member"] not in o["buyers"]:
            o["buyers"].append(r["member"])
        o["n_buys"] += 1
        o["last"] = max(o["last"], r["date"])
        if r["amount"]:
            o["amounts"].append(r["amount"])
    return out


def _download() -> list[dict]:
    import requests
    key = os.getenv("QUIVER_API_KEY")
    rows: list[dict] = []
    if key:
        r = requests.get(QUIVER, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}, timeout=15)
        r.raise_for_status()
        return normalise(r.json())
    for u in URLS:
        try:
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            rows += normalise(r.json())
        except Exception:  # noqa: BLE001
            continue
    return rows


def get_politician_buys(tickers, top: list[str] | None = None) -> dict[str, dict]:
    """{ticker: summary} for the watch-listed politicians' buys; {} on any failure or when synthetic."""
    import data
    tickers = sorted(set(tickers))
    if not tickers or data.is_synthetic():
        return {}
    try:
        rows = data._cached("politician_trades_v1", _download, keep_if=bool)
    except Exception:  # noqa: BLE001
        return {}
    return summarise(rows or [], tickers, top or config.POLITICIAN_WATCHLIST)
