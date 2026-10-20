"""
Morning Brief (IPS C0.3, Stage 5). One pre-open call on Gemini Flash (free tier, the `gemini` provider), falling
back to gpt-oss-120b on Groq. Inputs: ES/NQ/YM futures, Europe/Asia closes, VIX, 10y, DXY, oil, sector ETF moves,
~60-100 market + held-name headlines, earnings due this week, macro releases (config.MACRO_CALENDAR).
Output: two paragraphs (what happened overnight and why it matters for US equities today; what the fund should
watch) + tickers in play. Fed to every agent, the Devil's Advocate and the daily report.

Note: the IPS doc (C0.3) says "one gpt-oss-20b call, one paragraph"; the Oct 20 spec says Gemini Flash, two
paragraphs. We follow the spec (newer) and fall back to Groq, so the brief never blocks a run.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

import config
import data
from llm import LLMClient

MARKETS = {"ES=F": "S&P 500 futures", "NQ=F": "Nasdaq-100 futures", "YM=F": "Dow futures",
           "^STOXX50E": "Euro Stoxx 50", "^GDAXI": "DAX", "^FTSE": "FTSE 100", "^N225": "Nikkei 225",
           "^HSI": "Hang Seng", "^VIX": "VIX", "^TNX": "US 10y yield", "DX-Y.NYB": "Dollar index", "CL=F": "WTI oil"}
SYSTEM = """You are the fund's pre-market strategist writing the Morning Brief for an AI investment committee that
trades S&P 500 stocks and leveraged ETFs over a 2-week competition. Use ONLY the data given: no invented numbers.
Reply with ONE JSON object:
{"overnight": "paragraph 1, max 110 words: what happened overnight and why it matters for US equities today",
 "watch": "paragraph 2, max 90 words: what the fund should watch today: sectors, held names, scheduled events",
 "tickers_in_play": ["max 8 tickers from the headlines or holdings, most important first"]}"""


def _moves() -> dict:
    try:
        import yfinance as yf
        raw = yf.download(list(MARKETS) + list(data.SECTOR_ETFS.values()), period="7d", interval="1d",
                          auto_adjust=True, progress=False, threads=True)["Close"]
        out = {}
        for t in raw.columns:
            s = raw[t].dropna()
            if len(s) >= 2:
                chg = float(s.iloc[-1] - s.iloc[-2]) if t == "^TNX" else float(s.iloc[-1] / s.iloc[-2] - 1) * 100
                out[MARKETS.get(t, t)] = {"last": round(float(s.iloc[-1]), 2),
                                          ("chg_bp" if t == "^TNX" else "chg_pct"): round(chg * (100 if t == "^TNX" else 1), 2)}
        return out
    except Exception:  # noqa: BLE001
        return {}


def _market_headlines(n: int = 40) -> list[dict]:
    import requests
    feeds = [("https://news.google.com/rss/search?q=stock+market+today+when:1d&hl=en-US&gl=US&ceid=US:en", "Google News"),
             ("https://news.google.com/rss/search?q=(futures+OR+treasury+OR+fed)+site:reuters.com+when:1d&hl=en-US&gl=US&ceid=US:en", "Google News"),
             ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance")]
    out, seen = [], set()
    for url, src in feeds:
        try:
            r = requests.get(url, headers=data._HTTP_HEADERS, timeout=12)
            if r.status_code == 200:
                for it in data._parse_feed(r.text, 2, src):
                    k = it["title"].lower()[:70]
                    if k not in seen:
                        seen.add(k)
                        out.append(it)
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda it: it["ts_utc"], reverse=True)
    return out[:n]


def macro_events(today: dt.date | None = None, days: int = 7) -> list[dict]:
    today = today or dt.date.today()
    return [e for e in config.MACRO_CALENDAR
            if 0 <= (dt.date.fromisoformat(e["date"]) - today).days <= days]


def build(held_news: dict[str, list[dict]], earnings: dict[str, dict], holdings: list[str],
          provider: str | None = None) -> dict:
    """Returns {text, overnight, watch, tickers_in_play, model, inputs}. Never raises."""
    moves = _moves()
    market = _market_headlines()
    held = [{"t": t, "src": h.get("source"), "h": h.get("title", "")[:110]}
            for t in holdings for h in (held_news.get(t) or [])[:3]]
    earn = {t: {"date": e.get("next_date"), "time_et": e.get("next_time_et")}
            for t, e in earnings.items() if e.get("reports_soon")}
    inputs = {"date": str(dt.date.today()), "markets": moves,
              "market_headlines": [{"src": h["source"], "ts": h["ts"], "h": h["title"][:110]} for h in market],
              "held_name_headlines": held[:45], "earnings_this_week": earn, "macro_calendar": macro_events(),
              "holdings": holdings}
    user = "DATA:\n" + json.dumps(inputs, separators=(",", ":"), default=str)

    def mock():
        mv = ", ".join(f"{k} {v.get('chg_pct', v.get('chg_bp'))}" for k, v in list(moves.items())[:5])
        ev = "; ".join(f"{e['date']} {e['event']}" for e in inputs["macro_calendar"]) or "no scheduled macro releases"
        return {"overnight": f"[mock] Overnight moves: {mv or 'n/a'}. {len(market)} market headlines collected.",
                "watch": f"[mock] Watch: {ev}. Earnings soon: {', '.join(earn) or 'none in the shortlist'}.",
                "tickers_in_play": list(earn)[:8]}

    model, out = "mock", None
    order = [] if (provider or config.LLM_PROVIDER) == "mock" else [config.MORNING_BRIEF_PROVIDER, provider or config.LLM_PROVIDER]
    for prov in order:
        try:
            out, model = LLMClient(prov).chat_json(SYSTEM, user, tier="smart", agent_name="Morning Brief", mock_fn=mock)
            if out.get("overnight"):
                break
        except Exception as e:  # noqa: BLE001
            model = f"{prov} failed ({str(e)[:80]})"
            out = None
    if not out:
        out = mock()
        model = model if model.startswith("mock") else f"{model}; fallback template"
    text = f"{str(out.get('overnight', '')).strip()}\n\n{str(out.get('watch', '')).strip()}"
    return {"text": text, "overnight": out.get("overnight", ""), "watch": out.get("watch", ""),
            "tickers_in_play": [str(t).upper() for t in (out.get("tickers_in_play") or [])][:8],
            "model": model, "n_headlines": len(market) + len(held), "macro_calendar": inputs["macro_calendar"],
            "earnings_this_week": earn}
