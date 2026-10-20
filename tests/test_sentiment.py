"""Stage 2 (Oct 20): Sentiment produces real signal (IPS C4.1-C4.6, C5.2).
Run:  /opt/miniconda3/bin/python3 tests/test_sentiment.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analytics, config, data
from committee.sentiment_analyst import SentimentAnalyst

H = lambda title, fb=None: {"title": title, "source": "Reuters", "ts": "2026-10-20 15:00", "finbert": fb}


def test_no_news_is_na_not_zero():
    ctx = {"tickers": ["AAA", "BBB", "CCC"], "news": {"AAA": [H("AAA beats and raises guidance", 0.9)], "BBB": [], "CCC": [H("x")]}}
    out = SentimentAnalyst().validate({"sentiment": {"AAA": {"score": 0.4, "based_on": [0]}}}, ctx)["sentiment"]
    assert out["AAA"]["score"] == 0.4 and out["AAA"]["finbert"] == 0.9 and out["AAA"]["based_on"]
    assert out["BBB"]["score"] is None and out["BBB"]["key_driver"] == "no news"       # n/a, never 0.0
    assert out["CCC"]["score"] is None                                                 # LLM gave no answer -> no signal


def test_na_redistributes_weight_pro_rata():
    W = config.SIGNAL_WEIGHTS
    c = analytics.composite_scores(["AAA"], {"AAA": "Energy"}, {"AAA": "stock"},
                                   {"AAA": {"score": 0.5, "rating": "buy"}}, {"AAA": {"score": 0.8}},
                                   {"AAA": {"score": None}}, {"Energy": "neutral"}, "neutral")["AAA"]
    tot = W["quant"] + W["research"] + W["macro"]
    expect = (W["quant"] * 0.8 + W["research"] * 0.5) / tot
    assert abs(c["composite"] - round(expect, 3)) < 1e-3 and "sentiment" in c["na"]
    assert abs(sum(c["weights_used"].values()) - 1) < 1e-2 and "sentiment" not in c["weights_used"]


class _FakeLLM:
    provider = "groq"
    def __init__(self): self.calls = []
    def chat_json(self, system, user, tier="fast", agent_name=None, mock_fn=None):
        self.calls.append(tier)
        ticks = [f"T{i}" for i in range(10)]
        if tier == "fast":
            return {"sentiment": {t: {"score": 0.0} for t in ticks}}, "gpt-oss-20b"
        return {"sentiment": {t: {"score": 0.3 if i % 2 else -0.2} for i, t in enumerate(ticks)}}, "gpt-oss-120b"


def test_lazy_zero_scores_retry_on_senior_model():
    ticks = [f"T{i}" for i in range(10)]
    ctx = {"tickers": ticks, "news": {t: [H(f"{t} news")] for t in ticks}}
    llm = _FakeLLM()
    out, model = SentimentAnalyst().run(ctx, llm)
    assert llm.calls == ["fast", "smart"] and "retry" in model
    assert sum(1 for v in out["sentiment"].values() if v["score"] == 0.0) == 0


def test_feed_parser_keeps_source_and_madrid_time():
    rss = """<rss><channel><item><title>Phillips 66 locks out workers - Reuters</title>
    <pubDate>Mon, 19 Oct 2026 16:35:00 GMT</pubDate><source>Reuters</source><link>u</link></item></channel></rss>"""
    import datetime as dt
    items = data._parse_feed(rss, 3650, "Google News")
    assert items[0]["title"] == "Phillips 66 locks out workers" and items[0]["source"] == "Reuters"
    assert items[0]["ts"] == "2026-10-19 18:35"                                        # CEST = UTC+2
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>8-K - Current report</title>
    <updated>2026-10-19T08:00:00-04:00</updated><link href="https://sec.gov/x"/></entry></feed>"""
    items = data._parse_feed(atom, 3650, "SEC EDGAR 8-K")
    assert items[0]["source"] == "SEC EDGAR 8-K" and items[0]["ts"] == "2026-10-19 14:00" and items[0]["url"]


def test_relevance_filter_drops_unrelated_headlines():
    keys = data._name_keys("NVDA", "Nvidia")
    assert any(k in "nvidia stocks fall" for k in keys) and not any(k in "nuscale stock closed lower" for k in keys)


def test_intraday_recheck_triggers_sell_through_normal_path():
    import orchestrator, database as db, tempfile
    old_path = config.DB_PATH
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db"); db.init_db()     # never the live fund
    db.execute_trade(1, "AAA", "BUY", 10, 100.0, "seed")
    seen = {}
    orig_news, orig_run = data.get_news, orchestrator.run_committee
    orig_mock = SentimentAnalyst.mock
    try:
        data.get_news = lambda tickers, *a, **k: {t: [H("AAA under fraud probe, guidance withdrawn")] for t in tickers}
        SentimentAnalyst.mock = lambda self, ctx: {"sentiment": {"AAA": {"score": -0.9, "based_on": [0]}}, "market_mood": ""}
        orchestrator.run_committee = lambda *a, **k: seen.update(k) or {"status": "proposed", "orders": []}
        orchestrator.intraday_recheck("mock")
        assert "AAA" in seen["extra_forced"] and seen["_skip_run_limit"]
        SentimentAnalyst.mock = lambda self, ctx: {"sentiment": {"AAA": {"score": -0.3}}, "market_mood": ""}
        seen.clear()
        res = orchestrator.intraday_recheck("mock")
        assert res["status"] == "recheck_clear" and not seen
    finally:
        data.get_news, orchestrator.run_committee, SentimentAnalyst.mock = orig_news, orig_run, orig_mock
        config.DB_PATH = db._DB["path"] = old_path


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
