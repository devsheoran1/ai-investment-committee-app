"""Oct 21: price accuracy (prices.py). No network: every source is stubbed.
Run:  /opt/miniconda3/bin/python3 tests/test_prices.py"""
import os, sys, tempfile, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import config, database as db, prices

IDX = pd.bdate_range("2026-10-01", periods=5)


def _raw(tickers):
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], tickers])
    return pd.DataFrame(100.0, index=IDX, columns=cols)


def test_missing_ticker_triggers_fallback_chain():
    calls = []
    def fetch(tk, period):          # batch drops ABNB and ZZZ; single retry recovers ABNB only
        calls.append(tuple(tk))
        keep = [t for t in tk if t not in ("ABNB", "ZZZ")] if len(tk) > 1 else [t for t in tk if t == "ABNB"]
        return _raw(keep) if keep else pd.DataFrame()
    stooq = lambda t: pd.Series(50.0, index=IDX) if t == "ZZZ" else None
    f = prices.download_history(["SPY", "ABNB", "ZZZ", "QQQ"], "1mo", fetch=fetch, stooq=stooq, pause=0)
    assert ("ABNB",) in calls and f["close"]["ABNB"].iloc[-1] == 100 and f["close"]["ZZZ"].iloc[-1] == 50
    assert prices.QUALITY["coverage"] == (4, 4) and prices.QUALITY["sources"]["history"]["stooq"] == 1


def test_mismatch_blocks_order():
    out = prices.live_prices(["AAPL", "MSFT"], finnhub=lambda t: {"AAPL": 102.0, "MSFT": 400.0}[t],
                             yfin=lambda tk: {"AAPL": 100.0, "MSFT": 401.0})
    assert prices.blocked(out["AAPL"]).startswith(prices.UNVERIFIED)       # 2% gap
    assert prices.blocked(out["MSFT"]) is None and out["MSFT"]["px"] == 400.0
    single = prices.live_prices(["NVDA"], finnhub=lambda t: None, yfin=lambda tk: {"NVDA": 180.0})["NVDA"]
    assert single["flag"] == prices.SINGLE and not prices.blocked(single)


def test_stale_price_blocks_trading():
    out = prices.live_prices(["ABNB"], last_verified={"ABNB": 130.0}, finnhub=lambda t: None, yfin=lambda tk: {})
    assert out["ABNB"]["px"] == 130.0 and prices.blocked(out["ABNB"]) == prices.STALE
    assert "ABNB stale price" in prices.quality_line()


def test_nav_uses_verified_source():
    old, real = config.DB_PATH, db.now
    try:
        config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db"); db.init_db()
        db.now = lambda: dt.datetime(2026, 10, 22, 21, 45, tzinfo=db.MADRID)
        db.execute_trade(None, "AAPL", "BUY", 100, 100.0, "test")
        snap = prices.snapshot_prices(["AAPL", "SPY"], {"AAPL": 100.0, "SPY": 500.0}, "2026-10-22",
                                      polygon=lambda d: {"AAPL": 110.0, "SPY": 500.5})
        assert snap["AAPL"]["source"] == "polygon" and snap["AAPL"]["flag"]          # 10% gap logged, Polygon used
        px = {t: v["px"] for t, v in snap.items()}
        nav = db.record_snapshot(None, px, px["SPY"], price_sources={t: v["source"] for t, v in snap.items()})
        assert abs(nav - (config.STARTING_CAPITAL - 10_000 + 11_000)) < 1e-6
        assert "polygon" in db.get_snapshots()["price_sources"].iloc[-1]
    finally:
        db.now = real; config.DB_PATH = db._DB["path"] = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
