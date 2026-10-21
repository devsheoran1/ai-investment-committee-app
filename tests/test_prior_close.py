"""Oct 21: official prior close booked at run start (NYSE calendar, America/New_York, Polygon + yfinance <= 0.5%).
Run:  /opt/miniconda3/bin/python3 tests/test_prior_close.py"""
import os, sys, tempfile, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PUBLISH"] = "0"
import pandas as pd
import config, database as db, nyse_calendar as cal, orchestrator as orc

ET, MAD = cal.ET, db.MADRID


def test_last_completed_session_dst():
    # Oct 30 2026: EU already on CET (Oct 25), US still on EDT (until Nov 1) -> 1h gap shift; 21:45 Madrid = 16:45 ET
    assert cal.last_completed_session(dt.datetime(2026, 10, 30, 21, 45, tzinfo=MAD)) == dt.date(2026, 10, 30)
    # normal weeks 21:45 Madrid = 15:45 ET -> close not yet, previous session
    assert cal.last_completed_session(dt.datetime(2026, 10, 22, 21, 45, tzinfo=MAD)) == dt.date(2026, 10, 21)
    assert cal.last_completed_session(dt.datetime(2026, 10, 22, 22, 0, tzinfo=MAD)) == dt.date(2026, 10, 22)
    # Monday morning -> Friday; day after Thanksgiving half-day closes 13:00 ET
    assert cal.last_completed_session(dt.datetime(2026, 10, 26, 14, 30, tzinfo=MAD)) == dt.date(2026, 10, 23)
    assert cal.last_completed_session(dt.datetime(2026, 11, 27, 13, 5, tzinfo=ET)) == dt.date(2026, 11, 27)
    assert cal.close_dt("2026-11-02").astimezone(MAD).hour == 22    # US back on EST: 16:00 ET = 22:00 Madrid


def _fund():
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db"); db.init_db()
    with db.conn() as c:
        c.execute("INSERT INTO positions VALUES ('AAPL', 100, 90.0, '2026-10-20')")
        c.execute("INSERT INTO transactions (date, hour, action, symbol, shares, price, total) "
                  "VALUES ('2026-10-21', '00:53:08', 'BUY', 'AAPL', 100, 90.0, 9000)")
        c.execute("UPDATE simulation SET cash_balance=1000")


def test_books_close_once_and_intraday_cannot_overwrite():
    old, real_now, real_lcs = config.DB_PATH, db.now, cal.last_completed_session
    try:
        _fund()
        cal.last_completed_session = lambda now=None: dt.date(2026, 10, 21)
        db.now = lambda: dt.datetime(2026, 10, 22, 14, 30, tzinfo=MAD)
        close = pd.DataFrame({"AAPL": [99.0, 100.0], "SPY": [499.0, 500.0]},
                             index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
        poly = lambda day: {"AAPL": 100.3, "SPY": 503.0}           # AAPL 0.3% ok, SPY 0.6% -> flagged
        assert orc.record_prior_close(close, emit=lambda *a: None, polygon=poly) == "2026-10-21"
        s = db.get_snapshots().set_index("date").loc["2026-10-21"]
        assert s["kind"] == "close" and abs(s["total_equity"] - (1000 + 100 * 100.3)) < 1e-6
        assert "gap" in s["price_sources"]                          # SPY flagged > 0.5%
        assert orc.record_prior_close(close, emit=lambda *a: None, polygon=poly) is None   # already booked
        db.now = lambda: dt.datetime(2026, 10, 21, 21, 45, tzinfo=MAD)
        db.record_snapshot(None, {"AAPL": 1.0}, 1.0)                # intraday mark on the same date: ignored
        assert db.get_snapshots().set_index("date").loc["2026-10-21", "kind"] == "close"
    finally:
        db.now, cal.last_completed_session = real_now, real_lcs; config.DB_PATH = db._DB["path"] = old


def test_skips_when_book_changed_after_close():
    old, real_lcs = config.DB_PATH, cal.last_completed_session
    try:
        _fund()
        cal.last_completed_session = lambda now=None: dt.date(2026, 10, 20)   # trade at 00:53 Madrid Oct 21 is after it
        close = pd.DataFrame({"AAPL": [99.0], "SPY": [499.0]}, index=pd.to_datetime(["2026-10-20"]))
        assert orc.record_prior_close(close, emit=lambda *a: None, polygon=lambda d: {}) is None
        assert db.get_snapshots().empty
    finally:
        cal.last_completed_session = real_lcs; config.DB_PATH = db._DB["path"] = old


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
