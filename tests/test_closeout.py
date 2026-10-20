"""Stage 8: NYSE calendar, no phantom snapshots, HALT kill switch, prompt logging.
Run:  /opt/miniconda3/bin/python3 tests/test_closeout.py"""
import os, sys, tempfile, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, database as db, nyse_calendar as cal
from llm import LLMClient


def test_calendar():
    assert cal.is_trading_day("2026-10-22") and not cal.is_trading_day("2026-10-24")          # Thu yes, Sat no
    assert not cal.is_trading_day("2026-11-26") and cal.early_close("2026-11-27") == "13:00 ET"  # Thanksgiving / half day
    assert str(cal.previous_trading_day("2026-10-26")) == "2026-10-23"


def _tmp():
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db"); db.init_db()


def test_no_phantom_snapshot_on_weekend():
    old, real = config.DB_PATH, db.now
    try:
        _tmp()
        db.now = lambda: dt.datetime(2026, 10, 24, 22, 0, tzinfo=db.MADRID)                    # Saturday
        db.record_snapshot(None, {"SPY": 500.0}, 500.0)
        assert db.get_snapshots().empty
        db.now = lambda: dt.datetime(2026, 10, 23, 22, 0, tzinfo=db.MADRID)                    # Friday
        db.record_snapshot(None, {"SPY": 500.0}, 500.0)
        assert len(db.get_snapshots()) == 1
    finally:
        db.now = real; config.DB_PATH = db._DB["path"] = old


def test_halt_file_blocks_execution():
    import orchestrator
    halt = os.path.join(config.BASE_DIR, "HALT")
    assert not os.path.exists(halt), "remove the real HALT file before running tests"
    old = config.DB_PATH
    try:
        _tmp()
        open(halt, "w").close()
        try:
            orchestrator.execute_run(1)
            raise AssertionError("executed despite HALT")
        except RuntimeError as e:
            assert "HALT" in str(e)
    finally:
        os.remove(halt); config.DB_PATH = db._DB["path"] = old


def test_prompts_are_logged():
    old = config.DB_PATH
    try:
        _tmp()
        llm = LLMClient("mock")
        llm.chat_json("SYSTEM TEXT", "USER TEXT", mock_fn=lambda: {"ok": 1})
        rid = db.start_run("mock", "x", "daily")
        db.log_message(rid, "Macro Strategist", "t", "mock", "analysis", 0, {"ok": 1}, prompt=llm.last)
        with db.conn() as c:
            r = c.execute("SELECT system_prompt, user_prompt, raw_output FROM messages").fetchone()
        assert r["system_prompt"] == "SYSTEM TEXT" and r["user_prompt"] == "USER TEXT" and '"ok"' in r["raw_output"]
    finally:
        config.DB_PATH = db._DB["path"] = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
