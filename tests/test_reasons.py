"""Stage 3 (Oct 20): per-trade plain-English reasons, names, Madrid time, stale-proposal guard (E1, E2, E7, D13).
Run:  /opt/miniconda3/bin/python3 tests/test_reasons.py"""
import os, sys, tempfile, csv, json, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, database as db
from committee.rationale_writer import TradeRationaleWriter, template

F = {"ticker": "PSX", "name": "Phillips 66", "sector": "Energy", "side": "BUY", "shares": 343, "notional": 99659,
     "weight_before": 0.026, "weight_after": 0.126, "why": "refiners lead on strong margins", "invalidation_px": 276.0}


def test_template_has_company_why_portfolio_and_exit():
    t = template(F)
    assert "Phillips 66" in t and "refiners" in t and "12.6%" in t and "$276.00" in t


def test_writer_rejects_text_naming_another_ticker_or_invented_price():
    ctx = {"facts": [F], "all_tickers": ["PSX", "FAS", "VLO"]}
    w = TradeRationaleWriter()
    out = w.validate({"rationales": {"PSX": "Bought Phillips 66; we also add FAS to lift beta. Exit below $276."}}, ctx)
    assert out["rationales"]["PSX"] == template(F) and out["fallback_used_for"] == ["PSX"]
    out = w.validate({"rationales": {"PSX": "Bought Phillips 66, a refiner. We exit below $250."}}, ctx)
    assert out["fallback_used_for"] == ["PSX"]
    good = "Bought Phillips 66, a US refiner, because refiners lead on margins. It lifts the fund's beta. We exit if it closes below $276."
    out = w.validate({"rationales": {"PSX": good}}, ctx)
    assert out["rationales"]["PSX"] == good and not out["fallback_used_for"]


def test_trades_csv_columns_and_names():
    old = (config.DB_PATH, config.TRADES_CSV)
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db"); config.TRADES_CSV = tempfile.mktemp(suffix=".csv")
    try:
        db.init_db()
        db.execute_trade(1, "PSX", "BUY", 10, 290.0, "Bought Phillips 66 ...", name="Phillips 66",
                         signals="composite 7.1/10 · Q +0.60 · S n/a", proposed_at=db._now())
        rows = list(csv.reader(open(db.export_trades_csv())))
        assert rows[0] == ["date", "hour_madrid", "ticker", "name", "side", "price", "shares", "total", "reason", "signals"]
        assert rows[1][3] == "Phillips 66" and "composite" in rows[1][9] and "composite" not in rows[1][8]
    finally:
        config.DB_PATH, config.TRADES_CSV = old; db._DB["path"] = config.DB_PATH


def test_label_and_madrid_et():
    assert data.label("PSX").startswith("PSX · ") and "Phillips" in data.label("PSX")
    assert data.label("TQQQ").startswith("TQQQ · ProShares")
    assert db.madrid_et("2026-10-20T15:34:00+02:00") == "2026-10-20 15:34 Madrid (09:34 ET)"
    assert db.madrid_et("2026-10-20T13:34:00+00:00") == "2026-10-20 15:34 Madrid (09:34 ET)"


def test_stale_proposal_is_refused():
    import orchestrator
    old = config.DB_PATH
    config.DB_PATH = db._DB["path"] = tempfile.mktemp(suffix=".db")
    try:
        db.init_db()
        rid = db.start_run("mock", "test", "daily")
        db.finish_run(rid, "proposed", "x", proposal={"mode": "daily", "orders": [], "last_px": {}, "benchmark_px": 1})
        with db.conn() as c:
            c.execute("UPDATE runs SET finished_at=? WHERE id=?", ((db.now() - dt.timedelta(hours=5)).isoformat(), rid))
        try:
            orchestrator.execute_run(rid)
            raise AssertionError("stale proposal executed")
        except RuntimeError as e:
            assert "old" in str(e) and db.get_runs().set_index("id").loc[rid, "status"] == "expired"
    finally:
        config.DB_PATH = db._DB["path"] = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
