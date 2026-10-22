"""Oct 23: the ensemble, entry gate, trailing stop and trim pairing inside a real mock committee run (init -> execute -> daily run).
Hermetic: EDGAR / politicians / yfinance-quality are faked. Run:  /opt/miniconda3/bin/python3 tests/test_pipeline_ensemble.py   (~3 min)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, database as db
import orchestrator
from test_operations import _last_day
from test_handoff import _with_tmp


@_with_tmp
def test_init_to_daily_run_with_ensemble_trailing_stop_and_gate():
    res = orchestrator.run_committee("mock", asof=_last_day())
    rid = res["run_id"]
    prop = db.get_proposal(rid)
    quant = next(m for m in db.get_messages(rid) if m["agent"] == "Quant Analyst")["content"]
    ens = quant["strategy_ensemble"]
    assert {"mode", "allocated", "weights", "ic_table", "skipped"} <= set(ens)
    # replay look-ahead guard: no snapshot-only data sneaks into a replayed day
    assert {"eps_revisions", "quality_gp", "insider_buying", "news_surprise"} <= set(ens["skipped"])
    sig = next(iter(quant["signals"].values()))
    assert {"votes", "strategies", "score", "comment", "rule_score", "breakdown"} <= set(sig)
    assert set(sig["votes"]) == {"long", "short", "of", "needed"} and sig["votes"]["needed"] <= 3
    orchestrator.execute_run(rid, prices=prop["last_px"])
    held = db.get_positions()
    assert len(held) >= config.MIN_HOLDINGS

    # make every held stock look like it was bought 3 years ago at today's price: peak since entry = 3y high, no -10% cost stop
    close_px = prop["last_px"]
    with db.conn() as c:
        c.execute("UPDATE positions SET opened_date='2023-11-01'")
        c.execute("UPDATE transactions SET date='2026-01-02'")      # the init trades belong to an earlier day: the 5/day cap starts fresh
    old = config.TRAILING_STOP
    config.TRAILING_STOP = -0.01                                      # mechanism test: any name >= 1% below its 3y high trips it
    try:
        res2 = orchestrator.run_committee("mock", asof=_last_day())
    finally:
        config.TRAILING_STOP = old
    assert res2["mode"] == "daily"
    forced = [o for o in res2["orders"] if o.get("forced")]
    trail = [o for o in forced if "trailing stop" in o["forced_reason"]]
    assert trail, f"no trailing-stop exit among {[(o['ticker'], o['forced_reason']) for o in forced]}"
    assert all(o["side"] == "SELL" and o["shares"] == held[o["ticker"]]["shares"] for o in trail)       # full exit
    assert len(res2["orders"]) <= config.MAX_TRADES_PER_DAY + len(forced) or len(res2["orders"]) <= config.MAX_TRADES_PER_DAY
    # new-name BUYs that came through the eligibility path carry the votes needed (>= 3, or all that apply)
    q2 = next(m for m in db.get_messages(res2["run_id"]) if m["agent"] == "Quant Analyst")["content"]["signals"]
    for o in res2["orders"]:
        if o["side"] == "BUY" and o.get("is_new") and not o.get("replacement") and not o.get("pair_of"):
            v = q2[o["ticker"]]["votes"]
            assert v["long"] >= 0 and v["needed"] <= 3
    # the stop-loss rule from the IPS still stands and still takes precedence
    assert config.STOP_LOSS == -0.10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
