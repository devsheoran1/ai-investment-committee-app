"""Structural snapshot of the Streamlit dashboard (what the class sees): tabs, headers, expanders, buttons, toggles,
metric labels, table columns, element counts. Values are NOT compared (they change every day); labels and layout are.

    python tests/ui_snapshot.py --app /path/to/app.py --db tests/fixtures/ui_fixture.db --out snapshot.json
"""
import argparse, json, os, sys


def _walk(node, out, depth=0):
    kids = getattr(node, "children", None)
    if kids is None:
        return
    for k in (kids.values() if isinstance(kids, dict) else kids):
        t = type(k).__name__
        rec = {"d": depth, "t": t}
        for attr in ("label", "options"):
            v = getattr(k, attr, None)
            if v is not None and not callable(v):
                rec[attr] = list(v) if isinstance(v, (list, tuple)) else str(v)
        if t in ("Header", "Subheader", "Title"):
            rec["text"] = str(getattr(k, "value", ""))
        if t == "Dataframe":
            try:
                rec["columns"] = [str(c) for c in k.value.columns]
            except Exception:  # noqa: BLE001
                pass
        out.append(rec)
        _walk(k, out, depth + 1)


def snapshot(app: str, db: str) -> dict:
    os.environ.update({"FUND_DB_PATH": db, "PUBLISH": "0", "VIEWER_MODE": "0"})
    sys.path.insert(0, os.path.dirname(os.path.abspath(app)))
    os.chdir(os.path.dirname(os.path.abspath(app)))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.abspath(app), default_timeout=300)
    at.run()
    elems: list = []
    _walk(at.main, elems)
    side: list = []
    _walk(at.sidebar, side)
    diag: list = []
    for t in at.toggle:                                   # "Show diagnostics" opens the prompts / raw-output expanders
        if "diagnostics" in str(t.label).lower():
            t.set_value(True)
    at.run()
    _walk(at.main, diag)
    return {"diag_main": diag, "diag_expanders": [e.label for e in at.expander], "diag_exceptions": [str(e.value)[:200] for e in at.exception], **{"exceptions": [str(e.value)[:200] for e in at.exception],
            "tabs": [t.label for t in at.tabs], "buttons": [b.label for b in at.button],
            "expanders": [e.label for e in at.expander], "toggles": [t.label for t in at.toggle],
            "selectboxes": [(s.label, list(s.options)) for s in at.selectbox],
            "metrics": [m.label for m in at.metric], "headers": [h.value for h in at.header] + [h.value for h in at.subheader],
            "main": elems, "sidebar": side}}


def contract(db: str) -> dict:
    """Non-UI things the class has seen: agent roster and names, IPS limits, report headings, trades.csv columns."""
    os.environ.update({"FUND_DB_PATH": db, "PUBLISH": "0"})
    import re
    import config, database, report
    from agents import BRIEF_ROLES, ROSTER
    day = database.get_snapshots()["date"].iloc[-1] if len(database.get_snapshots()) else database.today()
    import tempfile
    report.REPORT_DIR = tempfile.mkdtemp()                 # never write into the repo's reports/
    rep = open(report.daily_report(str(day))).read()
    heads = [ln.strip() for ln in rep.splitlines() if ln.lstrip().startswith("#")]
    forum = report.forum_post(str(day))
    forum = forum if isinstance(forum, str) else str(forum)
    import tempfile
    csv_path = os.path.join(tempfile.mkdtemp(), "trades.csv")
    database.export_trades_csv(csv_path)
    cols = open(csv_path).readline().strip().split(",")
    return {"roster": [getattr(a, "name", str(a)) for a in ROSTER], "brief_roles": BRIEF_ROLES,
            "risk_limits": {k: v for k, v in config.RISK_LIMITS.items()}, "stop_loss": config.STOP_LOSS,
            "drift_tolerance": config.DRIFT_TOLERANCE, "max_trades": config.MAX_TRADES_PER_DAY, "max_runs": config.MAX_RUNS_PER_DAY,
            "holdings": [config.MIN_HOLDINGS, config.MAX_HOLDINGS], "target_holdings": list(config.TARGET_HOLDINGS),
            "dd": [config.DD_WARN, config.DD_HARD], "signal_weights": config.SIGNAL_WEIGHTS,
            "report_headings": heads, "forum_headings": [ln for ln in forum.splitlines() if ln.startswith(("#", "**"))][:12],
            "trades_csv_columns": cols}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    snap = snapshot(a.app, os.path.abspath(a.db))
    snap["contract"] = contract(os.path.abspath(a.db))
    json.dump(snap, open(os.path.abspath(a.out) if not os.path.isabs(a.out) else a.out, "w"), indent=1, default=str)
    print(f"{len(snap['main'])} main elements, tabs {snap['tabs']}, exceptions {snap['exceptions']}")
