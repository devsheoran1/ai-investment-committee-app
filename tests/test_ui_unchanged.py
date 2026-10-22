"""Oct 23 HARD CONSTRAINT: nothing the class has seen may change (dashboard layout, tab names, expanders, agent roster,
IPS limits, report headings, trades.csv columns). tests/golden/ui_snapshot.json was produced by the pre-change code (git
b528a18) on tests/fixtures/ui_fixture.db; the current code must reproduce it exactly, with diagnostics on and off.
If you change the UI ON PURPOSE, regenerate the golden:
    cp tests/fixtures/ui_fixture.db /tmp/f.db && python tests/ui_snapshot.py --app app.py --db /tmp/f.db --out tests/golden/ui_snapshot.json
Run:  /opt/miniconda3/bin/python3 tests/test_ui_unchanged.py"""
import json, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _current():
    tmp = tempfile.mkdtemp()
    db, out = os.path.join(tmp, "f.db"), os.path.join(tmp, "snap.json")
    shutil.copy(os.path.join(ROOT, "tests", "fixtures", "ui_fixture.db"), db)
    env = {**os.environ, "PUBLISH": "0"}
    env.pop("FUND_DB_PATH", None); env.pop("FUND_TRADES_CSV", None)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "ui_snapshot.py"), "--app", os.path.join(ROOT, "app.py"),
                        "--db", db, "--out", out], capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(out))


GOLD = json.load(open(os.path.join(ROOT, "tests", "golden", "ui_snapshot.json")))
CUR = _current()


def test_app_renders_without_exceptions_in_both_diagnostic_modes():
    assert CUR["exceptions"] == [] and CUR["diag_exceptions"] == []


def test_tabs_buttons_expanders_toggles_metrics_are_identical():
    for k in ("tabs", "buttons", "expanders", "toggles", "selectboxes", "metrics", "headers", "diag_expanders"):
        assert CUR[k] == GOLD[k], f"{k} changed:\n{GOLD[k]}\n->\n{CUR[k]}"
    assert CUR["tabs"] == ["Decision Room", "Portfolio", "Performance", "Trades", "Committee"]


def test_element_tree_is_identical_with_diagnostics_off_and_on():
    assert CUR["main"] == GOLD["main"] and CUR["sidebar"] == GOLD["sidebar"]
    assert CUR["diag_main"] == GOLD["diag_main"]


def test_roster_limits_report_and_csv_contract_unchanged():
    g, c = GOLD["contract"], CUR["contract"]
    for k in g:
        assert c[k] == g[k], f"{k} changed:\n{g[k]}\n->\n{c[k]}"
    assert c["trades_csv_columns"][:5] == ["date", "hour_madrid", "ticker", "name", "side"]
    assert len(c["roster"]) == 9 and "Quant Analyst" in c["roster"] and "CIO" in c["roster"]
    assert len(c["report_headings"]) >= 5 and c["report_headings"][0].startswith("# Daily report")
    assert c["risk_limits"]["max_single_weight"] == 0.20 and c["risk_limits"]["max_sector_weight"] == 0.35
    assert c["stop_loss"] == -0.10 and c["max_trades"] == 5 and c["max_runs"] == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
