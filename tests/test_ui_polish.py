"""Oct 23 UI polish: plain-English Model Lab wording, chart-card helpers, number formats, report wording.
Run:  /opt/miniconda3/bin/python3 tests/test_ui_polish.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotly.graph_objects as go
import lab_summary, ui


def _race(weights=None, ic=None):
    ic = ic or {}
    sb = {"lstm": {"hit": .50, "ic": -.07, "ic_t": -1.1, "always_up": .57}, "kronos": {"hit": .50, "ic": .054, "ic_t": 1.3, "always_up": .55},
          "lgbm": {"hit": .53, "ic": .036, "ic_t": .6, "always_up": .57}, "naive": {"hit": .51, "ic": .1, "ic_t": 3.0}}
    for m, v in ic.items():
        sb[m].update(v)
    return {"scoreboard": sb, "weights": weights or {"lstm": 0.0, "kronos": 0.0, "lgbm": 0.0}}


def test_no_model_trusted_says_so_and_names_slight_skill_only_when_t_is_at_least_one():
    s = lab_summary.plain_summary(_race())
    assert s.startswith("No model beat the simple 'stocks usually go up' guess (56%) this week, so Quant used price signals only.")
    assert "Kronos showed slight ranking skill but not enough." in s and "LightGBM" not in s and "naive" not in s.lower()
    assert "None showed reliable ranking skill" in lab_summary.plain_summary(_race(ic={"kronos": {"ic_t": 0.2}}))


def test_trusted_models_are_named_with_their_weights():
    s = lab_summary.plain_summary(_race({"lstm": 0.0, "kronos": 0.25, "lgbm": 0.5}))
    assert s.startswith("Kronos and LightGBM beat") and "Kronos 25%" in s and "LightGBM 50%" in s and "blended them in" in s
    assert "blended it in" in lab_summary.plain_summary(_race({"lgbm": 0.25}))


def test_missing_scoreboard_and_how_to_read_cover_the_terms():
    assert "price signals only" in lab_summary.plain_summary(None) and "price signals only" in lab_summary.plain_summary({})
    for term in ("Hit rate", "Always-up baseline", "IC", "**t**", "C3.4", "52%", "55%"):
        assert term in lab_summary.HOW_TO_READ, term


def test_same_wording_in_quant_comments_and_daily_report():
    src = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "committee", "quant_analyst.py")).read()
    for phrase in ("always-up", "above 55% = 50%", "t = IC divided by its noise"):
        assert phrase in src
    import report
    assert report.lab_summary is lab_summary


def test_number_formats():
    assert ui.usd(1002357.4) == "$1,002,357" and ui.usd(-6046, True) == "−$6,046" and ui.usd(1759, True) == "+$1,759"
    assert ui.pct(0.0024) == "+0.24%" and ui.pct(-0.0061) == "-0.61%" and ui.pct(0.5, signed=False, d=1) == "50.0%"


def test_chart_helpers_csv_png_and_toolbar_hidden():
    f = go.Figure([go.Bar(name="w", x=["A", "B"], y=[1, 2]), go.Scatter(name="s", x=["A", "B"], y=[3, 4])]).update_layout(title="Market-cap mix (% of NAV)")
    df = ui.fig_to_df(f)
    assert list(df["trace"].unique()) == ["w", "s"] and list(df["y"]) == [1, 2, 3, 4]
    assert ui._title(f) == "market_cap_mix_of_nav"
    assert ui._PLOT_CONFIG["displayModeBar"] is False
    png = ui._png(f)
    assert png[:4] == b"\x89PNG" or png == b""                      # PNG when kaleido + Chrome exist, never an exception
    hm = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], x=["a", "b"], y=["c", "d"]))
    assert ui.fig_to_df(hm).shape[0] == 2
    assert "grid" not in ui.CSS or ui.GRID.startswith("rgba")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
