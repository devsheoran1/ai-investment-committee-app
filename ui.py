"""
Reusable dashboard components (Stage 9). Institutional look: navy background #1B3354, panels #234068, one soft accent blue
#6BA3EC + lighter blues, green/red only for P&L. Inter for text, Source Serif for big numbers. One Plotly template.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, PANEL, PANEL2, LINE = "#1B3354", "#234068", "#2B4C78", "#3A5F8E"   # Oct 21: lightened
ACCENT, BLUE2, BLUE3, BLUE4 = "#6BA3EC", "#8BB9F2", "#A9CBF6", "#CFE1FA"
TEXT, MUTED = "#F2F6FC", "#AFC2DB"
GRID = "rgba(175,194,219,0.12)"          # muted gridlines
GREEN, RED, AMBER = "#22C55E", "#EF4444", "#F59E0B"
SERIES = [ACCENT, BLUE3, BLUE2, BLUE4, "#1D4ED8", "#2563EB", "#7DD3FC", "#A5B4FC"]

AGENT_STYLE = {   # role colour, owner (IPS C0.2), initials
    "Morning Brief": ("#7DD3FC", "team", "MB"), "Macro Strategist": ("#60A5FA", "Rima", "MS"),
    "Equity Research Analyst": ("#93C5FD", "Marina", "ER"), "Quant Analyst": ("#3B82F6", "Jayesh", "QA"),
    "Sentiment Analyst": ("#A5B4FC", "Mer", "SA"), "Portfolio Manager": ("#2563EB", "Rima", "PM"),
    "Devil's Advocate": ("#F59E0B", "team", "DA"), "Risk Manager": ("#EF4444", "Dev", "RM"),
    "CIO": ("#BFDBFE", "Mer", "CIO"), "Trader": ("#22C55E", "Marina", "TR"),
    "Reflection Analyst": ("#7DD3FC", "team", "RF"), "Trade Rationale Writer": ("#93C5FD", "team", "TW"),
    "System": ("#AFC2DB", "Python", "PY"),
    "Screener": ("#8BB9F2", "Marina", "⏷"), "Regime Engine": ("#A9CBF6", "Rima", "◔"),
    "Model Lab": ("#6BA3EC", "Jayesh", "⚗"),
}
RULES_ENGINES = {"Screener", "Regime Engine", "Model Lab"}

CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stTabs {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {NAVY}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; color: {TEXT}; letter-spacing: -0.01em; }}
.kpi {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 14px; min-height: 92px; margin-bottom: 8px; }}
.kpi .lbl {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi .val {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.65rem; color: {TEXT}; line-height: 1.25; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.78rem; margin-top: 2px; }}
.pos {{ color: {GREEN} !important; }} .neg {{ color: {RED} !important; }} .warn {{ color: {AMBER} !important; }}
.badge {{ display:inline-block; white-space: nowrap; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em; }}
.chip {{ display:inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; margin-right: 6px; }}
.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
.card .head {{ display:flex; align-items:center; gap:10px; }}
.avatar {{ width: 36px; height: 36px; border-radius: 50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size: 0.85rem; color: {NAVY}; flex-shrink: 0; }}
.card .who {{ font-weight: 600; color: {TEXT}; }} .card .meta {{ color: {MUTED}; font-size: 0.75rem; }}
.card .msg {{ color: {TEXT}; margin-top: 8px; font-size: 0.92rem; line-height: 1.45; }}
.nums {{ margin-top: 6px; color: {BLUE3}; font-size: 0.8rem; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; }}
.cell {{ background: {PANEL2}; border-radius: 8px; padding: 8px 10px; border-left: 3px solid {GREEN}; }}
.cell.fail {{ border-left-color: {RED}; }} .cell.warnc {{ border-left-color: {AMBER}; }}
.cell .r {{ color: {MUTED}; font-size: 0.72rem; }} .cell .v {{ color: {TEXT}; font-size: 0.88rem; font-weight: 600; }}
.timeline {{ display:flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px 0; }}
.step {{ padding: 4px 10px; border-radius: 999px; background: {PANEL2}; color: {MUTED}; font-size: 0.75rem; border: 1px solid {LINE}; }}
.step.done {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
.empty {{ background: {PANEL}; border: 1px dashed {LINE}; border-radius: 12px; padding: 28px; text-align: center; color: {MUTED}; }}
.empty b {{ color: {TEXT}; }}
div[data-testid="stTabs"] button p {{ font-size: 0.95rem; font-weight: 600; }}
html, body, .stApp {{ font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased; }}
[data-testid="stDataFrame"], [data-testid="stMetricValue"], .kpi .val, .nums, .cell .v {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1, "cv11" 1; }}
.kpi {{ border-radius: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.18), 0 6px 18px -8px rgba(0,0,0,.35); }}
.kpi .val {{ font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: -0.02em; font-size: 1.6rem; }}
.card, .empty {{ border-radius: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.14); }}
/* chart cards: soft 1px border, 14px corners, subtle shadow, even padding and breathing room */
[class*="st-key-card_"] {{ background: {PANEL}; border: 1px solid {LINE}66; border-radius: 14px; padding: 14px 16px 8px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.18), 0 8px 24px -12px rgba(0,0,0,.45); margin-bottom: 18px; position: relative; }}
[class*="st-key-card_"] [data-testid="stPopover"] {{ position: absolute; top: 10px; right: 12px; z-index: 5; width: auto; }}
[class*="st-key-card_"] [data-testid="stPopover"] button {{ border: none; background: transparent; color: {MUTED}; min-height: 1.8rem;
  padding: 0 .5rem; font-size: 1.15rem; line-height: 1; }}
[class*="st-key-card_"] [data-testid="stPopover"] button:hover {{ background: {PANEL2}; color: {TEXT}; }}
[class*="st-key-card_"] [data-testid="stPopoverButton"] [aria-hidden="true"] {{ display: none; }}
[class*="st-key-card_"] [data-testid="stPopoverButton"] p {{ font-size: 1.25rem; margin: 0; }}
div[data-testid="stExpander"] {{ border-radius: 12px; border-color: {LINE}88; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: 10px 12px; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tpl = go.layout.Template()
    tpl.layout = go.Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
                           font=dict(family="Inter, sans-serif", color=TEXT, size=12),
                           xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=False, ticks="", tickfont=dict(color=MUTED, size=11)),
                           yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=False, ticks="", tickfont=dict(color=MUTED, size=11)),
                           title=dict(font=dict(size=14, color=TEXT), x=0.0, xanchor="left"),
                           legend=dict(orientation="h", y=1.17, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                       font=dict(size=11, color=MUTED)),
                           margin=dict(l=10, r=10, t=44, b=10), hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT, bordercolor=LINE))
    pio.templates["fund"] = tpl
    pio.templates.default = "fund"


# One number style everywhere: $1,002,357 · +0.24% · at most 2 decimals
def usd(x: float, signed: bool = False) -> str:
    x = round(float(x))                       # a value that rounds to $0 carries no sign ("−$0")
    sign = ("+" if x >= 0 else "−") if signed else ("−" if x < 0 else "")
    return f"{sign}${abs(x):,.0f}"


def pct(x: float, signed: bool = True, d: int = 2) -> str:
    import valuation
    return valuation.pct(x, d, signed)


def esc(x) -> str:
    return html.escape(str(x))


def kpi(col, label: str, value: str, sub: str = "", tone: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="lbl">{esc(label)}</div><div class="val {tone}">{esc(value)}</div>'
                 f'<div class="sub">{esc(sub)}</div></div>', unsafe_allow_html=True)


def tone(x: float | None) -> str:
    return "" if x is None or abs(x) < 1e-12 else ("pos" if x > 0 else "neg")


REGIME_COLORS = {"risk_on": (GREEN, "RISK-ON"), "neutral": (BLUE2, "NEUTRAL"), "risk_off": (RED, "RISK-OFF")}


def regime_badge(regime: str | None) -> str:
    c, t = REGIME_COLORS.get(regime or "", (MUTED, "NO RUN YET"))
    return f'<span class="badge" style="background:{c}22;color:{c};border:1px solid {c}">{t}</span>'


STANCE_COLORS = {"bullish": GREEN, "approve": GREEN, "proceed": GREEN, "bearish": RED, "reject": RED, "drop": RED,
                 "neutral": BLUE2, "downsize": AMBER, "info": MUTED}


def chip(text: str) -> str:
    c = STANCE_COLORS.get(text.lower(), BLUE2)
    return f'<span class="chip" style="background:{c}22;color:{c}">{esc(text.upper())}</span>'


def agent_card(agent: str, title: str, model: str, latency: float | None, stance: str, numbers: list[str], message: str) -> None:
    color, owner, ini = AGENT_STYLE.get(agent, (BLUE2, "team", agent[:2].upper()))
    lat = f" · {latency:.1f}s" if latency else ""
    nums = " · ".join(esc(n) for n in numbers if n)
    st.markdown(f'''<div class="card"><div class="head"><div class="avatar" style="background:{color}">{esc(ini)}</div>
<div><div class="who">{esc(agent)} {chip(stance) if stance else ""}</div>
<div class="meta">{esc(title)} · owner {esc(owner)} · {esc("Rules engine" if agent in RULES_ENGINES else model)}{lat}</div></div></div>
{f'<div class="nums">{nums}</div>' if nums else ''}<div class="msg">{esc(message)}</div></div>''', unsafe_allow_html=True)


def limit_grid(checks: list[dict]) -> None:
    cells = []
    for c in checks:
        cls = "" if c["status"] == "pass" else ("fail" if c["kind"] in ("hard", "soft") else "warnc")
        mark = "✓" if c["status"] == "pass" else "✗"
        cells.append(f'<div class="cell {cls}"><div class="r">{esc(c["rule"])} · {esc(c["kind"])}</div>'
                     f'<div class="v">{mark} {esc(c["value"])} <span style="color:{MUTED};font-weight:400">(limit {esc(c["limit"])})</span></div></div>')
    st.markdown(f'<div class="grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def timeline(steps: list[str], done: int) -> str:
    return '<div class="timeline">' + "".join(
        f'<span class="step {"done" if i < done else ""}">{esc(s)}</span>' for i, s in enumerate(steps)) + "</div>"


def empty(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><b>{esc(title)}</b><br>{esc(body)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chart cards: toolbar hidden, a "⋮" menu (download PNG / CSV / full screen) top-right of each card
# --------------------------------------------------------------------------- #
_PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False}
_n = {"i": 0}


def _title(fig) -> str:
    t = getattr(fig.layout.title, "text", None) or "chart"
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(t).lower()).strip("_")[:48] or "chart"


def fig_to_df(fig):
    """The numbers behind a figure as one tidy table (trace, x, y / label, value / z) for the CSV download."""
    import pandas as pd
    frames = []
    for i, tr in enumerate(fig.data):
        nm = getattr(tr, "name", None) or f"trace{i + 1}"
        d = {}
        for k in ("x", "y", "labels", "values", "parents", "text"):
            v = getattr(tr, k, None)
            if v is not None and not isinstance(v, str):
                try:
                    d[k] = list(v)
                except TypeError:
                    pass
        z = getattr(tr, "z", None)
        if z is not None:
            try:
                zz = pd.DataFrame(z, index=getattr(tr, "y", None), columns=getattr(tr, "x", None))
                frames.append(zz.reset_index().assign(trace=nm))
                continue
            except Exception:  # noqa: BLE001
                pass
        if not d and getattr(tr, "value", None) is not None:
            d = {"value": [tr.value]}
        if d:
            n = max(len(v) for v in d.values())
            frames.append(pd.DataFrame({k: v + [None] * (n - len(v)) for k, v in d.items()}).assign(trace=nm))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _png(fig) -> bytes:
    try:
        return fig.to_image(format="png", width=1200, height=int(fig.layout.height or 420), scale=2)
    except Exception:  # noqa: BLE001  kaleido/Chrome missing: say so instead of failing the page
        return b""


def chart(host, fig, name: str | None = None) -> None:
    """Draw `fig` inside a styled card on `host` (a column or st). Same figure, same numbers; only the chrome changes."""
    _n["i"] += 1
    key = f"card_{_n['i']}"
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")      # the card is the background
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    name = name or _title(fig)
    with host.container(key=key):
        with st.popover("⋮", help="Chart options"):
            # callables: the PNG / CSV are only built when the button is clicked, not on every rerun
            st.download_button("Download PNG", data=lambda: _png(fig), file_name=f"{name}.png", mime="image/png",
                               key=f"png_{_n['i']}", width="stretch")
            st.download_button("Download CSV", data=lambda: fig_to_df(fig).to_csv(index=False), file_name=f"{name}.csv",
                               mime="text/csv", key=f"csv_{_n['i']}", width="stretch")
            if st.button("Full screen", key=f"fs_{_n['i']}", width="stretch"):
                _fullscreen(fig, name)
        st.plotly_chart(fig, width="stretch", config=_PLOT_CONFIG, theme=None, key=f"plot_{_n['i']}")


@st.dialog("Chart", width="large")
def _fullscreen(fig, name: str) -> None:
    st.plotly_chart(fig.update_layout(height=640), width="stretch", config=_PLOT_CONFIG, theme=None)
