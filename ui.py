"""
Reusable dashboard components (Stage 9). Institutional look: navy background #0B1F3A, panels #12294A, one accent blue
#3B82F6 + lighter blues, green/red only for P&L. Inter for text, Source Serif for big numbers. One Plotly template.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, PANEL, PANEL2, LINE = "#0B1F3A", "#12294A", "#183457", "#25456E"
ACCENT, BLUE2, BLUE3, BLUE4 = "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"
TEXT, MUTED = "#E6EDF7", "#8EA3C0"
GREEN, RED, AMBER = "#22C55E", "#EF4444", "#F59E0B"
SERIES = [ACCENT, BLUE3, BLUE2, BLUE4, "#1D4ED8", "#2563EB", "#7DD3FC", "#A5B4FC"]

AGENT_STYLE = {   # role colour, owner (IPS C0.2), initials
    "Morning Brief": ("#7DD3FC", "team", "MB"), "Macro Strategist": ("#60A5FA", "Rima", "MS"),
    "Equity Research Analyst": ("#93C5FD", "Marina", "ER"), "Quant Analyst": ("#3B82F6", "Jayesh", "QA"),
    "Sentiment Analyst": ("#A5B4FC", "Mer", "SA"), "Portfolio Manager": ("#2563EB", "Rima", "PM"),
    "Devil's Advocate": ("#F59E0B", "team", "DA"), "Risk Manager": ("#EF4444", "Dev", "RM"),
    "CIO": ("#BFDBFE", "Mer", "CIO"), "Trader": ("#22C55E", "Marina", "TR"),
    "Reflection Analyst": ("#7DD3FC", "team", "RF"), "Trade Rationale Writer": ("#93C5FD", "team", "TW"),
    "System": ("#8EA3C0", "Python", "PY"),
}

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
.avatar {{ width: 36px; height: 36px; border-radius: 50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size: 0.78rem; color: {NAVY}; flex-shrink: 0; }}
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
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: 10px 12px; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tpl = go.layout.Template()
    tpl.layout = go.Layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
                           font=dict(family="Inter, sans-serif", color=TEXT, size=12),
                           xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
                           yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
                           legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                           margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(bgcolor=PANEL2, font_color=TEXT))
    pio.templates["fund"] = tpl
    pio.templates.default = "fund"


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
<div class="meta">{esc(title)} · owner {esc(owner)} · {esc(model)}{lat}</div></div></div>
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
