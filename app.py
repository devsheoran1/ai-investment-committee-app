"""
AI Investment Committee — dashboard (Brief §3/§8.3, IPS E6/E7, Stage 9 redesign).   Run:  streamlit run app.py

Header: KPI cards + positions table (always visible).
Tabs:   Decision Room (run, live timeline, meeting transcript, execute) · Portfolio · Performance · Trades · Committee.
Brief §8.3 mapping: Decision Room = Trading Floor (Run Agents -> reasoning -> Execute Trades); Portfolio; Performance.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import config
import data
import database as db
import lab_summary
import reflection
import report
import tracker
import ui
import valuation

# Viewer mode (Streamlit Cloud): read-only, never imports the heavy model/LLM stack.
VIEWER = os.getenv("VIEWER_MODE") == "1"
if not VIEWER:
    try:
        from orchestrator import RunLimitError, execute_run, intraday_recheck, run_committee, snapshot_only
    except ImportError:
        VIEWER = True
try:
    from agents import ROSTER
except ImportError:
    ROSTER = []

def publish() -> None:
    """Push finance.db / trades.csv / reports/ so the cloud viewer updates (live fund only)."""
    if not REPLAY and os.getenv("PUBLISH", "1") == "1" and not os.getenv("FUND_DB_PATH"):
        import subprocess
        subprocess.Popen(["bash", os.path.join(config.BASE_DIR, "publish.sh")], cwd=config.BASE_DIR)


st.set_page_config(page_title="AI Investment Committee", page_icon="🏛️", layout="wide")
ui.inject()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner="Loading S&P 500 prices…")
def load_market():
    u = data.get_universe()
    md = data.get_market_data(u["symbol"].tolist())
    import prices as px_src
    return (u, md["close"], md["volume"], data.DATA_SOURCE["prices"], data.DATA_SOURCE["universe"], data.is_synthetic(),
            dict(px_src.QUALITY))


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(tickers: tuple, sectors: tuple):
    try:
        return data.get_fundamentals(list(tickers), dict(sectors))
    except Exception:  # noqa: BLE001
        return {}


universe, close, volume, source, uni_source, synthetic, quality = load_market()
db.use_synthetic(synthetic)
sector_of = dict(zip(universe["symbol"], universe["sector"]))
kind_of = dict(zip(universe["symbol"], universe["asset_type"]))
last_px = {t: float(close[t].dropna().iloc[-1]) for t in close.columns if close[t].notna().any()}

with st.sidebar:
    st.markdown("### 🏛️ AI Investment Committee")
    fund_view = st.radio("Fund", ["Live", "Replay (20 days)"], horizontal=True,
                         help="Replay = finance_replay.db from `python orchestrator.py --replay 20`. Read-only here.")
REPLAY = fund_view.startswith("Replay")
if REPLAY:
    db._DB["path"] = config.REPLAY["db"]
    config.TRADES_CSV = config.REPLAY["trades_csv"]
db.init_db()

positions = db.get_positions()
snaps = db.get_snapshots()
trades = db.get_trades()
marks = valuation.mark(positions, close, trades, snaps)          # never a close dated before the fill; same date as SPY
last_px.update(marks.px)
weights, nav = db.current_weights(last_px)
cash = db.get_cash()
runs = db.get_runs()
halted = os.path.exists(os.path.join(config.BASE_DIR, "HALT"))


def latest_regime() -> str | None:
    with db.conn() as c:
        r = c.execute("SELECT effective FROM regime_state ORDER BY date DESC LIMIT 1").fetchone()
    return r[0] if r else None


def messages_of(run_id: int) -> list[dict]:
    return db.get_messages(int(run_id))


def last_message(agent: str, key: str | None = None):
    for rid in runs["id"].tolist() if not runs.empty else []:
        for m in reversed(messages_of(rid)):
            if m["agent"] == agent and (key is None or key in m["content"]):
                return m
    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
import nyse_calendar
_lbl = nyse_calendar.session_labels(nyse_calendar.last_completed_session() if not nyse_calendar.is_trading_day(db.today())
                                    else db.today())       # Madrid hours follow the NY clock (DST gaps differ US/EU)
with st.sidebar:
    st.caption(f"{'REPLAY fund (read-only)' if REPLAY else 'Live fund'} · {db.current_db_name()}")
    st.markdown(f"**{db.today()}** · Madrid {db.now():%H:%M}")
    st.caption(f"Last close in data: {close.index[-1].date()} · {source}")
    providers = [p for p in config.PROVIDERS if p != "gemini"]   # Gemini = Morning Brief only, never the agents
    provider = st.selectbox("AI brain", providers, index=providers.index(config.LLM_PROVIDER)
                            if config.LLM_PROVIDER in providers else 0,
                            help="groq = free cloud (default). mock = rule-based, no API calls.")
    c1, c2 = st.columns(2)
    c1.metric("Runs today", f"{db.runs_today()}/{config.MAX_RUNS_PER_DAY}")
    n_today = db.trades_today()
    c2.metric("Trades (init day)" if n_today > config.MAX_TRADES_PER_DAY else "Trades today",
              f"{n_today}" if n_today > config.MAX_TRADES_PER_DAY else f"{n_today}/{config.MAX_TRADES_PER_DAY}")
    if VIEWER:
        _last = runs["finished_at"].dropna().max() if len(runs) and "finished_at" in runs else None
        st.info(f"Viewer: updated after each laptop run, last update {db.madrid_et(_last) if _last else 'n/a'}")
    if not REPLAY and not VIEWER:
        if st.button("📅 Intraday mark (not the close)", width="stretch",
                     help=f"{_lbl['mark']} Madrid = {_lbl['mark_et']} ET, before the US close. Reprices holdings, no agents, no trades. "
                          "The official close is booked automatically at the start of the next committee run."):
            load_market.clear()
            st.toast(f"Intraday mark saved · NAV ${snapshot_only():,.0f}")
            publish()
            st.rerun()
        if positions and st.button("📰 Intraday news re-check", width="stretch",
                                   help="IPS C4.5, uses a run slot: held names only; <= -0.8 triggers a sell proposal."):
            with st.spinner("Re-checking held names…"):
                try:
                    res = intraday_recheck(provider)
                    st.session_state["selected_run"] = res.get("run_id")
                except RunLimitError as e:
                    st.error(str(e))
            st.rerun()
        with st.expander("Start new simulation"):
            st.caption(f"Resets {db.current_db_name()} to ${config.STARTING_CAPITAL:,.0f} cash. Cannot be undone.")
            if st.checkbox("I understand this wipes all trades and minutes") and st.button("Reset database", type="primary"):
                db.init_db(reset=True)
                st.rerun()
    if halted:
        st.error("⛔ HALT file present: execution disabled.")
    st.caption(f"Universe: {uni_source} + {len(config.LEVERAGED_ETFS)} leveraged / {len(config.INVERSE_ETFS)} inverse ETFs")

# --------------------------------------------------------------------------- #
# Header: KPI cards + positions table
# --------------------------------------------------------------------------- #
st.markdown(f"<h2 style='margin-bottom:0'>AI Investment Committee</h2>"
            f"<div style='color:{ui.MUTED};margin-bottom:12px'>{'Replay fund · ' if REPLAY else ''}"
            f"autonomous paper-trading fund · S&P 500 + leveraged ETFs · benchmark SPY</div>", unsafe_allow_html=True)
if synthetic:
    st.warning("Yahoo Finance unreachable and nothing cached: this is the synthetic sandbox fund.")

# Oct 21: Data quality box. Held names with no price today are valued at the last verified close and can't trade.
import prices as px_src
_stale = {t: px_src.STALE for t in db.get_positions() if t not in last_px} | marks.stale
px_src.QUALITY.update({**quality, "flags": {**quality.get("flags", {}), **_stale}})
_quality_line = px_src.quality_line()   # shown in a collapsed expander at the bottom of the page

ret = nav / config.STARTING_CAPITAL - 1
spy_ret = marks.spy_return                                      # inception SPY -> SPY at the same mark date as the fund
prev = snaps[snaps["date"] < db.today()]["total_equity"] if len(snaps) else pd.Series(dtype=float)
day_pnl = nav - (prev.iloc[-1] if len(prev) else config.STARTING_CAPITAL)
eq = snaps["total_equity"] if len(snaps) else pd.Series([config.STARTING_CAPITAL])
mdd = float((eq / eq.cummax().clip(lower=config.STARTING_CAPITAL) - 1).min()) if len(eq) else 0.0
cash_pct = cash / nav if nav else 1.0
k = st.columns(4) + st.columns(4)
ui.kpi(k[0], "Portfolio value", ui.usd(nav), f"{ui.usd(nav - config.STARTING_CAPITAL, True)} since inception", ui.tone(round(nav - config.STARTING_CAPITAL)))
ui.kpi(k[1], "Cash available", ui.usd(cash), f"{cash_pct:.1%} of NAV · limit {config.RISK_LIMITS['max_cash_weight']:.0%}",
       "neg" if positions and cash_pct > config.RISK_LIMITS["max_cash_weight"] else "")
alpha = None if spy_ret is None else int(round((ret - spy_ret) * 1e4))
ui.kpi(k[2], "Return vs SPY", ui.pct(ret), f"SPY {ui.pct(spy_ret)} · alpha {alpha:+d} bps" if alpha is not None else "no snapshot yet", ui.tone(round(ret, 4)))
ui.kpi(k[3], "Today's P&L", ui.usd(day_pnl, True), ui.pct(day_pnl / nav) if nav else "", ui.tone(round(day_pnl)))
ui.kpi(k[4], "Max drawdown", ui.pct(mdd, signed=False), "C11: -7% warn · -10% hard", "neg" if mdd <= config.DD_WARN else "")
ui.kpi(k[5], "Trades today", f"{n_today}/{config.MAX_TRADES_PER_DAY}" if n_today <= config.MAX_TRADES_PER_DAY else f"{n_today} (init)",
       f"{len(trades)} trades in total")
k[6].markdown(f'<div class="kpi"><div class="lbl">Regime (C1.3)</div><div style="margin-top:10px">{ui.regime_badge(latest_regime())}</div>'
              f'<div class="sub">rules in Python, Macro explains</div></div>', unsafe_allow_html=True)
last_upd = runs.iloc[0]["started_at"] if not runs.empty else None
ui.kpi(k[7], "Last update", db.madrid_et(last_upd)[5:16].replace("-", "/", 1) if last_upd else "—",
       "Madrid time" if last_upd else "no run yet")

if marks.stale:
    st.caption(f"Stale mark: {', '.join(sorted(marks.stale))} valued at the fill price (no close on or after the fill session yet).")
st.write("")
pos_df = pd.DataFrame()
if not positions:
    ui.empty("No positions yet", "The live fund starts on initialisation day: open the Decision Room and run the committee "
             "(10-20 names, fully invested, the 5-trade cap applies from day 2).")
else:
    held_stocks = tuple(sorted(t for t in positions if kind_of.get(t) == "stock"))
    fund = load_fundamentals(held_stocks, tuple((t, sector_of.get(t, "")) for t in held_stocks))

    def cap_bucket(t):
        if kind_of.get(t) != "stock":
            return "ETF"
        mc = (fund.get(t) or {}).get("mkt_cap_bn")
        return "–" if mc is None else "Mega" if mc > 200 else "Large" if mc > 10 else "Mid" if mc > 2 else "Small"
    rows = []
    trade_name = ({s: n for s, n in zip(trades["symbol"], trades["name"]) if isinstance(n, str) and n}
                  if len(trades) and "name" in trades.columns else {})
    for t, p in positions.items():
        px = last_px.get(t, p["avg_cost"])
        mv = p["shares"] * px
        inv = db.latest_buy_reflection(t) or {}
        spark = close[t].dropna().tail(20).round(2).tolist() if t in close.columns else []
        rows.append({"Name": data.label(t, trade_name.get(t)), "Sector": sector_of.get(t, "?"), "Cap": cap_bucket(t), "Shares": p["shares"],
                     "Weight": 100 * mv / nav, "Cost": p["avg_cost"], "Total cost": p["shares"] * p["avg_cost"],
                     "Market value": mv, "Price": px, "P&L $": round(p["shares"] * (px - p["avg_cost"])) + 0.0,
                     "P&L %": round(100 * (px / p["avg_cost"] - 1), 1) + 0.0, "20 days": spark, "Exit below": inv.get("invalidation_px")})
    pos_df = pd.DataFrame(rows).sort_values("Weight", ascending=False)
    st.dataframe(pos_df, hide_index=True, width="stretch", height=min(42 + 35 * len(pos_df), 420), column_config={
        "Weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=20),
        "Cost": st.column_config.NumberColumn(format="$%.2f"),
        "Total cost": st.column_config.NumberColumn(format="$%,.0f"), "Market value": st.column_config.NumberColumn(format="$%,.0f"), "Price": st.column_config.NumberColumn(format="$%.2f"),
        "P&L $": st.column_config.NumberColumn(format="$%+,.0f"), "P&L %": st.column_config.NumberColumn(format="%+.1f%%"),
        "20 days": st.column_config.LineChartColumn("20 days"), "Exit below": st.column_config.NumberColumn(format="$%.2f")})

tab_room, tab_pf, tab_perf, tab_tr, tab_com = st.tabs(["Decision Room", "Portfolio", "Performance", "Trades", "Committee"])

# --------------------------------------------------------------------------- #
# Decision Room
# --------------------------------------------------------------------------- #
STEPS = ["Morning Brief", "Macro", "Research / Quant / Sentiment", "PM", "Devil's Advocate", "Risk", "CIO", "Trader"]
STAGE_TO_STEP = {"data": 0, "screen": 0, "brief": 0, "model": 1, "reflection": 1, "analysis": 2, "proposal": 3,
                 "devils_advocate": 4, "proposal_answers_da": 4, "risk_review": 5, "decision": 6, "done": 8}


RULES_CARDS = {"Screening funnel": ("Screener", "Screening funnel (IPS Part B)"),
               "Regime rules": ("Regime Engine", "Regime rules (IPS C1.3)"),
               "Model horse race": ("Model Lab", "Model horse race + LSTM model card (C3.4)"),
               "Quant desk": ("Model Lab", "Model horse race + LSTM model card (C3.4)")}


def rules_cards(msgs: list[dict]) -> list[dict]:
    """Display only: rename the Python "System" steps and merge horse race + LSTM card into one Model Lab card."""
    out, lab = [], None
    for m in msgs:
        key = next((k for k in RULES_CARDS if m["agent"] == "System" and m["title"].startswith(k)), None)
        if not key:
            out.append(m)
            continue
        name, title = RULES_CARDS[key]
        if name != "Model Lab":
            out.append({**m, "agent": name, "title": title})
            continue
        part = "race" if key == "Model horse race" else "lstm"
        if lab is None:
            lab = {**m, "agent": name, "title": title, "content": {}, "system_prompt": None}
            out.append(lab)
        lab["content"][part] = m["content"]
    return out


def model_lab_line(c: dict) -> str:
    race = c.get("race") or {}
    w = {k: v for k, v in (race.get("weights") or {}).items() if v > 0}
    if not race:
        return "Horse race skipped: LSTM only." if c.get("lstm") else "No model output this run."
    if not w:
        return "No model beat the always-up baseline → technicals only."
    return "Models trusted this run: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()) + " (rest weight 0)."


def summarize(m: dict) -> tuple[str, list[str], str]:
    a, c = m["agent"], m["content"]

    def avg(d, key="score"):
        v = [x.get(key) for x in d.values() if isinstance(x, dict) and x.get(key) is not None]
        return float(np.mean(v)) if v else 0.0

    def stance_of(x):
        return "bullish" if x > 0.1 else "bearish" if x < -0.1 else "neutral"

    if a == "Morning Brief":
        return "info", [f"tickers in play: {', '.join(c.get('tickers_in_play') or []) or '—'}"], c.get("text", "")
    if a == "Macro Strategist":
        st_ = {"risk_on": "bullish", "risk_off": "bearish"}.get(c.get("regime"), "neutral")
        ow = [s for s, v in (c.get("sector_tilts") or {}).items() if v == "overweight"]
        return st_, [f"regime {c.get('regime')} ({c.get('regime_note', '')})", f"leveraged room {c.get('max_leveraged', 0):.0%}",
                     f"overweight {', '.join(ow)}"], c.get("rationale", "")
    if a == "Equity Research Analyst":
        r = c.get("ratings") or {}
        top = (r.get(c.get("top_pick") or "", {}) or {}).get("thesis", "")
        return stance_of(avg(r)), [f"top pick {c.get('top_pick')}", f"least {c.get('least_preferred')}",
                                   f"{sum(1 for v in r.values() if v.get('rating') == 'insufficient_data')} insufficient"], top
    if a == "Quant Analyst":
        s = c.get("signals") or {}
        mw = c.get("model_weights") or {}
        return stance_of(avg(s)), [f"model weights {', '.join(f'{k} {v:.0%}' for k, v in mw.items())}",
                                   f"fallback: {c['quant_fallback']}" if c.get("quant_fallback") else ""], \
            c.get("model_scoreboard") or c.get("note", "")
    if a == "Sentiment Analyst":
        s = c.get("sentiment") or {}
        na = sum(1 for v in s.values() if v.get("score") is None)
        return stance_of(avg(s)), [f"{len(s) - na} names scored · {na} n/a (no news)"], c.get("market_mood", "")
    if a == "Portfolio Manager":
        return "info", [f"exposure {c.get('equity_exposure', 0):.0%}", m["stage"].replace("_", " ")], \
            c.get("rationale", "") + (f" Response to Risk: {c['response_to_risk']}" if c.get("response_to_risk") else "")
    if a == "Devil's Advocate":
        v = [o["verdict"] for o in (c.get("objections") or {}).values()]
        worst = "drop" if "drop" in v else "downsize" if "downsize" in v else "proceed"
        return worst, [f"{len(v)} ideas challenged"], " · ".join(f"{t}: {o['bear_case']}" for t, o in (c.get("objections") or {}).items())
    if a == "Risk Manager":
        return c.get("decision", ""), [f"{len(c.get('hard_breaches') or [])} hard breaches", f"{len(c.get('soft_rejects') or [])} soft",
                                       f"{len(c.get('warnings') or [])} warnings", c.get("override") or ""], c.get("rationale", "")
    if a == "CIO":
        return c.get("decision", ""), c.get("key_risks") or [], c.get("committee_summary", "")
    if a == "Trader":
        return "approve" if c.get("execute") else "info", [f"{len(c.get('approved_orders') or [])} orders",
                                                          f"{len(c.get('rejected') or [])} rejected"], c.get("execution_notes", "")
    if a == "Trade Rationale Writer":
        return "info", [], " ".join(f"{t}: {x}" for t, x in (c.get("rationales") or {}).items())[:600]
    if a == "Screener":
        f = c.get("funnel") or {}
        return "info", [f"top screen: {', '.join(list((c.get('top_10_screen') or {}))[:5])}"], (
            f"{f.get('universe')} names → {f.get('passed_hard_filters')} pass filters → "
            f"{f.get('stage1_kept', '')} after Stage 1 → {f.get('shortlist_total')} shortlisted.")
    if a == "Regime Engine":
        return {"risk_on": "bullish", "risk_off": "bearish"}.get(c.get("regime"), "neutral"), [], \
            f"Raw {c.get('raw_regime')} → {c.get('regime')}. {c.get('note') or ''}".strip()
    if a == "Model Lab":
        return "info", [], model_lab_line(c)
    if a == "System":
        if "scoreboard" in c:
            return "info", [], c.get("scoreboard_line", "")
        if "funnel" in c:
            f = c["funnel"]
            return "info", [], (f"Screening: {f.get('universe')} names → {f.get('passed_hard_filters')} pass filters → "
                                f"{f.get('stage1_kept', '')} after Stage 1 → {f.get('shortlist_total')} shortlisted.")
        if "raw_regime" in c:
            return "info", [], f"Regime rules: raw {c.get('raw_regime')} → {c.get('regime')} ({c.get('note')})."
        return "info", [], ", ".join(f"{k}: {v}" for k, v in c.items() if not isinstance(v, (list, dict)))[:300]
    return "info", [], ", ".join(f"{k}: {v}" for k, v in c.items() if not isinstance(v, (list, dict)))[:300]


STATUS_AGENTS = ["Morning Brief", "Macro Strategist", "Equity Research Analyst", "Quant Analyst", "Sentiment Analyst",
                 "Portfolio Manager", "Devil's Advocate", "Risk Manager", "CIO", "Trader"]


def stage_status(msgs: list[dict], agent: str) -> str:
    """ok / fallback / failed / — for one agent in one run (worst message wins)."""
    mine = [m for m in msgs if m["agent"] == agent]
    if not mine:
        return "—"
    bad = [str(m["model"]).startswith("FAILED") for m in mine]
    if all(bad):
        return "failed"                       # LLM down, neutral defaults only (D11)
    if any(m["content"].get("agent_failed") or str(m["model"]).startswith("FAILED") or "fallback" in (m["title"] or "").lower()
           or str(m["model"]).endswith(": template") for m in mine):
        return "fallback"
    return "ok"


def run_status_row(msgs: list[dict], status: str) -> str:
    color = {"ok": ui.GREEN, "fallback": ui.AMBER, "failed": ui.RED}
    cells = []
    for a in STATUS_AGENTS:
        v = stage_status(msgs, a)
        c = color.get(v, ui.MUTED)
        cells.append(f'<span class="chip" style="background:{c}22;color:{c}">{ui.esc(a.split()[0])}: {v}</span>')
    cio = next((m["content"].get("decision") for m in reversed(msgs) if m["agent"] == "CIO"), None)
    risk = next((m["content"].get("decision") for m in reversed(msgs) if m["agent"] == "Risk Manager"), None)
    ex = "yes" if status == "executed" else "no"
    summary = (f"<b>Decision</b> {ui.esc(cio or '—')} · <b>Risk verdict</b> {ui.esc(risk or '—')} · "
               f"<b>Executed</b> {ex} <span style='color:{ui.MUTED}'>({ui.esc(status)})</span>")
    return f"<div class='card'><div style='margin-bottom:6px'>{summary}</div><div>{' '.join(cells)}</div></div>"


with tab_room:
    if REPLAY:
        st.info("Replay fund (read-only). Generate it with `python orchestrator.py --replay 20`.")
    elif halted:
        st.error("⛔ HALT file present in the repo root: you can run the analysis, execution is disabled.")
    left, right = st.columns([1, 3])
    with left:
        run_clicked = (not REPLAY and not VIEWER) and st.button("▶️ Run committee", type="primary", width="stretch")
        st.caption("Init day: 10-20 buys, cap exempt. Daily: max 5 trades. Max 3 runs/day.")
    timeline_box = right.empty()
    timeline_box.markdown(ui.timeline(STEPS, 0), unsafe_allow_html=True)
    if run_clicked:
        state = {"done": 0}
        log = st.empty()

        def progress(stage, msg):
            state["done"] = max(state["done"], STAGE_TO_STEP.get(stage, state["done"]))
            timeline_box.markdown(ui.timeline(STEPS, state["done"]), unsafe_allow_html=True)
            log.caption(f"{stage}: {msg}")
        try:
            res = run_committee(provider, progress=progress)
            st.session_state["selected_run"] = res["run_id"]
        except RunLimitError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.exception(e)
        load_market.clear()
        st.rerun()

    if runs.empty:
        ui.empty("No committee meeting yet", "Press Run committee: the Morning Brief, four analysts, the PM, the Devil's "
                 "Advocate, the Risk Manager, the CIO and the Trader will meet here, step by step.")
    else:
        r = runs.set_index("id")
        ids = r.index.tolist()
        default = st.session_state.get("selected_run", ids[0])
        run_id = st.selectbox("Meeting", ids, index=ids.index(default) if default in ids else 0,
                              format_func=lambda i: f"#{i} · {db.madrid_et(r.loc[i, 'started_at'])} · {r.loc[i, 'mode']} · "
                                                    f"{r.loc[i, 'status']} · {r.loc[i, 'provider']}")
        if r.loc[run_id, "summary"]:
            st.markdown(f"<div class='card'><div class='who'>CIO minutes</div><div class='msg'>{ui.esc(r.loc[run_id, 'summary'])}</div></div>",
                        unsafe_allow_html=True)
        msgs = rules_cards(messages_of(run_id))
        prop = db.get_proposal(run_id)
        run_status = r.loc[run_id, "status"]
        book_state = "EXECUTED" if run_status == "executed" else "PROPOSED"
        st.markdown(run_status_row(msgs, run_status), unsafe_allow_html=True)
        has_da_answer = any(x["agent"] == "Portfolio Manager" and x["stage"] == "proposal_answers_da" for x in msgs)
        show_diag = st.toggle("Show diagnostics (agent-by-agent minutes, prompts, raw output)", value=False)
        for m in (msgs if show_diag else []):
            stance, nums, msg = summarize(m)
            if m["content"].get("agent_failed"):
                stance, msg = "info", f"⚠ Agent failed → {m['content'].get('fallback', 'rules fallback')}. " + (msg or "")
            ui.agent_card(m["agent"], m["title"], m["model"], m.get("latency_s"), stance, nums, msg)
            c = m["content"]
            if m["agent"] == "Risk Manager" and c.get("limit_checks"):
                st.caption(f"**{book_state}** post-trade book (run status: {run_status}). The CURRENT book is in the Portfolio tab.")
                ui.limit_grid(c["limit_checks"])
            if m["agent"] == "Devil's Advocate" and c.get("objections"):
                ans = next((x["content"].get("da_responses") or {} for x in msgs
                            if x["agent"] == "Portfolio Manager" and x["stage"] == "proposal_answers_da"), {})
                d1, d2 = st.columns(2)
                d1.markdown("**Devil's Advocate**")
                d2.markdown("**PM's answer**")
                for t, o in c["objections"].items():
                    d1.markdown(f"{ui.chip(o['verdict'])} **{ui.esc(data.label(t))}** — {ui.esc(o['bear_case'])} "
                                f"<span style='color:{ui.MUTED}'>({ui.esc(o['evidence'])})</span>", unsafe_allow_html=True)
                    d2.markdown(f"**{ui.esc(t)}** — {ui.esc(ans.get(t, '—'))}", unsafe_allow_html=True)
            if m["agent"] == "Portfolio Manager" and (m["stage"] == "proposal_answers_da" or (not has_da_answer and m["round"] == 1)):
                conv = pd.Series(c.get("convictions") or {}).sort_values(ascending=False)
                if len(conv):
                    st.caption("Top convictions: " + ", ".join(f"{t} {v:+.2f}" for t, v in conv.head(8).items()))
            if m["agent"] == "Model Lab":
                st.markdown(f"**In plain English:** {lab_summary.plain_summary(c.get('race') or c)}")
                with st.expander("How to read this"):
                    st.markdown(lab_summary.HOW_TO_READ)
                with st.expander("Model card · horse race scoreboard + LSTM"):
                    if (c.get("race") or {}).get("scoreboard_line"):
                        st.caption(c["race"]["scoreboard_line"])
                    if c.get("lstm"):
                        st.json(c["lstm"], expanded=False)
            with st.expander(f"What {m['agent']} saw · raw output"):
                trk = f"_{tracker.line(tracker.from_db(db, close))}_\n\n" if m["agent"] == "Quant Analyst" else ""
                if trk and not m.get("system_prompt"):
                    st.caption(trk.strip("_\n"))
                if m.get("system_prompt"):
                    st.markdown(trk + "**System prompt**")      # tracker rides in the existing element: UI tree unchanged
                    st.code(m["system_prompt"][:6000], language=None)
                    st.markdown("**User prompt**")
                    st.code((m.get("user_prompt") or "")[:12000], language=None)
                    st.caption(f"tokens {m.get('tokens')} · latency {m.get('latency_s')}s")
                st.json(c, expanded=False)

        if prop and prop.get("orders"):
            st.markdown("#### Trade ticket")
            po = pd.DataFrame(prop["orders"])
            po["name"] = po["ticker"].map(data.label)
            st.dataframe(po[["name", "side", "shares", "price", "notional", "target_weight", "reason"]], hide_index=True, width="stretch",
                         column_config={"notional": st.column_config.NumberColumn(format="$%,.0f"),
                                        "target_weight": st.column_config.NumberColumn("target", format="%.3f")})
        if r.loc[run_id, "status"] == "proposed" and not REPLAY and not VIEWER:
            st.caption(f"Proposed {db.madrid_et(r.loc[run_id, 'finished_at'])}. Execution reprices at the latest price; "
                       f"proposals older than 2 h are refused.")
            if halted:
                st.error("⛔ HALT file present: execution disabled.")
            else:
                ok = st.checkbox("I have read the reasoning and the Risk Manager's checks")
                if st.button("✅ Execute trades", type="primary", disabled=not ok):
                    try:
                        out = execute_run(run_id)
                        st.toast(f"Executed {len(out['trades'])} trade(s) · NAV ${out['nav']:,.0f}")
                        publish()
                    except RuntimeError as e:
                        st.error(str(e))
                        st.stop()
                    st.rerun()

# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
with tab_pf:
    if not positions:
        ui.empty("Nothing to analyse yet", "Exposure, risk and liquidity views appear after initialisation.")
    else:
        w = pd.Series(weights)
        st.caption("**CURRENT book** · executed positions only. These numbers change only when trades execute; "
                   "a pending proposal's post-trade risk is in the Decision Room, labelled PROPOSED.")
        a1, a2 = st.columns(2)
        caps = pos_df.groupby("Cap")["Weight"].sum().reindex(["Mega", "Large", "Mid", "Small", "ETF", "–"]).dropna()
        ui.chart(a1, go.Figure(go.Bar(x=caps.index, y=caps.values, text=[f"{v:.0f}%" for v in caps.values],
                                         textposition="outside", marker_color=ui.ACCENT)).update_layout(
            title="Market-cap mix (% of NAV)", height=320, yaxis_ticksuffix="%"))
        tm = pd.DataFrame({"t": w.index, "w": w.values, "s": [sector_of.get(t, "?") for t in w.index]})
        secs = list(tm["s"].unique())
        ui.chart(a2, go.Figure(go.Treemap(labels=secs + list(tm["t"]), parents=[""] * len(secs) + list(tm["s"]),
                                             ids=[f"s:{x}" for x in secs] + list(tm["t"]),
                                             values=[tm[tm["s"] == x]["w"].sum() for x in secs] + list(tm["w"]),
                                             branchvalues="total", texttemplate="%{label}<br>%{value:.1%}")
                                  ).update_traces(parents=[""] * len(secs) + [f"s:{x}" for x in tm["s"]]).update_layout(
            title="Sector → position", height=320))
        import attribution
        spy_w, asof = attribution.spy_sector_weights()
        sw = tm[tm["s"].isin(spy_w)].groupby("s")["w"].sum()
        act = pd.Series({x: sw.get(x, 0.0) - v for x, v in spy_w.items()}).sort_values()
        b1, b2 = st.columns(2)
        ui.chart(b1, go.Figure(go.Bar(x=act.values * 100, y=act.index, orientation="h",
                                         marker_color=[ui.GREEN if v > 0 else ui.RED for v in act.values])).update_layout(
            title=f"Active sector weights vs SPY (pp; SPY as of {asof})", height=360))
        held = [t for t in w.index if t in close.columns]
        rets = close[held + [config.BENCHMARK]].pct_change().dropna().tail(252)
        cov = rets[held].cov().values * 252
        wv = w[held].values
        port_var = float(wv @ cov @ wv)
        rc = wv * (cov @ wv) / port_var if port_var > 0 else wv * 0
        betas = np.array([float(np.cov(rets[t], rets[config.BENCHMARK])[0, 1] / rets[config.BENCHMARK].var()) for t in held])
        rc_df = pd.DataFrame({"t": held, "weight": wv * 100, "risk": rc * 100, "beta_c": betas * wv})
        ui.chart(b2, go.Figure([go.Bar(name="weight %", x=rc_df["t"], y=rc_df["weight"], marker_color=ui.BLUE3),
                                   go.Bar(name="risk contribution %", x=rc_df["t"], y=rc_df["risk"], marker_color=ui.ACCENT)]
                                  ).update_layout(title=f"Current · risk contribution vs weight (vol {np.sqrt(port_var):.1%})",
                                                  height=360, barmode="group"))
        c1_, c2_ = st.columns(2)
        ui.chart(c1_, go.Figure(go.Bar(x=rc_df["t"], y=rc_df["beta_c"], marker_color=ui.BLUE2)).update_layout(
            title=f"Current · beta contribution (book beta {rc_df['beta_c'].sum():.2f}, 1y daily vs SPY)", height=340))
        etf_n = {t: weights[t] * config.LEVERAGE_FACTOR[t] for t in held if t in config.LEVERAGE_FACTOR}
        net_n = sum(weights[t] * config.LEVERAGE_FACTOR.get(t, 1.0) for t in held)
        c1_.caption("Current notional exposure: " + (", ".join(f"{t} {weights[t]:.1%} × {config.LEVERAGE_FACTOR[t]:g} = {v:.1%}"
                                                           for t, v in etf_n.items()) or "no leveraged ETFs")
                    + f" · net notional {net_n:.0%} of NAV")
        corr = rets[held].corr()
        ui.chart(c2_, go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                              colorscale=[[0, ui.RED], [0.5, ui.PANEL2], [1, ui.ACCENT]])).update_layout(
            title="Correlation of holdings (1y daily)", height=340))
        d1_, d2_, d3_ = st.columns(3)
        lev = sum(v for t, v in weights.items() if kind_of.get(t) == "leveraged_etf")
        cap_r = min(config.RISK_LIMITS["max_leveraged_total"],
                    config.REGIME_RULES.get(latest_regime() or "neutral", config.REGIME_RULES["neutral"])["max_leveraged"])
        ui.chart(d1_, go.Figure(go.Indicator(mode="gauge+number", value=lev * 100, number={"suffix": "%"},
                                                gauge={"axis": {"range": [0, 30]}, "bar": {"color": ui.ACCENT},
                                                       "threshold": {"line": {"color": ui.RED, "width": 3}, "value": cap_r * 100}})
                                   ).update_layout(title="Leveraged sleeve vs regime cap", height=280))
        adv = analytics.adv20({"close": close, "volume": volume}, list(positions))
        liq = pd.Series({t: positions[t]["shares"] * last_px.get(t, 0) / (0.25 * adv[t]) for t in positions if adv.get(t)})
        ui.chart(d2_, go.Figure(go.Bar(x=liq.index, y=liq.values, marker_color=ui.BLUE3)).update_layout(
            title="Days to exit at 25% of ADV", height=280))
        uni = [t for t in close.columns if kind_of.get(t) == "stock"]
        mom = (close[uni].shift(21) / close[uni].shift(252) - 1).iloc[-1]
        vol3 = close[uni].pct_change().tail(63).std() * np.sqrt(252)
        trend = close[uni].iloc[-1] / close[uni].tail(200).mean() - 1
        book = [t for t in positions if t in uni]
        style = pd.Series({"momentum (12-1m)": mom.rank(pct=True)[book].mean(), "trend (vs 200dma)": trend.rank(pct=True)[book].mean(),
                           "low vol": 1 - vol3.rank(pct=True)[book].mean()}) * 100
        ui.chart(d3_, go.Figure(go.Bar(x=style.index, y=style.values, marker_color=ui.ACCENT,
                                          text=[f"{v:.0f}" for v in style.values], textposition="outside")
                                   ).add_hline(y=50, line_dash="dot", line_color=ui.MUTED).update_layout(
            title="Style of the book vs the S&P 500 (percentile)", height=280, yaxis_range=[0, 110]))

# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
with tab_perf:
    sess = snaps.drop_duplicates("date", keep="last") if len(snaps) else snaps   # one point per NYSE session
    if len(sess) < 2:
        ui.empty("Performance builds up day by day", "NAV vs SPY, drawdown, attribution and the committee's hit rate appear "
                 f"after the second daily snapshot (after-close snapshot at {_lbl['mark']} Madrid).")
    else:
        s = sess.copy()
        s["date"] = pd.to_datetime(s["date"])
        PCT = dict(ticksuffix="%", tickformat=".2f")          # no nano-scale axes on flat early days
        SESS = dict(type="category", title=None)              # one tick per session, no weekend gaps
        s["d"] = s["date"].dt.strftime("%b %d")
        p1, p2 = st.columns(2)
        ui.chart(p1, go.Figure([go.Scatter(x=s["d"], y=(s["total_equity"] / config.STARTING_CAPITAL - 1) * 100, name="Fund",
                                              mode="lines+markers", line=dict(width=3)),
                                   go.Scatter(x=s["d"], y=(s["benchmark_px"] / s["benchmark_px"].iloc[0] - 1) * 100, name="SPY",
                                              mode="lines+markers", line=dict(dash="dot", color=ui.BLUE3))]).update_layout(
            title="Cumulative return vs SPY (%)", height=320, yaxis=PCT, xaxis=SESS))
        dd = s["total_equity"] / s["total_equity"].cummax().clip(lower=config.STARTING_CAPITAL) - 1
        fig = go.Figure(go.Scatter(x=s["d"], y=dd * 100, fill="tozeroy", mode="lines+markers", line=dict(color=ui.RED)))
        for lvl in (config.DD_WARN, config.DD_HARD):
            fig.add_hline(y=lvl * 100, line_dash="dot", line_color=ui.AMBER, annotation_text=f"C11 {lvl:.0%}")
        ui.chart(p2, fig.update_layout(title="Drawdown (underwater, %)", height=320, xaxis=SESS,
                                          yaxis={**PCT, "range": [min(-11, dd.min() * 100 - 1), 1]}))
        dr = s.set_index("d")["total_equity"].pct_change().dropna()
        q1, q2 = st.columns(2)
        if len(dr) < 2:
            with q1:
                ui.empty("Daily return (%)", "Needs at least 2 daily returns (3 sessions).")
        else:
            ui.chart(q1, go.Figure(go.Bar(x=dr.index, y=dr.values * 100, marker_color=[ui.GREEN if v >= 0 else ui.RED for v in dr])
                                      ).update_layout(title="Daily return (%)", height=280, yaxis=PCT, xaxis=SESS))
        rv = dr.rolling(5).std().dropna() * np.sqrt(252) * 100
        if len(rv) < 2:
            with q2:
                ui.empty("Rolling 5-day vol (annualised %)", "Needs 6+ sessions.")
        else:
            ui.chart(q2, go.Figure(go.Scatter(x=rv.index, y=rv.values, mode="lines+markers", line=dict(color=ui.BLUE2))
                                      ).update_layout(title="Rolling 5-day vol (annualised %)", height=280, yaxis=PCT, xaxis=SESS))
        att = report.attribution_result()
        r1, r2 = st.columns(2)
        if att:
            pp = att["cumulative"]["parts"]
            labels = {"sector_calls": "Sector calls", "benchmark_tracking": "SPY vs sector ETFs", "stock_picking": "Selection",
                      "leverage_beta": "Leverage beta", "leverage_decay": "Leverage decay", "cash_drag": "Cash drag"}
            ui.chart(r1, go.Figure(go.Waterfall(x=[labels[x] for x in labels] + ["Total vs SPY"],
                                                   measure=["relative"] * len(labels) + ["total"], y=[pp[x] * 100 for x in labels] + [0],
                                                   increasing={"marker": {"color": ui.GREEN}}, decreasing={"marker": {"color": ui.RED}},
                                                   totals={"marker": {"color": ui.ACCENT}})).update_layout(
                title="Attribution vs SPY (Brinson-Fachler, Carino)", height=320, yaxis_ticksuffix="%"))
            r1.caption(att["line"])
        else:
            with r1:
                ui.empty("Attribution", "Needs two snapshots after the first trade.")
        refl = db.get_reflections(evaluated_only=True)
        if not refl.empty:
            rr_ = refl.sort_values("eval_date")
            hit = (rr_["verdict"] == "right").cumsum() / ((rr_["verdict"] != "flat").cumsum().replace(0, np.nan))
            ui.chart(r2, go.Figure(go.Scatter(x=pd.to_datetime(rr_["eval_date"]), y=hit * 100, mode="lines+markers")).add_hline(
                y=50, line_dash="dot", line_color=ui.MUTED).update_layout(title="Committee hit rate over time (%)", height=320))
        else:
            with r2:
                ui.empty("Committee hit rate", "The first calls are graded 5 trading days after the first trades.")
    hr = last_message("System", "scoreboard")
    if hr:
        st.markdown("#### Model horse race (same out-of-sample origins, IPS C3.4 weights)")
        sb = pd.DataFrame(hr["content"]["scoreboard"]).T
        sb["weight"] = pd.Series(hr["content"].get("weights", {}))
        st.markdown(f"**In plain English:** {lab_summary.plain_summary(hr['content'])}")
        st.dataframe(sb, width="stretch")
        st.caption(hr["content"].get("scoreboard_line", ""))
        with st.expander("How to read this"):
            st.markdown(lab_summary.HOW_TO_READ)

# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
with tab_tr:
    if trades.empty:
        ui.empty("No trades yet", "Every trade lands here with its plain-English reason (IPS E2) and its signals.")
    else:
        f1, f2, f3 = st.columns(3)
        dates = sorted(trades["date"].unique(), reverse=True)
        pick = f1.multiselect("Date", dates, default=dates[:1])
        side = f2.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
        q = f3.text_input("Ticker or name contains")
        t_ = trades[trades["date"].isin(pick or dates) & trades["action"].isin(side)].copy()
        t_["name"] = t_["symbol"].map(data.label)
        if q:
            t_ = t_[t_["name"].str.contains(q, case=False)]
        for col in ("proposed_at", "price_source", "price_time"):
            if col not in t_:
                t_[col] = None
        t_["decided"] = t_["proposed_at"].map(lambda x: db.madrid_et(x)[:16] if isinstance(x, str) and x else "—")
        t_["price_source"] = t_["price_source"].fillna("— (before Oct 21)")
        st.dataframe(t_[["date", "hour", "decided", "action", "name", "shares", "price", "price_source", "price_time", "total", "reason"]]
                     .rename(columns={"hour": "executed (Madrid)", "decided": "decided (Madrid)", "price_source": "price source",
                                      "price_time": "price time (Madrid)"}),
                     hide_index=True, width="stretch", column_config={"total": st.column_config.NumberColumn(format="$%,.0f"),
                                                                      "price": st.column_config.NumberColumn(format="$%.2f")})
        with st.expander("Signals behind each trade"):
            for r_ in t_.itertuples():
                st.markdown(f"**{r_.date} {r_.hour} {r_.action} {data.label(r_.symbol)}** — {r_.signals or '—'}")
        db.export_trades_csv()
        if os.path.exists(config.TRADES_CSV):
            with open(config.TRADES_CSV, "rb") as fh:
                st.download_button("⬇️ trades.csv (Brief §10.4)", fh, file_name=os.path.basename(config.TRADES_CSV))
    st.markdown("#### Forum post (17:00–18:00 Madrid)")
    st.code(report.forum_post(), language=None)
    if not VIEWER and st.button("Build daily report + Word summary"):
        st.session_state["final_doc"] = (report.daily_report(), report.final_summary())
    if st.session_state.get("final_doc"):
        with open(st.session_state["final_doc"][1], "rb") as fh:
            st.download_button("⬇️ final_summary.docx", fh, file_name="final_summary.docx")

# --------------------------------------------------------------------------- #
# Committee
# --------------------------------------------------------------------------- #
with tab_com:
    all_msgs = [m for rid in runs["id"].tolist() for m in messages_of(rid)] if not runs.empty else []
    cols = st.columns(3)
    for i, cls in enumerate(ROSTER):
        a = cls()
        color, owner, ini = ui.AGENT_STYLE.get(a.name, (ui.BLUE2, "team", a.name[:2]))
        mine = [m for m in all_msgs if m["agent"] == a.name]
        valid = sum(1 for m in mine if not str(m["model"]).startswith("FAILED"))
        rel = f"{valid} of {len(mine)} valid" if mine else "no runs yet"
        model = "rules (no LLM)" if a.tier == "rules" else config.PROVIDERS.get(config.LLM_PROVIDER, {}).get(a.tier, a.tier)
        brief = (a.role_prompt.strip().split(". ")[0] + ".")[:220]
        cols[i % 3].markdown(f'''<div class="card"><div class="head"><div class="avatar" style="background:{color}">{ui.esc(ini)}</div>
<div><div class="who">{ui.esc(a.name)}</div><div class="meta">owner {ui.esc(owner)} · {ui.esc(model)}</div></div></div>
<div class="msg">{ui.esc(brief)}</div><div class="nums">reliability: {ui.esc(rel)}</div></div>''', unsafe_allow_html=True)
    st.markdown("#### Lessons learned (reflection memory)")
    rec = reflection.track_record()
    refl_all = db.get_reflections()
    if refl_all.empty:
        ui.empty("No calls recorded yet", "Every executed trade is stored with its thesis and invalidation price and graded "
                 "vs SPY after 5 trading days.")
    else:
        if rec.get("calls"):
            st.caption(f"{rec['calls']} graded calls · hit rate {rec['hit_rate']} · last 10 {rec['last10_hit_rate']} · "
                       f"avg excess {rec['avg_excess_ret']:+.2%}")
        refl_all["name"] = refl_all["ticker"].map(data.label)
        st.dataframe(refl_all[["decided_at", "name", "action", "verdict", "excess_ret", "lesson", "thesis", "invalidation_px"]],
                     hide_index=True, width="stretch")
    st.markdown("#### Morning Brief archive")
    briefs = [m for m in all_msgs if m["agent"] == "Morning Brief"]
    if not briefs:
        ui.empty("No brief yet", "The Morning Brief is written before every committee run.")
    for m in briefs[:5]:
        with st.expander(f"{db.madrid_et(m['ts'])} · {m['model']}"):
            st.write(m["content"].get("text", ""))
    st.markdown("#### Agent failure log (D11)")
    fails = [m for m in all_msgs if str(m["model"]).startswith("FAILED")]
    if fails:
        st.dataframe(pd.DataFrame([{"time": db.madrid_et(m["ts"]), "agent": m["agent"], "stage": m["stage"], "error": m["model"]}
                                   for m in fails]), hide_index=True, width="stretch")
    else:
        st.caption("No agent failures recorded.")

st.write("")
with st.expander("Data quality", expanded=False):
    (st.warning if px_src.QUALITY["flags"] else st.info)(_quality_line)
