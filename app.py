"""
Streamlit dashboard (Brief §3 + §8.3).   Run:  streamlit run app.py

Sidebar:   current simulation date · Start New Simulation (reset) · Advance Day (snapshot)
Main:      Total Portfolio Value · Cash Available
Tabs:      Trading Floor (Run Agents -> reasoning -> Execute Trades) · Portfolio ·
           Performance (Plotly, Total Return / Max Drawdown / Win Rate, daily snapshots) ·
           Trades (day by day, with reasoning; trades.csv) · Committee
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data
import database as db
import report
from agents import BRIEF_ROLES, ROSTER
from orchestrator import RunLimitError, execute_run, run_committee, snapshot_only

st.set_page_config(page_title=config.FUND_NAME, page_icon="🏛️", layout="wide")

BLUE, LIGHT, GREY = "#1F4E9C", "#9DB9E3", "#5A6B85"
AVATARS = {
    "CIO": "👔", "Macro Strategist": "🌍", "Equity Research Analyst": "📑",
    "Quant Analyst": "🧠", "Sentiment Analyst": "📰", "Portfolio Manager": "📈",
    "Risk Manager": "🛡️", "Trader": "💹", "System": "⚙️",
}
st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {BLUE}; }}
  [data-testid="stMetricValue"] {{ color: {BLUE}; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }}
  .ok {{ background:#DCEBFF; color:{BLUE}; }}
  .no {{ background:#FDE2E2; color:#9C1F1F; }}
  .muted {{ color:{GREY}; font-size:0.85rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner="Loading S&P 500 prices…")
def load_market():
    u = data.get_universe()
    close = data.get_market_data(u["symbol"].tolist())["close"]
    return u, close, data.DATA_SOURCE["prices"], data.DATA_SOURCE["universe"], data.is_synthetic()


def badge(text: str, good: bool) -> str:
    return f'<span class="badge {"ok" if good else "no"}">{text}</span>'


universe, close, source, uni_source, synthetic = load_market()
db.use_synthetic(synthetic)
db.init_db()
last_px = {t: float(close[t].dropna().iloc[-1]) for t in close.columns if close[t].notna().any()}
sector_of = dict(zip(universe["symbol"], universe["sector"]))

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"### 🏛️ {config.FUND_NAME}")
    st.markdown("**Current Simulation Date**")
    st.markdown(f"## {db.today()}")
    st.caption(f"Last market close in data: {close.index[-1].date()} · Madrid time {db.now():%H:%M}")
    st.divider()
    providers = list(config.PROVIDERS)
    provider = st.selectbox("AI brain (LLM provider)", providers,
                            index=providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0,
                            help="groq = free cloud (default). mock = rule-based answers, for offline testing.")
    c1, c2 = st.columns(2)
    c1.metric("Runs today", f"{db.runs_today()}/{config.MAX_RUNS_PER_DAY}")
    n_today = db.trades_today()
    c2.metric("Trades today", f"{n_today} (init day)" if n_today > config.MAX_TRADES_PER_DAY
              else f"{n_today}/{config.MAX_TRADES_PER_DAY}")
    if st.button("📅 Advance Day (revalue + snapshot)", width="stretch",
                 help="After-close run (21:45 Madrid): reprice holdings and save today's snapshot. No agents, no trades."):
        load_market.clear()
        nav = snapshot_only()
        st.toast(f"Snapshot saved · NAV ${nav:,.0f}")
        st.rerun()
    if synthetic and st.button("🔄 Retry Yahoo Finance download", width="stretch"):
        load_market.clear()
        st.rerun()
    with st.expander("Start New Simulation"):
        st.caption(f"Resets {db.current_db_name()} to ${config.STARTING_CAPITAL:,.0f} cash. Cannot be undone.")
        if st.checkbox("I understand this wipes all trades and minutes"):
            if st.button("Reset database", type="primary"):
                db.init_db(reset=True)
                st.session_state.pop("selected_run", None)
                st.rerun()

st.title("AI Investment Committee")
st.markdown(f'<span class="muted">Universe: {uni_source} + {len(config.LEVERAGED_ETFS)} leveraged / '
            f'{len(config.INVERSE_ETFS)} inverse ETFs · Benchmark: {config.BENCHMARK} · Prices: {source}</span>',
            unsafe_allow_html=True)
if source.startswith("SYNTHETIC"):
    st.warning("Yahoo Finance couldn't be reached and nothing is cached, so this is a **synthetic sandbox fund** "
               "(separate database — the real fund is untouched).")

# --------------------------------------------------------------------------- #
# Top metrics (Brief §8.3)
# --------------------------------------------------------------------------- #
weights, nav = db.current_weights(last_px)
cash = db.get_cash()
snaps = db.get_snapshots()
bench_ret = None
if len(snaps) and snaps["benchmark_px"].iloc[0]:
    bench_ret = last_px.get(config.BENCHMARK, snaps["benchmark_px"].iloc[-1]) / snaps["benchmark_px"].iloc[0] - 1
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Portfolio Value", f"${nav:,.2f}", f"{nav - config.STARTING_CAPITAL:+,.0f} since inception")
m2.metric("Cash Available", f"${cash:,.2f}", f"{cash / nav:.1%} of NAV", delta_color="off")
m3.metric("Return since inception", f"{nav / config.STARTING_CAPITAL - 1:+.2%}",
          f"SPY {bench_ret:+.2%}" if bench_ret is not None else None, delta_color="off")
m4.metric("Positions", len(weights), f"limit {config.MIN_HOLDINGS}-{config.MAX_HOLDINGS}", delta_color="off")

tab_floor, tab_pf, tab_perf, tab_trades, tab_team = st.tabs(
    ["🏛️ Trading Floor", "📊 Portfolio", "📈 Performance", "🧾 Trades", "👥 Committee"])


# --------------------------------------------------------------------------- #
# Trading Floor: Run Agents -> reasoning -> Execute Trades
# --------------------------------------------------------------------------- #
def render_message(m: dict) -> None:
    a, c = m["agent"], m["content"]
    with st.chat_message(a, avatar=AVATARS.get(a, "🤖")):
        rnd = f" · round {m['round']}" if m["stage"] in ("proposal", "risk_review", "risk_override") else ""
        st.markdown(f"**{a}** — {m['title']}<br><span class='muted'>{m['stage']}{rnd} · model: {m['model']}</span>",
                    unsafe_allow_html=True)
        if c.get("agent_failed"):
            st.error("Agent failed this cycle — treated as no signal (IPS D11).")
        if a == "System":
            st.json(c, expanded=False)
        elif a == "Macro Strategist":
            st.markdown(f"Regime **{c['regime']}** ({c.get('regime_note', '')}) · equity exposure "
                        f"**{c['equity_exposure_target']:.0%}** · leveraged room **{c.get('max_leveraged', 0):.0%}**")
            st.write(c["rationale"])
            if c.get("tilt_override"):
                st.info(f"Tilt override: {c['tilt_override']}")
            st.dataframe(pd.Series(c["sector_tilts"], name="tilt"), width="stretch")
        elif a == "Equity Research Analyst":
            st.markdown(f"Top pick **{c.get('top_pick')}** · least preferred **{c.get('least_preferred')}**")
            st.dataframe(pd.DataFrame(c["ratings"]).T, width="stretch")
        elif a == "Quant Analyst":
            st.markdown(f"LSTM reliability: **{c['model_reliability']}** — {c['note']}")
            st.dataframe(pd.DataFrame(c["signals"]).T, width="stretch")
        elif a == "Sentiment Analyst":
            st.write(c.get("market_mood", ""))
            st.dataframe(pd.DataFrame(c["sentiment"]).T, width="stretch")
        elif a == "Portfolio Manager":
            st.markdown(f"Equity exposure **{c['equity_exposure']:.0%}**")
            st.write(c["rationale"])
            if c.get("response_to_risk"):
                st.info(f"Response to Risk: {c['response_to_risk']}")
            conv = pd.Series(c["convictions"], name="conviction").sort_values()
            st.bar_chart(conv[conv != 0], color=LIGHT)
        elif a == "Risk Manager":
            st.markdown(badge(c["decision"].upper(), c["decision"] == "approve"), unsafe_allow_html=True)
            st.write(c.get("rationale", ""))
            for i in c.get("issues", []):
                st.markdown(f"- {i}")
            for i in c.get("soft_concerns", []):
                st.markdown(f"- ⚠️ {i}")
            if c.get("required_changes"):
                st.warning(c["required_changes"])
        elif a == "CIO":
            st.markdown(badge(c["decision"].upper(), c["decision"] == "approve"), unsafe_allow_html=True)
            st.write(c["committee_summary"])
            if c.get("key_risks"):
                st.markdown("**Key risks:** " + " · ".join(c["key_risks"]))
            if c.get("dissent_noted"):
                st.caption(f"Dissent: {c['dissent_noted']}")
        elif a == "Trader":
            st.write(c.get("execution_notes", ""))
            if c.get("approved_orders"):
                st.dataframe(pd.DataFrame(c["approved_orders"])[["ticker", "side", "shares", "price", "notional", "reason"]],
                             width="stretch", hide_index=True)
            if c.get("rejected"):
                st.markdown("**Rejected:** " + "; ".join(f"{r['ticker']} ({r['reason']})" for r in c["rejected"]))
            if c.get("skipped_by_planner"):
                with st.expander(f"{len(c['skipped_by_planner'])} trade idea(s) not taken"):
                    st.dataframe(pd.DataFrame(c["skipped_by_planner"]), width="stretch", hide_index=True)
            for f in c.get("flags", []):
                st.markdown(f"- ⚠️ {f}")
        else:
            st.json(c)


with tab_floor:
    mode = "Initialisation (build 10-20 positions, trade cap exempt)" if not db.is_initialized() else \
        f"Daily (max {config.MAX_TRADES_PER_DAY} trades)"
    st.markdown(f"**Mode:** {mode}")
    if st.button("▶️ Run Agents", type="primary"):
        with st.status("Investment committee in session…", expanded=True) as status:
            def progress(stage, msg):
                status.write(f"**{stage}** — {msg}")
            try:
                res = run_committee(provider, progress=progress)
                status.update(label=f"Proposal: {res['status']} · {len(res['orders'])} order(s) · "
                                    f"{res['usage']['calls']} LLM calls · regime {res['regime']}", state="complete")
                st.session_state["selected_run"] = res["run_id"]
            except RunLimitError as e:
                status.update(label="Run limit reached", state="error")
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                status.update(label="Meeting failed", state="error")
                st.exception(e)
        load_market.clear()
        st.rerun()

    runs = db.get_runs()
    if runs.empty:
        st.info("No meetings yet. Press **Run Agents**.")
    else:
        ids = runs["id"].tolist()
        default = st.session_state.get("selected_run", ids[0])
        r = runs.set_index("id")
        run_id = st.selectbox("Meeting", ids, index=ids.index(default) if default in ids else 0,
                              format_func=lambda i: f"#{i} — {r.loc[i, 'started_at']} · {r.loc[i, 'mode']} · "
                                                    f"{r.loc[i, 'status']} · {r.loc[i, 'provider']}")
        status_ = r.loc[run_id, "status"]
        if r.loc[run_id, "summary"]:
            st.success(f"**CIO minutes:** {r.loc[run_id, 'summary']}")
        prop = db.get_proposal(run_id)
        if status_ == "proposed" and prop:
            st.subheader("Proposed trades")
            st.dataframe(pd.DataFrame(prop["orders"])[["ticker", "side", "shares", "price", "notional", "reason"]],
                         width="stretch", hide_index=True)
            if st.button("✅ Execute Trades", type="primary"):
                out = execute_run(run_id)
                st.toast(f"Executed {len(out['trades'])} trade(s) · NAV ${out['nav']:,.0f}")
                st.rerun()
        st.subheader("Agent reasoning")
        for m in db.get_messages(run_id):
            render_message(m)

# --------------------------------------------------------------------------- #
# Portfolio (Brief §8.3 tab 2) + risk summary (Brief Fig. 1)
# --------------------------------------------------------------------------- #
with tab_pf:
    pos = db.get_positions()
    if not pos:
        st.info("All cash. Run the agents to build the starting portfolio.")
    else:
        rows = []
        for t, p in pos.items():
            px = last_px.get(t, p["avg_cost"])
            mv = p["shares"] * px
            rows.append({"Symbol": t, "Sector": sector_of.get(t, "?"), "Shares": p["shares"],
                         "Avg cost": p["avg_cost"], "Price": px, "Market Value": mv, "Weight": mv / nav,
                         "P/L": p["shares"] * (px - p["avg_cost"]), "P/L %": px / p["avg_cost"] - 1})
        df = pd.DataFrame(rows).sort_values("Weight", ascending=False)
        left, right = st.columns([3, 2])
        left.dataframe(df.style.format({"Avg cost": "${:,.2f}", "Price": "${:,.2f}", "Market Value": "${:,.0f}",
                                        "Weight": "{:.1%}", "P/L": "${:+,.0f}", "P/L %": "{:+.1%}"}),
                       width="stretch", hide_index=True)
        sec = df.groupby("Sector")["Weight"].sum().sort_values()
        fig = go.Figure(go.Bar(x=sec.values, y=sec.index, orientation="h", marker_color=BLUE,
                               text=[f"{v:.0%}" for v in sec.values], textposition="outside"))
        fig.add_vline(x=config.RISK_LIMITS["max_sector_weight"], line_dash="dot", line_color="#9C1F1F",
                      annotation_text="35% sector cap")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), title="Exposure by sector",
                          xaxis_tickformat=".0%")
        right.plotly_chart(fig, width="stretch")

        st.subheader("Position limits / risk summary")
        lev = df[df["Sector"] == data.LEVERAGED_SECTOR]["Weight"].sum()
        inv = df[df["Sector"] == data.INVERSE_SECTOR]["Weight"].sum()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Max position", f"{df['Weight'].max():.1%}", f"limit {config.RISK_LIMITS['max_single_weight']:.0%}",
                  delta_color="off")
        k2.metric("Trades today", f"{db.trades_today()}/{config.MAX_TRADES_PER_DAY}")
        k3.metric("Leveraged ETFs", f"{lev:.1%}", f"limit {config.RISK_LIMITS['max_leveraged_total']:.0%}",
                  delta_color="off")
        k4.metric("Inverse ETFs", f"{inv:.1%}", "risk-off only", delta_color="off")
        k5.metric("Drawdown from peak", f"{db.drawdown_from_peak(nav):.1%}",
                  f"limit {config.RISK_LIMITS['max_drawdown']:.0%}", delta_color="off")

# --------------------------------------------------------------------------- #
# Performance (Brief §8.3 tab 3, Fig. 2)
# --------------------------------------------------------------------------- #
with tab_perf:
    if snaps.empty:
        st.info("No snapshots yet. They are saved after every run and by **Advance Day**.")
    else:
        s = snaps.copy()
        s["date"] = pd.to_datetime(s["date"])
        s["fund_idx"] = s["total_equity"] / config.STARTING_CAPITAL * 100
        s["spy_idx"] = s["benchmark_px"] / s["benchmark_px"].iloc[0] * 100
        dd = s["total_equity"] / s["total_equity"].cummax().clip(lower=config.STARTING_CAPITAL) - 1
        daily = s["total_equity"].pct_change().dropna()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Return", f"{s['total_equity'].iloc[-1] / config.STARTING_CAPITAL - 1:+.2%}")
        k2.metric("Max Drawdown", f"{dd.min():.2%}")
        k3.metric("Win Rate (up days)", f"{(daily > 0).mean():.0%}" if len(daily) else "–")
        k4.metric("vs SPY", f"{(s['fund_idx'].iloc[-1] - s['spy_idx'].iloc[-1]):+.2f} pts")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["date"], y=s["total_equity"], name="Portfolio value",
                                 line=dict(color=BLUE, width=3), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=s["date"], y=s["spy_idx"] / 100 * config.STARTING_CAPITAL, name="SPY (same $1M)",
                                 line=dict(color=LIGHT, width=2, dash="dot")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickprefix="$",
                          yaxis_tickformat=",.0f", legend=dict(orientation="h", y=1.08), hovermode="x unified",
                          title="Portfolio value over time")
        st.plotly_chart(fig, width="stretch")
        st.subheader("Daily snapshots")
        tr = db.get_trades()
        n_tr = tr.groupby("date").size() if not tr.empty else pd.Series(dtype=int)
        tbl = snaps[["date", "cash", "invested", "total_equity", "n_positions"]].copy()
        tbl["trades"] = tbl["date"].map(n_tr).fillna(0).astype(int)
        st.dataframe(tbl.rename(columns={"invested": "holdings", "total_equity": "total"}).style.format(
            {"cash": "${:,.0f}", "holdings": "${:,.0f}", "total": "${:,.0f}"}), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Trades day by day, with reasoning (Brief §1, §10.4)
# --------------------------------------------------------------------------- #
with tab_trades:
    trades = db.get_trades()
    if trades.empty:
        st.info("No trades yet.")
    else:
        for day, g in trades.groupby("date", sort=False):
            with st.expander(f"{day} — {len(g)} trade(s)", expanded=day == trades["date"].iloc[0]):
                st.dataframe(g[["hour", "action", "symbol", "shares", "price", "total", "reason"]],
                             width="stretch", hide_index=True)
    with st.expander("📣 Today's forum post (17:00–18:00 Madrid) and reports"):
        st.code(report.forum_post(), language=None)
        if st.button("Build daily report + 15-day Word summary"):
            st.session_state["final_doc"] = (report.daily_report(), report.final_summary())
        if st.session_state.get("final_doc"):
            daily_path, final_path = st.session_state["final_doc"]
            st.caption(f"Daily report: {daily_path}")
            with open(final_path, "rb") as f:
                st.download_button("⬇️ final_summary.docx (Brief §11)", f, file_name="final_summary.docx")
    if not trades.empty:
        db.export_trades_csv()
        if os.path.exists(config.TRADES_CSV):
            with open(config.TRADES_CSV, "rb") as f:
                st.download_button("⬇️ trades.csv (sequential log, Brief §10.4)", f, file_name="trades.csv")

# --------------------------------------------------------------------------- #
with tab_team:
    st.markdown("Every agent = **a role prompt** + **inputs from Python tools** + **a JSON answer** that Python "
                "validates. Hard limits live in code; the Trader is pure rules.")
    st.markdown("**How our committee maps onto the brief's four agents:** " +
                " · ".join(f"*{k}* → {', '.join(v)}" for k, v in BRIEF_ROLES.items()))
    for cls in ROSTER:
        a = cls()
        with st.expander(f"{AVATARS.get(a.name, '🤖')} {a.name} — {a.title}  ·  {a.tier}"):
            st.markdown(a.role_prompt)
            st.code(a.output_schema, language="json")
