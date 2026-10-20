"""
Reporting (Brief §1, §10.5, §11 · IPS E2–E5). Reads only the database: no agents, no LLM.

    daily_report(day)   reports/daily_<date>.md  holdings, P&L vs SPY, trades + reasons,
                        CIO minutes, risk metrics, and the forum post block (17:00–18:00 Madrid)
    final_summary()     reports/final_summary.docx  the 15-day Word summary (Brief §11)

Terminal:  python report.py [--day YYYY-MM-DD] [--final]
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd

import config
import data
import database as db

REPORT_DIR = os.getenv("FUND_REPORT_DIR", os.path.join(config.BASE_DIR, "reports"))


def _last_prices(day: str) -> dict[str, float]:
    """Price of each holding as of `day`: prices.csv if present, else the run's stored prices."""
    px = {}
    if os.path.exists(config.PRICES_CSV):
        p = pd.read_csv(config.PRICES_CSV, index_col="date", parse_dates=True)
        p = p[p.index <= pd.Timestamp(day) + pd.Timedelta(days=1)]
        if len(p):
            px = p.ffill().iloc[-1].dropna().to_dict()
    with db.conn() as c:
        row = c.execute("SELECT proposal FROM runs WHERE substr(started_at,1,10)<=? AND proposal IS NOT NULL "
                        "ORDER BY id DESC LIMIT 1", (day,)).fetchone()
    if row:
        px = {**json.loads(row["proposal"]).get("last_px", {}), **px} if not px else px
    return px


def _kpis(snaps: pd.DataFrame) -> dict:
    if snaps.empty:
        return {}
    eq = snaps["total_equity"]
    peak = eq.cummax().clip(lower=config.STARTING_CAPITAL)
    daily = eq.pct_change().dropna()
    spy = snaps["benchmark_px"]
    return {"nav": float(eq.iloc[-1]),
            "total_return": float(eq.iloc[-1] / config.STARTING_CAPITAL - 1),
            "spy_return": float(spy.iloc[-1] / spy.iloc[0] - 1) if spy.iloc[0] else 0.0,
            "max_drawdown": float((eq / peak - 1).min()),
            "win_rate_days": float((daily > 0).mean()) if len(daily) else None,
            "days": int(len(snaps))}


def _holdings_table(px: dict[str, float], nav: float) -> pd.DataFrame:
    rows = []
    for t, p in db.get_positions().items():
        last = float(px.get(t, p["avg_cost"]))
        mv = p["shares"] * last
        rows.append({"ticker": t, "shares": p["shares"], "avg_cost": p["avg_cost"], "price": last,
                     "value": mv, "weight": mv / nav if nav else 0, "pnl": p["shares"] * (last - p["avg_cost"]),
                     "pnl_pct": last / p["avg_cost"] - 1})
    return pd.DataFrame(rows).sort_values("weight", ascending=False) if rows else pd.DataFrame()


def _day_messages(day: str) -> dict:
    """Messages of the day's last committee run, keyed by agent (and stage for the PM)."""
    runs = db.get_runs()
    if runs.empty:
        return {}
    r = runs[(runs["started_at"].str[:10] == day) & (runs["mode"] != "recheck")].sort_values("id")
    if r.empty:
        return {}
    out = {}
    for m in db.get_messages(int(r.iloc[-1]["id"])):
        key = m["agent"] if m["agent"] != "Portfolio Manager" else f"PM:{m['stage']}"
        out[key] = m
        if m["agent"] == "Risk Manager":
            out["Risk Manager:last"] = m
        out.setdefault("_all", []).append(m)
    return out


def attribution_result():
    """Brinson-Fachler vs SPY for the current fund; None until there are two snapshots after the first trade."""
    try:
        import attribution
        u = data.get_universe()
        close = data.get_market_data(u["symbol"].tolist())["close"]
        return attribution.run(close, data.sector_etf_close(), dict(zip(u["symbol"], u["sector"])),
                               dict(zip(u["symbol"], u["asset_type"])), db.get_trades(), db.get_snapshots(),
                               db.get_reflections())
    except Exception:  # noqa: BLE001
        return None


def forum_post(day: str | None = None) -> str:
    """IPS E4: ticker · shares · weight · P&L, plus NAV and return since inception."""
    day = day or db.today()
    snaps = db.get_snapshots()
    k = _kpis(snaps[snaps["date"] <= day])
    nav = k.get("nav", config.STARTING_CAPITAL)
    h = _holdings_table(_last_prices(day), nav)
    lines = [f"{config.FUND_NAME} — portfolio as of {day}",
             f"NAV ${nav:,.0f} | since inception {k.get('total_return', 0):+.2%} | SPY {k.get('spy_return', 0):+.2%}",
             ""]
    for r in h.itertuples():
        lines.append(f"{data.label(r.ticker)[:34]:<34} {int(r.shares):>6} sh  {r.weight:>5.1%}  P&L {r.pnl:+,.0f} ({r.pnl_pct:+.1%})")
    lines.append(f"Cash ${db.get_cash():,.0f} ({db.get_cash() / nav:.1%})")
    # IPS E4 (Oct 20): + 3 lines
    tr = db.get_trades()
    t_day = tr[tr["date"] == day] if not tr.empty else tr
    if t_day is None or t_day.empty:
        l1 = "Today: no trades (nothing cleared the bar; doing nothing is a valid decision)."
    else:
        bits = [f"{'bought' if r.action == 'BUY' else 'sold'} {data.label(r.symbol).split(' · ')[-1]}" for r in t_day.itertuples()]
        first = str(t_day.iloc[0]["reason"]).split(". ")[0][:140]
        l1 = f"Today: {', '.join(bits)}. Lead reason: {first}."
    l2 = (f"Return since inception {k.get('total_return', 0):+.2%} vs SPY {k.get('spy_return', 0):+.2%} "
          f"({(k.get('total_return', 0) - k.get('spy_return', 0)) * 1e4:+.0f} bps).")
    msgs = _day_messages(day)
    brief = (msgs.get("Morning Brief") or {}).get("content") or {}
    import morning_brief
    nxt = [f"{e['date'][5:]} {e['event'].split(',')[0]}" for e in morning_brief.macro_events(dt.date.fromisoformat(day), 5)]
    watch = list(brief.get("tickers_in_play") or [])[:4]
    l3 = "Tomorrow's watch list: " + "; ".join(([", ".join(watch)] if watch else []) + nxt[:2] or ["no scheduled catalysts"]) + "."
    lines += ["", l1, l2, l3]
    return "\n".join(lines)


def daily_report(day: str | None = None) -> str:
    day = day or db.today()
    os.makedirs(REPORT_DIR, exist_ok=True)
    snaps = db.get_snapshots()
    k = _kpis(snaps[snaps["date"] <= day])
    nav = k.get("nav", config.STARTING_CAPITAL)
    trades = db.get_trades()
    t_day = trades[trades["date"] == day].sort_values("id") if not trades.empty else trades
    runs = db.get_runs()
    r_day = runs[runs["started_at"].str[:10] == day].sort_values("id") if not runs.empty else runs

    md = [f"# Daily report — {day}", "",
          f"**NAV** ${nav:,.0f} · **since inception** {k.get('total_return', 0):+.2%} "
          f"(SPY {k.get('spy_return', 0):+.2%}) · **max drawdown** {k.get('max_drawdown', 0):.2%} · "
          f"**trades today** {len(t_day)}", ""]

    md += ["## Trades and reasoning", ""]
    if t_day is None or t_day.empty:
        md += ["No trades today (a no-trade day is a valid decision; see the committee minutes).", ""]
    else:
        md += ["| time (Madrid / ET) | side | name | shares | price | total | reason | signals |",
               "|---|---|---|---|---|---|---|---|"]
        for r in t_day.itertuples():
            when = db.madrid_et(f"{r.date}T{r.hour}").split(" ", 1)[-1]
            md.append(f"| {when} | {r.action} | {data.label(r.symbol)} | {r.shares} | {r.price:,.2f} | {r.total:,.0f} | "
                      f"{str(r.reason).replace('|', '/')} | {str(getattr(r, 'signals', '') or '').replace('|', '/')} |")
        md.append("")

    md += ["## Committee minutes", ""]
    for r in r_day.itertuples() if not r_day.empty else []:
        md.append(f"**Run #{r.id}** ({db.madrid_et(r.started_at)}, {r.mode}, {r.status}): {r.summary or '—'}")
        for m in db.get_messages(r.id):
            c = m["content"]
            if m["agent"] == "Macro Strategist":
                md.append(f"- Macro: regime **{c.get('regime')}** ({c.get('regime_note', '')}). {c.get('rationale', '')}")
            elif m["agent"] == "Risk Manager" and m["stage"] in ("risk_review", "risk_override"):
                last_risk = c
            elif m["agent"] == "System" and "scoreboard" in c:
                md.append("- Model horse race (same out-of-sample origins; weights per IPS C3.4):")
                md.append("")
                md.append("  | model | hit rate | rank IC | IC t-stat | trust hit (walk-forward) | always-up | weight |")
                md.append("  |---|---|---|---|---|---|---|")
                f = lambda v, fmt: "–" if v is None else fmt.format(v)
                for mname, sc in c["scoreboard"].items():
                    md.append(f"  | {mname} | {f(sc.get('hit'), '{:.1%}')} | {f(sc.get('ic'), '{:+.3f}')} | "
                              f"{f(sc.get('ic_t'), '{:+.2f}')} | {f(sc.get('trust_hit'), '{:.1%}')} | "
                              f"{f(sc.get('always_up'), '{:.1%}')} | {f((c.get('weights') or {}).get(mname), '{:.0%}')} |")
                md.append("")
            elif m["agent"] == "Quant Analyst" and c.get("model_scoreboard"):
                md.append(f"- Models: {c['model_scoreboard']}")
            elif m["agent"] == "CIO":
                if c.get("key_risks"):
                    md.append(f"- Key risks: {'; '.join(c['key_risks'])}")
        if "last_risk" in locals():
            md.append(f"- Risk: {last_risk.get('decision')} — {last_risk.get('rationale', '')}")
            del last_risk
        md.append("")
    if r_day.empty:
        md += ["No committee run today.", ""]

    md += ["## Lessons learned (reflection memory)", ""]
    import reflection
    rec = reflection.track_record()
    refl = db.get_reflections(evaluated_only=True)
    if rec.get("calls"):
        md.append(f"Track record: {rec['calls']} graded calls · hit rate {rec['hit_rate']} (last 10: {rec['last10_hit_rate']}) · "
                  f"avg excess vs SPY {rec['avg_excess_ret']:+.2%}")
        md.append("")
        for r in refl[refl["eval_date"] == day].itertuples() if not refl.empty else []:
            md.append(f"- **{data.label(r.ticker)} {r.action}** ({r.verdict}, {r.excess_ret:+.1%} vs SPY): {r.lesson}")
        if refl.empty or refl[refl["eval_date"] == day].empty:
            md.append("No calls graded today; latest lessons:")
            for r in refl.head(3).itertuples():
                md.append(f"- **{r.ticker} {r.action}** ({r.verdict}): {r.lesson}")
    else:
        md.append("No calls graded yet (the first become due 5 trading days after the first trades).")
    md.append("")

    msgs = _day_messages(day)
    brief = (msgs.get("Morning Brief") or {}).get("content") or {}
    if brief.get("text"):
        md += ["## Morning Brief", "", f"*{brief.get('model', '')}*", "", brief["text"], "",
               f"Tickers in play: {', '.join(brief.get('tickers_in_play') or []) or '—'}", ""]
    da = (msgs.get("Devil's Advocate") or {}).get("content") or {}
    if da.get("objections"):
        ans = ((msgs.get("PM:proposal_answers_da") or {}).get("content") or {}).get("da_responses") or {}
        md += ["## Devil's Advocate vs the PM", "", "| idea | bear case | evidence | DA verdict | PM answer |", "|---|---|---|---|---|"]
        for t, o in da["objections"].items():
            md.append(f"| {data.label(t)} | {o['bear_case']} | {o['evidence']} | {o['verdict']} | {ans.get(t, '—')} |")
        md.append("")
    sent = ((msgs.get("Sentiment Analyst") or {}).get("content") or {}).get("sentiment") or {}
    held = list(db.get_positions())
    if sent and held:
        md += ["## Top headlines on held names", ""]
        for t in held:
            v = sent.get(t) or {}
            if v.get("based_on") or v.get("key_driver"):
                sc = "n/a" if v.get("score") is None else f"{v['score']:+.2f}"
                md.append(f"- **{data.label(t)}** (S {sc}, FinBERT {v.get('finbert')}): {v.get('key_driver', '')}")
                for h_ in (v.get("based_on") or [])[:2]:
                    md.append(f"  - {h_}")
        md.append("")
    risk = (msgs.get("Risk Manager:last") or {}).get("content") or {}
    if risk.get("limit_checks"):
        md += ["## Risk Manager: hard-limit checks (post-trade book)", "",
               f"Verdict: **{risk.get('decision', '').upper()}**" + (f" · override: {risk['override']}" if risk.get("override") else ""),
               "", "| rule | value | limit | kind | result |", "|---|---|---|---|---|"]
        for c in risk["limit_checks"]:
            md.append(f"| {c['rule']} | {c['value']} | {c['limit']} | {c['kind']} | {'✅' if c['status'] == 'pass' else '❌'} |")
        for w_ in risk.get("warnings", []):
            md.append(f"- warning: {w_}")
        md.append("")
    fails = [m for m in msgs.get("_all", []) if str(m.get("model", "")).startswith("FAILED")]
    md += ["## Agent failures (D11)", ""] + ([f"- {m['agent']} ({m['stage']}): {m['model']}" for m in fails] or ["None today."]) + [""]
    att = attribution_result()
    if att:
        md += ["## Performance attribution vs SPY (Brinson-Fachler, Carino-linked)", "", att["line"],
               f"(holdings-based; execution/timing vs NAV {att['cumulative']['execution_timing']:+.2%}; "
               f"SPY sector weights as of {att['spy_weights_asof']})", ""]
        if att.get("by_agent"):
            md.append("Selection by driving agent: " + ", ".join(f"{a} {v:+.2%}" for a, v in att["by_agent"].items()))
            md.append("")

    h = _holdings_table(_last_prices(day), nav)
    md += ["## Holdings", ""]
    if h.empty:
        md += ["All cash.", ""]
    else:
        md += ["| ticker | shares | avg cost | price | value | weight | P&L |", "|---|---|---|---|---|---|---|"]
        for r in h.itertuples():
            md.append(f"| {data.label(r.ticker)} | {int(r.shares)} | {r.avg_cost:,.2f} | {r.price:,.2f} | {r.value:,.0f} | "
                      f"{r.weight:.1%} | {r.pnl:+,.0f} ({r.pnl_pct:+.1%}) |")
        md.append("")

    md += ["## Forum post (17:00–18:00 Madrid)", "", "```", forum_post(day), "```", ""]
    path = os.path.join(REPORT_DIR, f"daily_{day}.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def final_summary(path: str | None = None) -> str:
    """Brief §11: Word document tracking the whole evolution of the portfolio."""
    from docx import Document
    from docx.shared import Pt

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = path or os.path.join(REPORT_DIR, "final_summary.docx")
    snaps = db.get_snapshots()
    trades = db.get_trades().sort_values("id") if not db.get_trades().empty else db.get_trades()
    k = _kpis(snaps)
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(f"{config.FUND_NAME} — competition summary", 0)
    if k:
        doc.add_paragraph(f"{snaps['date'].iloc[0]} to {snaps['date'].iloc[-1]} · {k['days']} daily snapshots")
        t = doc.add_table(rows=1, cols=2)
        t.style = "Light Grid Accent 1"
        t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Metric", "Value"
        for name, val in [("Final NAV", f"${k['nav']:,.0f}"), ("Total return", f"{k['total_return']:+.2%}"),
                          ("SPY over the same period", f"{k['spy_return']:+.2%}"),
                          ("Excess return vs SPY", f"{k['total_return'] - k['spy_return']:+.2%}"),
                          ("Max drawdown", f"{k['max_drawdown']:.2%}"),
                          ("Win rate (up days)", f"{k['win_rate_days']:.0%}" if k["win_rate_days"] is not None else "–"),
                          ("Trades executed", f"{len(trades)}")]:
            row = t.add_row().cells
            row[0].text, row[1].text = name, val
    else:
        doc.add_paragraph("No snapshots recorded yet.")

    from docx.shared import Inches
    import tempfile
    charts = _charts(snaps)
    if charts.get("nav"):
        doc.add_heading("NAV vs SPY", 1)
        doc.add_picture(charts["nav"], width=Inches(6.2))

    att = attribution_result()
    doc.add_heading("Performance attribution vs SPY", 1)
    if att:
        doc.add_paragraph(att["line"])
        p_ = att["cumulative"]["parts"]
        doc.add_paragraph(f"Brinson-Fachler on daily holdings, linked with Carino smoothing (parts sum to the total). "
                          f"Execution/timing vs the NAV return: {att['cumulative']['execution_timing']:+.2%}. "
                          f"SPY sector weights as of {att['spy_weights_asof']}.")
        wf = _waterfall(p_)
        if wf:
            doc.add_picture(wf, width=Inches(6.0))
        if att.get("by_agent"):
            doc.add_paragraph("Stock picking by the agent that drove each position: " +
                              ", ".join(f"{a} {v:+.2%}" for a, v in att["by_agent"].items()))
    else:
        doc.add_paragraph("Not enough history yet (needs two snapshots after the first trade).")

    doc.add_heading("Graded calls (reflection memory)", 1)
    import reflection
    rec = reflection.track_record()
    refl = db.get_reflections(evaluated_only=True)
    if rec.get("calls"):
        doc.add_paragraph(f"{rec['calls']} calls graded vs SPY over 5 trading days: hit rate {rec['hit_rate']}, "
                          f"average excess {rec['avg_excess_ret']:+.2%}.")
        for r in refl.head(30).itertuples():
            doc.add_paragraph(f"{data.label(r.ticker)} {r.action} ({r.verdict}, {r.excess_ret:+.1%} vs SPY): {r.lesson}",
                              style="List Bullet")
    else:
        doc.add_paragraph("No calls graded yet.")

    doc.add_heading("Model horse race (latest)", 1)
    hr = _latest_horse_race()
    if hr:
        doc.add_paragraph(hr.get("scoreboard_line", ""))
    doc.add_paragraph("Short-horizon returns are close to unpredictable: a model only gets weight if it beats the "
                      "always-up baseline, clears 52% and has an IC t-stat of at least 1 (IPS C3.4).")

    doc.add_heading("What worked / what didn't", 1)
    if att and att["cumulative"].get("per_name"):
        pn = sorted(att["cumulative"]["per_name"].items(), key=lambda kv: kv[1])
        doc.add_paragraph("Best stock picks (selection vs sector): " +
                          ", ".join(f"{data.label(t)} {v:+.2%}" for t, v in pn[::-1][:3]))
        doc.add_paragraph("Worst stock picks: " + ", ".join(f"{data.label(t)} {v:+.2%}" for t, v in pn[:3]))
    p_ = (att or {}).get("cumulative", {}).get("parts", {})
    if p_:
        best = max(p_, key=p_.get); worst = min(p_, key=p_.get)
        doc.add_paragraph(f"Largest positive driver: {best.replace('_', ' ')} ({p_[best]:+.2%}); "
                          f"largest drag: {worst.replace('_', ' ')} ({p_[worst]:+.2%}).")

    doc.add_heading("Limitations", 1)
    for lim in ("15 trading days is mostly noise: treat P&L as anecdotal, not as evidence of skill.",
                "Free data only: yfinance prices (~15 min delayed), Yahoo/Google News RSS and SEC 8-K headlines; "
                "no Bloomberg/FactSet consensus.",
                "The 20-day replay runs on prices only with fundamentals frozen at today's snapshot (small look-ahead), "
                "no historical news (Sentiment n/a) and today's S&P 500 constituents (survivorship bias).",
                "LLM agents can be inconsistent run to run; hard limits are enforced in Python, not by the models.",
                "Attribution uses SPY sector weights from one snapshot and SPDR sector ETFs as sector benchmarks."):
        doc.add_paragraph(lim, style="List Bullet")

    doc.add_heading("Daily evolution", 1)
    if not snaps.empty:
        t = doc.add_table(rows=1, cols=5)
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(["Date", "NAV", "Day P&L", "Cash", "Trades"]):
            t.rows[0].cells[i].text = h
        n_tr = trades.groupby("date").size() if not trades.empty else {}
        prev = config.STARTING_CAPITAL
        for r in snaps.itertuples():
            row = t.add_row().cells
            row[0].text, row[1].text = r.date, f"${r.total_equity:,.0f}"
            row[2].text, row[3].text = f"{r.total_equity - prev:+,.0f}", f"${r.cash:,.0f}"
            row[4].text = str(int(n_tr.get(r.date, 0)) if len(n_tr) else 0)
            prev = r.total_equity

    doc.add_heading("Every trade and its reasoning", 1)
    for day, g in (trades.groupby("date", sort=False) if not trades.empty else []):
        doc.add_heading(day, 2)
        for r in g.itertuples():
            doc.add_paragraph(f"{r.hour} Madrid · {r.action} {r.shares} {data.label(r.symbol)} @ ${r.price:,.2f} (${r.total:,.0f}) — {r.reason}",
                              style="List Bullet")

    doc.add_heading("Committee decisions", 1)
    runs = db.get_runs().sort_values("id")
    for r in runs.itertuples():
        if r.summary:
            doc.add_paragraph(f"#{r.id} {r.started_at[:16]} ({r.mode}, {r.status}): {r.summary}")
    doc.save(path)
    return path


def _charts(snaps: pd.DataFrame) -> dict:
    if snaps is None or len(snaps) < 2:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s_ = snaps.copy()
        s_["date"] = pd.to_datetime(s_["date"])
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
        ax.plot(s_["date"], s_["total_equity"] / config.STARTING_CAPITAL * 100, color="#1F4E9C", lw=2, label="Fund")
        ax.plot(s_["date"], s_["benchmark_px"] / s_["benchmark_px"].iloc[0] * 100, color="#9DB9E3", lw=2, ls="--", label="SPY")
        ax.set_ylabel("rebased to 100"); ax.legend(frameon=False); ax.grid(alpha=0.2)
        fig.autofmt_xdate(); fig.tight_layout()
        path = os.path.join(REPORT_DIR, "nav_vs_spy.png"); fig.savefig(path); plt.close(fig)
        return {"nav": path}
    except Exception:  # noqa: BLE001
        return {}


def _waterfall(parts: dict) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = {"sector_calls": "Sector calls", "benchmark_tracking": "SPY vs sector ETFs", "stock_picking": "Selection",
                  "leverage_beta": "Leverage beta", "leverage_decay": "Leverage decay", "cash_drag": "Cash drag"}
        keys = [k for k in labels if k in parts]
        vals = [parts[k] * 100 for k in keys]
        cum = 0.0
        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        for i, (k, v) in enumerate(zip(keys, vals)):
            ax.bar(i, v, bottom=cum if v >= 0 else cum + v, color="#2E8B57" if v >= 0 else "#C0392B")
            cum += v
        ax.bar(len(keys), cum, color="#1F4E9C")
        ax.set_xticks(range(len(keys) + 1), [labels[k] for k in keys] + ["Total vs SPY"], rotation=20, fontsize=8)
        ax.set_ylabel("% excess return"); ax.axhline(0, color="black", lw=0.6); ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = os.path.join(REPORT_DIR, "attribution_waterfall.png"); fig.savefig(path); plt.close(fig)
        return path
    except Exception:  # noqa: BLE001
        return None


def _latest_horse_race() -> dict | None:
    runs = db.get_runs()
    for rid in runs["id"].tolist() if not runs.empty else []:
        for m in db.get_messages(int(rid)):
            if m["agent"] == "System" and "scoreboard" in m["content"]:
                return m["content"]
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Daily report / final summary")
    ap.add_argument("--day", help="YYYY-MM-DD (default today, Madrid)")
    ap.add_argument("--final", action="store_true", help="build the 15-day Word summary")
    args = ap.parse_args()
    db.init_db()
    print(daily_report(args.day))
    if args.final:
        print(final_summary())
