"""
Reporting (Brief §1, §10.5, §11 · IPS E2–E5). Reads only the database: no agents, no LLM.

    daily_report(day)   reports/daily_<date>.md  holdings, P&L vs SPY, trades + reasons,
                        CIO minutes, risk metrics, and the forum post block (17:00–18:00 Madrid)
    final_summary()     reports/final_summary.docx  the 15-day Word summary (Brief §11)

Terminal:  python report.py [--day YYYY-MM-DD] [--final]
"""
from __future__ import annotations

import json
import os

import pandas as pd

import config
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
        lines.append(f"{r.ticker:<6} {int(r.shares):>6} sh  {r.weight:>5.1%}  P&L {r.pnl:+,.0f} ({r.pnl_pct:+.1%})")
    lines.append(f"Cash ${db.get_cash():,.0f} ({db.get_cash() / nav:.1%})")
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
        md += ["| hour | side | ticker | shares | price | total | reason |", "|---|---|---|---|---|---|---|"]
        for r in t_day.itertuples():
            md.append(f"| {r.hour} | {r.action} | {r.symbol} | {r.shares} | {r.price:,.2f} | {r.total:,.0f} | "
                      f"{str(r.reason).replace('|', '/')} |")
        md.append("")

    md += ["## Committee minutes", ""]
    for r in r_day.itertuples() if not r_day.empty else []:
        md.append(f"**Run #{r.id}** ({r.started_at[11:16]}, {r.mode}, {r.status}): {r.summary or '—'}")
        for m in db.get_messages(r.id):
            c = m["content"]
            if m["agent"] == "Macro Strategist":
                md.append(f"- Macro: regime **{c.get('regime')}** ({c.get('regime_note', '')}). {c.get('rationale', '')}")
            elif m["agent"] == "Risk Manager" and m["stage"] in ("risk_review", "risk_override"):
                last_risk = c
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
            md.append(f"- **{r.ticker} {r.action}** ({r.verdict}, {r.excess_ret:+.1%} vs SPY): {r.lesson}")
        if refl.empty or refl[refl["eval_date"] == day].empty:
            md.append("No calls graded today; latest lessons:")
            for r in refl.head(3).itertuples():
                md.append(f"- **{r.ticker} {r.action}** ({r.verdict}): {r.lesson}")
    else:
        md.append("No calls graded yet (the first become due 5 trading days after the first trades).")
    md.append("")

    h = _holdings_table(_last_prices(day), nav)
    md += ["## Holdings", ""]
    if h.empty:
        md += ["All cash.", ""]
    else:
        md += ["| ticker | shares | avg cost | price | value | weight | P&L |", "|---|---|---|---|---|---|---|"]
        for r in h.itertuples():
            md.append(f"| {r.ticker} | {int(r.shares)} | {r.avg_cost:,.2f} | {r.price:,.2f} | {r.value:,.0f} | "
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
            doc.add_paragraph(f"{r.hour} {r.action} {r.shares} {r.symbol} @ ${r.price:,.2f} (${r.total:,.0f}) — {r.reason}",
                              style="List Bullet")

    doc.add_heading("Committee decisions", 1)
    runs = db.get_runs().sort_values("id")
    for r in runs.itertuples():
        if r.summary:
            doc.add_paragraph(f"#{r.id} {r.started_at[:16]} ({r.mode}, {r.status}): {r.summary}")
    doc.save(path)
    return path


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
