"""
SQLite persistence, following the brief's schema (§8.1) plus the committee minutes.

Brief tables:  simulation, positions, transactions, daily_snapshots
Extra tables:  runs, messages (every agent message, replayable), regime_state

Times are Madrid time (the competition clock). Live fund = finance.db; a
synthetic-data day writes to finance_synthetic_sandbox.db instead.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import pandas as pd

import config

MADRID = ZoneInfo("Europe/Madrid")

SCHEMA = """
CREATE TABLE IF NOT EXISTS simulation (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_date TEXT, cash_balance REAL, total_portfolio_value REAL, inception_date TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, shares INTEGER, average_cost REAL, opened_date TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, hour TEXT, action TEXT, symbol TEXT, shares INTEGER, price REAL,
    total REAL, reason TEXT, run_id INTEGER, cost REAL, name TEXT, signals TEXT, proposed_at TEXT,
    price_source TEXT, price_time TEXT
);
CREATE TABLE IF NOT EXISTS daily_snapshots (
    date TEXT PRIMARY KEY, ts TEXT, total_equity REAL, cash REAL, invested REAL,
    benchmark_px REAL, n_positions INTEGER, run_id INTEGER
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT, finished_at TEXT, provider TEXT, data_source TEXT,
    status TEXT, summary TEXT, mode TEXT, proposal TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, ts TEXT, agent TEXT, title TEXT, model TEXT,
    stage TEXT, round INTEGER, content TEXT
);
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at TEXT, run_id INTEGER, ticker TEXT, action TEXT, entry_px REAL,
    thesis TEXT, invalidation TEXT, invalidation_px REAL, horizon_days INTEGER DEFAULT 5,
    entry_date TEXT, regime TEXT, sector TEXT, signals TEXT,
    eval_date TEXT, ret REAL, spy_ret REAL, excess_ret REAL, verdict TEXT, lesson TEXT, tags TEXT
);
CREATE TABLE IF NOT EXISTS regime_state (
    date TEXT PRIMARY KEY, raw TEXT, effective TEXT, note TEXT
);
"""

_DB = {"path": config.DB_PATH}


def use_synthetic(flag: bool) -> None:
    _DB["path"] = config.DB_PATH.replace(".db", "_synthetic_sandbox.db") if flag else config.DB_PATH


def current_db_name() -> str:
    import os
    return os.path.basename(_DB["path"])


def now() -> dt.datetime:
    return dt.datetime.now(MADRID)


def _now() -> str:
    return now().isoformat(timespec="seconds")


ET = ZoneInfo("America/New_York")


def madrid_et(ts: str | None) -> str:
    """'2026-10-20 15:34 Madrid (09:34 ET)' from a stored ISO timestamp (IPS D13)."""
    if not ts:
        return ""
    try:
        d = dt.datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if d.tzinfo is None:
        d = d.replace(tzinfo=MADRID)
    m = d.astimezone(MADRID)
    return f"{m:%Y-%m-%d %H:%M} Madrid ({m.astimezone(ET):%H:%M} ET)"


def today() -> str:
    return now().date().isoformat()


@contextmanager
def conn():
    c = sqlite3.connect(_DB["path"])
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(reset: bool = False) -> None:
    with conn() as c:
        if reset:
            for t in ("simulation", "positions", "transactions", "daily_snapshots",
                      "runs", "messages", "regime_state", "reflections"):
                c.execute(f"DROP TABLE IF EXISTS {t}")
        c.executescript(SCHEMA)
        # forward-compatible migrations for DBs created before Oct 20
        cols = {r[1] for r in c.execute("PRAGMA table_info(transactions)")}
        for col in ("name", "signals", "proposed_at", "price_source", "price_time"):
            if col not in cols:
                c.execute(f"ALTER TABLE transactions ADD COLUMN {col} TEXT")
        if "price_sources" not in {r[1] for r in c.execute("PRAGMA table_info(daily_snapshots)")}:
            c.execute("ALTER TABLE daily_snapshots ADD COLUMN price_sources TEXT")   # Oct 21: per-holding source JSON
        if "kind" not in {r[1] for r in c.execute("PRAGMA table_info(daily_snapshots)")}:
            c.execute("ALTER TABLE daily_snapshots ADD COLUMN kind TEXT DEFAULT 'intraday'")   # 'close' = official close
        mcols = {r[1] for r in c.execute("PRAGMA table_info(messages)")}
        for col, typ in (("system_prompt", "TEXT"), ("user_prompt", "TEXT"), ("raw_output", "TEXT"),
                         ("tokens", "INTEGER"), ("latency_s", "REAL")):
            if col not in mcols:
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {typ}")
        if c.execute("SELECT COUNT(*) FROM simulation").fetchone()[0] == 0:
            c.execute("INSERT INTO simulation (id, current_date, cash_balance, total_portfolio_value, inception_date) "
                      "VALUES (1, ?, ?, ?, NULL)", (today(), config.STARTING_CAPITAL, config.STARTING_CAPITAL))


# ---------------------------------------------------------------- runs ---- #
def start_run(provider: str, data_source: str, mode: str) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO runs (started_at, provider, data_source, status, mode) VALUES (?,?,?,?,?)",
                        (_now(), provider, data_source, "running", mode))
        return cur.lastrowid


def finish_run(run_id: int, status: str, summary: str, proposal: dict | None = None) -> None:
    with conn() as c:
        if proposal is not None:
            c.execute("UPDATE runs SET finished_at=?, status=?, summary=?, proposal=? WHERE id=?",
                      (_now(), status, summary, json.dumps(proposal, default=str), run_id))
        else:
            c.execute("UPDATE runs SET finished_at=?, status=?, summary=? WHERE id=?",
                      (_now(), status, summary, run_id))


def claim_run(run_id: int, frm: str = "proposed", to: str = "executing") -> bool:
    """Atomic compare-and-set on a run's status: only one caller can move it from `frm` to `to`.
    Stops the same proposal executing twice (Streamlit rerun, double click, two tabs)."""
    with conn() as c:
        return c.execute("UPDATE runs SET status=? WHERE id=? AND status=?", (to, run_id, frm)).rowcount == 1


def set_run_status(run_id: int, status: str) -> None:
    with conn() as c:
        c.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))


def get_proposal(run_id: int) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT proposal FROM runs WHERE id=?", (run_id,)).fetchone()
    return json.loads(row["proposal"]) if row and row["proposal"] else None


def runs_today() -> int:
    with conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM runs WHERE substr(started_at,1,10)=? AND status!='error'",
                             (today(),)).fetchone()[0])


def mark_stale_runs() -> None:
    """Runs that crashed without finishing (e.g. Oct 11 run #1) are marked as errors."""
    with conn() as c:
        c.execute("UPDATE runs SET status='error', summary=COALESCE(summary,'crashed before finishing') "
                  "WHERE status='running' AND started_at < ?", ((now() - dt.timedelta(hours=1)).isoformat(),))


def log_message(run_id: int, agent: str, title: str, model: str, stage: str,
                rnd: int, content: dict, prompt: dict | None = None) -> None:
    """prompt (Stage 8): {system, user, raw, tokens, latency_s} -> 'what this agent saw' in the dashboard."""
    p = prompt or {}
    with conn() as c:
        c.execute("INSERT INTO messages (run_id, ts, agent, title, model, stage, round, content, system_prompt, "
                  "user_prompt, raw_output, tokens, latency_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (run_id, _now(), agent, title, model, stage, rnd, json.dumps(content, default=str),
                   p.get("system"), p.get("user"), p.get("raw"), p.get("tokens"), p.get("latency_s")))


def get_runs() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT id, started_at, finished_at, provider, data_source, status, summary, mode "
                           "FROM runs ORDER BY id DESC", c)


def get_messages(run_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM messages WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    return [{**dict(r), "content": json.loads(r["content"])} for r in rows]


# ----------------------------------------------------------- portfolio ---- #
def get_cash() -> float:
    with conn() as c:
        return float(c.execute("SELECT cash_balance FROM simulation WHERE id=1").fetchone()[0])


def is_initialized() -> bool:
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 0


def get_positions() -> dict[str, dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM positions WHERE shares > 0").fetchall()
    return {r["symbol"]: {"shares": int(r["shares"]), "avg_cost": float(r["average_cost"]),
                          "opened_date": r["opened_date"]} for r in rows}


def current_weights(last_px: dict[str, float]) -> tuple[dict[str, float], float]:
    pos = get_positions()
    cash = get_cash()
    values = {t: p["shares"] * last_px[t] for t, p in pos.items() if t in last_px}
    nav = cash + sum(values.values())
    return ({t: v / nav for t, v in values.items()}, nav)


def trades_today() -> int:
    with conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM transactions WHERE date=?", (today(),)).fetchone()[0])


def execute_trade(run_id: int, symbol: str, side: str, shares: int, price: float, reason: str,
                  meta: dict | None = None, name: str | None = None, signals: str | None = None,
                  proposed_at: str | None = None, price_source: str | None = None,
                  price_time: str | None = None) -> dict:
    shares = int(shares)
    t = now()
    with conn() as c:
        row = c.execute("SELECT shares, average_cost, opened_date FROM positions WHERE symbol=?", (symbol,)).fetchone()
        old_sh, old_cost, opened = (int(row["shares"]), float(row["average_cost"]), row["opened_date"]) \
            if row else (0, 0.0, None)
        cash = float(c.execute("SELECT cash_balance FROM simulation WHERE id=1").fetchone()[0])
        if side == "SELL":
            shares = min(shares, old_sh)
        total = shares * price
        cost = total * config.TRANSACTION_COST_BPS / 1e4
        if side == "BUY":
            if total + cost > cash + 1e-6:
                raise ValueError(f"BUY {symbol} needs ${total:,.0f} but cash is ${cash:,.0f} (Brief: cash >= 0)")
            new_sh = old_sh + shares
            new_cost = (old_sh * old_cost + total) / new_sh
            opened = opened if old_sh > 0 else t.date().isoformat()
            cash -= total + cost
        else:
            new_sh, new_cost = old_sh - shares, old_cost
            cash += total - cost
            if new_sh == 0:
                opened = None
        c.execute("INSERT INTO positions (symbol, shares, average_cost, opened_date) VALUES (?,?,?,?) "
                  "ON CONFLICT(symbol) DO UPDATE SET shares=excluded.shares, average_cost=excluded.average_cost, "
                  "opened_date=excluded.opened_date", (symbol, new_sh, new_cost, opened))
        c.execute("UPDATE simulation SET cash_balance=?, current_date=?, "
                  "inception_date=COALESCE(inception_date, ?) WHERE id=1",
                  (cash, t.date().isoformat(), t.date().isoformat()))
        c.execute("INSERT INTO transactions (date, hour, action, symbol, shares, price, total, reason, run_id, cost, "
                  "name, signals, proposed_at, price_source, price_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (t.date().isoformat(), t.strftime("%H:%M:%S"), side, symbol, shares, round(price, 4),
                   round(total, 2), reason, run_id, cost, name, signals, proposed_at, price_source, price_time))
        tid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        if meta is not None and shares > 0:    # every executed trade is a call the committee will be graded on
            c.execute("INSERT INTO reflections (decided_at, run_id, ticker, action, entry_px, thesis, invalidation, "
                      "invalidation_px, horizon_days, entry_date, regime, sector, signals) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (_now(), run_id, symbol, side, price, meta.get("thesis", ""), meta.get("invalidation", ""),
                       meta.get("invalidation_px"), meta.get("horizon_days", 5), meta.get("entry_date"),
                       meta.get("regime"), meta.get("sector"), json.dumps(meta.get("signals", {}), default=str)))
    return {"symbol": symbol, "side": side, "shares": shares, "price": price, "total": total, "id": tid}


def record_snapshot(run_id: int | None, last_px: dict[str, float], benchmark_px: float,
                    price_sources: dict | None = None, day: str | None = None, kind: str = "intraday") -> float:
    """kind='close' = official NYSE close of `day` (never overwritten by an intraday mark); 'intraday' = today's mark."""
    pos = get_positions()
    cash = get_cash()
    invested = sum(p["shares"] * last_px[t] for t, p in pos.items() if t in last_px)
    nav = cash + invested
    import nyse_calendar
    d = day or today()
    if not nyse_calendar.is_trading_day(d):          # Stage 8: never log a phantom weekend/holiday day
        return nav
    if kind == "intraday" and has_close(d):
        return nav                                    # the official close row wins
    with conn() as c:
        c.execute("INSERT INTO daily_snapshots (date, ts, total_equity, cash, invested, benchmark_px, n_positions, run_id, kind) "
                  "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(date) DO UPDATE SET ts=excluded.ts, "
                  "total_equity=excluded.total_equity, cash=excluded.cash, invested=excluded.invested, "
                  "benchmark_px=excluded.benchmark_px, n_positions=excluded.n_positions, run_id=excluded.run_id, "
                  "kind=excluded.kind",
                  (d, _now(), nav, cash, invested, benchmark_px, len(pos), run_id, kind))
        if price_sources is not None:
            c.execute("UPDATE daily_snapshots SET price_sources=? WHERE date=?", (json.dumps(price_sources), d))
        if day is None:
            c.execute("UPDATE simulation SET total_portfolio_value=?, current_date=? WHERE id=1", (nav, today()))
    return nav


def has_close(day: str) -> bool:
    with conn() as c:
        r = c.execute("SELECT kind FROM daily_snapshots WHERE date=?", (day,)).fetchone()
    return bool(r) and r[0] == "close"


def trades_after(ts: dt.datetime) -> int:
    """Trades executed after an aware datetime (transactions store Madrid date + hour)."""
    cut = ts.astimezone(MADRID).strftime("%Y-%m-%d %H:%M:%S")
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM transactions WHERE date || ' ' || hour > ?", (cut,)).fetchone()[0]


def get_trades() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM transactions ORDER BY id DESC", c)


def get_snapshots() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM daily_snapshots ORDER BY date", c)


def last_snapshot_equity() -> float | None:
    with conn() as c:
        row = c.execute("SELECT total_equity FROM daily_snapshots ORDER BY date DESC LIMIT 1").fetchone()
    return float(row[0]) if row else None


def drawdown_from_peak(nav: float) -> float:
    s = get_snapshots()
    peak = max([config.STARTING_CAPITAL, nav] + s["total_equity"].tolist())
    return nav / peak - 1


def held_trading_days(symbol: str, opened_date: str | None) -> int:
    if not opened_date:
        return 0
    return max(0, len(pd.bdate_range(opened_date, today())) - 1)


# -------------------------------------------------------- regime state ---- #
def previous_regime() -> tuple[str | None, str | None]:
    """(effective, raw) from the last trading day before today."""
    with conn() as c:
        row = c.execute("SELECT effective, raw FROM regime_state WHERE date < ? ORDER BY date DESC LIMIT 1",
                        (today(),)).fetchone()
    return (row["effective"], row["raw"]) if row else (None, None)


def save_regime(raw: str, effective: str, note: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO regime_state (date, raw, effective, note) VALUES (?,?,?,?) "
                  "ON CONFLICT(date) DO UPDATE SET raw=excluded.raw, effective=excluded.effective, note=excluded.note",
                  (today(), raw, effective, note))


# --------------------------------------------------------------- export --- #
def set_trade_reason(trade_id: int, reason: str) -> None:
    with conn() as c:
        c.execute("UPDATE transactions SET reason=? WHERE id=?", (reason, trade_id))


def export_trades_csv(path: str | None = None) -> str:
    """Brief §10.4 sequential CSV of every trade. IPS E1 (Oct 20) columns: date, hour (Madrid), ticker, name,
    side, price, shares, total, reason (plain English, E2), signals (scores, kept out of the reason)."""
    path = path or config.TRADES_CSV
    with conn() as c:
        rows = c.execute("SELECT date, hour, symbol, name, action, price, shares, total, reason, signals, "
                         "proposed_at, price_source, price_time "
                         "FROM transactions ORDER BY id").fetchall()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "hour_madrid", "ticker", "name", "side", "price", "shares", "total", "reason", "signals",
                    "decided_at", "price_source", "price_time"])
        for r in rows:
            w.writerow([r["date"], r["hour"], r["symbol"], r["name"] or "", r["action"], f"{r['price']:.2f}",
                        r["shares"], f"{r['total']:.2f}", r["reason"], r["signals"] or "",
                        r["proposed_at"] or "", r["price_source"] or "", r["price_time"] or ""])
    return path


# ---------------------------------------------------------- reflections --- #
def pending_reflections() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM reflections WHERE eval_date IS NULL ORDER BY id")]


def finish_reflection(rid: int, eval_date: str, ret: float, spy_ret: float, excess: float,
                      verdict: str, lesson: str, tags: list[str]) -> None:
    with conn() as c:
        c.execute("UPDATE reflections SET eval_date=?, ret=?, spy_ret=?, excess_ret=?, verdict=?, lesson=?, tags=? "
                  "WHERE id=?", (eval_date, ret, spy_ret, excess, verdict, lesson, json.dumps(tags), rid))


def get_reflections(evaluated_only: bool = False) -> pd.DataFrame:
    with conn() as c:
        q = "SELECT * FROM reflections" + (" WHERE eval_date IS NOT NULL" if evaluated_only else "") + " ORDER BY id DESC"
        return pd.read_sql(q, c)


def latest_buy_reflection(ticker: str) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM reflections WHERE ticker=? AND action='BUY' ORDER BY id DESC LIMIT 1",
                        (ticker,)).fetchone()
    return dict(row) if row else None
