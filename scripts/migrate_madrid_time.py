"""
One-off migration (IPS D13, Oct 20): make every stored timestamp Europe/Madrid and backfill new columns.

    python scripts/migrate_madrid_time.py path/to/finance.db [--dry-run]

- runs.started_at / finished_at, messages.ts, daily_snapshots.ts: any ISO timestamp with a non-Madrid offset (or no
  offset, assumed UTC) is rewritten in Madrid time with its offset.
- transactions: adds name / signals / proposed_at columns if missing; backfills name from the universe and
  proposed_at from the run's finished_at, so the report can show "proposed hh:mm / executed hh:mm".
Prints what it changed. Idempotent.
"""
import datetime as dt
import os
import sqlite3
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402

MADRID, UTC = ZoneInfo("Europe/Madrid"), dt.timezone.utc


def to_madrid(ts: str) -> str | None:
    if not ts:
        return None
    try:
        d = dt.datetime.fromisoformat(ts)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    m = d.astimezone(MADRID).isoformat(timespec="seconds")
    return None if m == ts else m


def main(path: str, dry: bool) -> None:
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    changed = 0
    for table, cols in (("runs", ("started_at", "finished_at")), ("messages", ("ts",)), ("daily_snapshots", ("ts",))):
        if not c.execute("SELECT name FROM sqlite_master WHERE name=?", (table,)).fetchone():
            continue
        for col in cols:
            for r in c.execute(f"SELECT rowid, {col} FROM {table}").fetchall():
                new = to_madrid(r[col])
                if new:
                    changed += 1
                    print(f"{table}.{col} rowid {r['rowid']}: {r[col]} -> {new}")
                    if not dry:
                        c.execute(f"UPDATE {table} SET {col}=? WHERE rowid=?", (new, r["rowid"]))
    cols = {r[1] for r in c.execute("PRAGMA table_info(transactions)")}
    for col in ("name", "signals", "proposed_at"):
        if col not in cols and not dry:
            c.execute(f"ALTER TABLE transactions ADD COLUMN {col} TEXT")
    names = data.names_map()
    gaps = []
    if not dry:
        for r in c.execute("SELECT t.id, t.symbol, t.date, t.hour, r.finished_at FROM transactions t "
                           "LEFT JOIN runs r ON r.id = t.run_id").fetchall():
            c.execute("UPDATE transactions SET name=COALESCE(name, ?), proposed_at=COALESCE(proposed_at, ?) WHERE id=?",
                      (names.get(r["symbol"]), r["finished_at"], r["id"]))
            if r["finished_at"]:
                ex = dt.datetime.fromisoformat(f"{r['date']}T{r['hour']}").replace(tzinfo=MADRID)
                lag = (ex - dt.datetime.fromisoformat(r["finished_at"])).total_seconds() / 3600
                if lag > 2:
                    gaps.append((r["symbol"], r["finished_at"][11:16], r["hour"][:5], round(lag, 1)))
        c.commit()
    print(f"{changed} timestamp(s) normalised to Madrid{' (dry run)' if dry else ''}.")
    if gaps:
        print(f"{len(gaps)} trade(s) executed > 2 h after their proposal (stale prices), e.g. "
              + ", ".join(f"{s} proposed {p} executed {e} (+{h} h)" for s, p, e, h in gaps[:3]))


if __name__ == "__main__":
    main(sys.argv[1], "--dry-run" in sys.argv)
