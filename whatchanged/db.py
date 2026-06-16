from __future__ import annotations
import sqlite3
import threading
from whatchanged.models import Entry, DOMAINS, EVENTS

_COLS = ["date"] + DOMAINS + EVENTS + ["note"]
_WRITE_LOCK = threading.Lock()


def init_db(path: str = "data/whatchanged.db") -> sqlite3.Connection:
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cols_sql = ",\n".join(
        ["date TEXT PRIMARY KEY"]
        + [f"{d} INTEGER" for d in DOMAINS]
        + [f"{e} INTEGER NOT NULL DEFAULT 0" for e in EVENTS]
        + ["note TEXT NOT NULL DEFAULT ''"]
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS entries (\n{cols_sql}\n)")
    conn.commit()
    return conn


def upsert_entry(conn: sqlite3.Connection, entry: Entry) -> None:
    d = entry.to_dict()
    placeholders = ", ".join("?" for _ in _COLS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _COLS if c != "date")
    with _WRITE_LOCK:
        conn.execute(
            f"INSERT INTO entries ({', '.join(_COLS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(date) DO UPDATE SET {updates}",
            [d[c] for c in _COLS],
        )
        conn.commit()


def get_entry(conn: sqlite3.Connection, date: str) -> Entry | None:
    row = conn.execute("SELECT * FROM entries WHERE date=?", (date,)).fetchone()
    return Entry.from_dict(dict(row)) if row else None


def get_range(conn: sqlite3.Connection, start: str, end: str) -> list[Entry]:
    rows = conn.execute(
        "SELECT * FROM entries WHERE date>=? AND date<=? ORDER BY date ASC",
        (start, end),
    ).fetchall()
    return [Entry.from_dict(dict(r)) for r in rows]
