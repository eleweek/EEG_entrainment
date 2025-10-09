#!/usr/bin/env python3
import argparse
import os
import sqlite3
import sys


def connect_readonly(db_path: str) -> sqlite3.Connection:
    abs_path = os.path.abspath(db_path)
    uri = f"file:{abs_path}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as e:
        print(f"Error: cannot open database '{db_path}' (read-only): {e}", file=sys.stderr)
        sys.exit(1)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,)
    ).fetchone()
    return row is not None


def count_unique_participants(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "session"):
        return 0
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT participant_id)
        FROM session
        WHERE participant_id IS NOT NULL AND TRIM(participant_id) <> ''
        """
    ).fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def count_records(conn: sqlite3.Connection) -> int:
    # Prefer counting trials as the primary unit of records if present
    if table_exists(conn, "trial"):
        row = conn.execute("SELECT COUNT(*) FROM trial;").fetchone()
        return int(row[0] if row else 0)
    # Fallback to counting sessions if trial table does not exist yet
    if table_exists(conn, "session"):
        row = conn.execute("SELECT COUNT(*) FROM session;").fetchone()
        return int(row[0] if row else 0)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print database stats without revealing blinded conditions."
    )
    ap.add_argument(
        "--db",
        type=str,
        default="study.db",
        help="Path to SQLite database (default: study.db)",
    )
    args = ap.parse_args()

    conn = connect_readonly(args.db)
    try:
        participants = count_unique_participants(conn)
        records = count_records(conn)
    finally:
        conn.close()

    # Intentionally do not print any breakdowns by condition to preserve blinding
    print(f"participants: {participants}")
    print(f"records: {records}")


if __name__ == "__main__":
    main()


