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


def count_unique_participants(conn: sqlite3.Connection, exclude_ids: list[str] = None) -> int:
    if not table_exists(conn, "session"):
        return 0
    exclude_ids = exclude_ids or []
    query = """
        SELECT COUNT(DISTINCT participant_id)
        FROM session
        WHERE participant_id IS NOT NULL AND TRIM(participant_id) <> ''
    """
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        query += f" AND participant_id NOT IN ({placeholders})"
        row = conn.execute(query, exclude_ids).fetchone()
    else:
        row = conn.execute(query).fetchone()
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


def count_unique_condition_participant_pairs(conn: sqlite3.Connection, exclude_ids: list[str] = None) -> int:
    # Requires both tables to exist
    if not (table_exists(conn, "trial") and table_exists(conn, "session")):
        return 0
    exclude_ids = exclude_ids or []
    query = """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT s.participant_id, t.cond
            FROM trial t
            JOIN session s ON s.id = t.session_id
            WHERE s.participant_id IS NOT NULL AND TRIM(s.participant_id) <> ''
              AND t.cond IS NOT NULL AND TRIM(t.cond) <> ''
    """
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        query += f" AND s.participant_id NOT IN ({placeholders})"
        query += ") AS distinct_pairs"
        row = conn.execute(query, exclude_ids).fetchone()
    else:
        query += ") AS distinct_pairs"
        row = conn.execute(query).fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def count_unique_conditions(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "trial"):
        return 0
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT cond)
        FROM trial
        WHERE cond IS NOT NULL AND TRIM(cond) <> ''
        """
    ).fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def get_first_condition_per_participant(conn: sqlite3.Connection, exclude_ids: list[str] = None) -> dict[str, str]:
    """Returns a dict mapping participant_id to their first condition (P or T)."""
    if not (table_exists(conn, "trial") and table_exists(conn, "session")):
        return {}
    exclude_ids = exclude_ids or []

    query = """
        SELECT s.participant_id, t.cond
        FROM trial t
        JOIN session s ON s.id = t.session_id
        WHERE s.participant_id IS NOT NULL AND TRIM(s.participant_id) <> ''
          AND t.cond IS NOT NULL AND TRIM(t.cond) <> ''
    """
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        query += f" AND s.participant_id NOT IN ({placeholders})"
        query += " ORDER BY t.id ASC"
        rows = conn.execute(query, exclude_ids).fetchall()
    else:
        query += " ORDER BY t.id ASC"
        rows = conn.execute(query).fetchall()

    # Get first condition for each participant
    first_cond = {}
    for pid, cond in rows:
        if pid not in first_cond:
            first_cond[pid] = cond

    return first_cond


def check_balance(current_count: int, target_n: int) -> tuple[bool, str]:
    """
    Check if the current balance can theoretically reach target_n with proper balance.
    Returns (is_balanced, message).
    """
    if current_count > target_n:
        return False, f"ERROR: current count ({current_count}) exceeds target ({target_n})"

    # For target_n, ideal balance is:
    # - if even: target_n/2 each
    # - if odd: (target_n-1)/2 and (target_n+1)/2
    half = target_n // 2
    target_min = half
    target_max = target_n - half

    # For current_count, acceptable range is:
    # - if even: current/2 each
    # - if odd: (current-1)/2 and (current+1)/2
    current_half = current_count // 2
    current_min = current_half
    current_max = current_count - current_half

    return True, f"acceptable range: {current_min}:{current_max} (can reach {target_min}:{target_max})"


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
    ap.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of participant IDs to exclude (e.g., p001,p002)",
    )
    ap.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Target number of participants for balance check",
    )
    args = ap.parse_args()

    # Parse excluded IDs
    exclude_ids = [pid.strip() for pid in args.exclude.split(",") if pid.strip()]

    conn = connect_readonly(args.db)
    try:
        participants = count_unique_participants(conn, exclude_ids)
        records = count_records(conn)
        num_conditions = count_unique_conditions(conn)
        cond_participant_pairs = count_unique_condition_participant_pairs(conn, exclude_ids)

        # Get first condition per participant for balance check
        first_conds = get_first_condition_per_participant(conn, exclude_ids)
    finally:
        conn.close()

    # Intentionally do not print any breakdowns by condition to preserve blinding
    print(f"participants: {participants}")
    if exclude_ids:
        print(f"excluded: {', '.join(exclude_ids)}")
    print(f"records: {records}")
    print(f"conditions: {num_conditions}")
    print(f"condition-participant pairs: {cond_participant_pairs}")

    # Balance check if target-n is provided
    if args.target_n is not None and first_conds:
        # Count how many started with each condition (without revealing which is which)
        cond_counts = {}
        for cond in first_conds.values():
            cond_counts[cond] = cond_counts.get(cond, 0) + 1

        # Get the two counts (order doesn't matter for balance check)
        # Sort them so we don't reveal which condition has which count
        counts = sorted(cond_counts.values())

        if len(counts) == 2:
            count_a, count_b = counts
            current_n = count_a + count_b

            # Check if current ratio can reach target with proper balance
            _, msg = check_balance(current_n, args.target_n)

            print(f"balance check: {count_a}:{count_b} ({current_n} total) - {msg}")
        elif len(counts) == 1:
            print(f"balance check: all {counts[0]} participants have same first condition (unbalanced)")
        else:
            print("balance check: no participants with conditions yet")


if __name__ == "__main__":
    main()


