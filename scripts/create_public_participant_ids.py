"""
Create stable public participant IDs in a study SQLite database.

The script reads distinct internal participant IDs from the session table,
scrambles them, and stores a one-time mapping in participant_public_id_mapping.
It refuses to run if that table already exists.
"""

from __future__ import annotations

import argparse
import random
import re
import secrets
import sqlite3
from datetime import datetime, timezone


TABLE_NAME = "participant_public_id_mapping"


def parse_participant_list(value: str | None) -> set[str]:
    if value is None or value.strip() == "":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_participant_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT participant_id
        FROM session
        WHERE participant_id IS NOT NULL AND TRIM(participant_id) <> ''
        ORDER BY participant_id
        """
    ).fetchall()
    return [row[0] for row in rows]


def numeric_suffix(value: str) -> int | None:
    match = re.search(r"(\d+)$", value)
    if match is None:
        return None
    return int(match.group(1))


def creates_obvious_numeric_match(internal_id: str, public_id: str) -> bool:
    internal_num = numeric_suffix(internal_id)
    public_num = numeric_suffix(public_id)
    return internal_num is not None and internal_num == public_num


def make_mapping(participant_ids: list[str], prefix: str, seed: int) -> list[tuple[str, str]]:
    width = max(2, len(str(len(participant_ids))))
    public_ids = [f"{prefix}{i:0{width}d}" for i in range(1, len(participant_ids) + 1)]

    if len(participant_ids) <= 1:
        return list(zip(participant_ids, public_ids))

    rng = random.Random(seed)
    for _ in range(10_000):
        shuffled = public_ids[:]
        rng.shuffle(shuffled)
        mapping = list(zip(participant_ids, shuffled))
        if not any(creates_obvious_numeric_match(internal_id, public_id) for internal_id, public_id in mapping):
            return mapping

    raise RuntimeError(
        "Could not create a scrambled mapping without numeric self-matches. "
        "Try a different --seed."
    )


def create_mapping_table(conn: sqlite3.Connection, mapping: list[tuple[str, str]], seed: int) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            participant_id TEXT PRIMARY KEY,
            public_participant_id TEXT NOT NULL UNIQUE,
            seed INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        f"""
        INSERT INTO {TABLE_NAME}
            (participant_id, public_participant_id, seed, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [(internal_id, public_id, seed, created_at) for internal_id, public_id in mapping],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", help="Path to SQLite database")
    parser.add_argument(
        "--prefix",
        default="S",
        help="Prefix for public participant IDs (default: S)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used to scramble public IDs. If omitted, a fresh random seed is generated.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated internal participant IDs to exclude from the public mapping.",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    exclude = parse_participant_list(args.exclude)

    with sqlite3.connect(args.db) as conn:
        if table_exists(conn, TABLE_NAME):
            raise SystemExit(
                f"Refusing to create mapping: table {TABLE_NAME!r} already exists in {args.db}."
            )

        participant_ids = load_participant_ids(conn)
        participant_ids = [pid for pid in participant_ids if pid not in exclude]
        if not participant_ids:
            raise SystemExit(f"No participant IDs found in session table in {args.db}.")

        mapping = make_mapping(participant_ids, prefix=args.prefix, seed=seed)
        create_mapping_table(conn, mapping, seed=seed)

    print(f"Created {TABLE_NAME} with {len(mapping)} participant mappings in {args.db}.")
    if exclude:
        print(f"Excluded: {', '.join(sorted(exclude))}")
    print(f"Seed: {seed}")
    print("Public IDs were scrambled; internal IDs were not modified.")


if __name__ == "__main__":
    main()
