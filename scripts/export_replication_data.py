"""
Export replication study public data to CSV files.

Reads the SQLite study DB and writes:
  - block_accuracy.csv: one row per (public participant, day, block)
  - sessions.csv: one row per (public participant, day)

The exported participant_id values are public scrambled IDs from the
participant_public_id_mapping table. Internal participant IDs are not exported.

Learning rates are not exported -- they are recomputed by analyze_data.py

The output directory is the input consumed by `analyze_data.py --from-export`.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import pandas as pd

from analyze_data import (
    add_day_index,
    load_trials,
    parse_participant_list,
)


PUBLIC_ID_TABLE = "participant_public_id_mapping"
BLOCK_ACCURACY_FILENAME = "block_accuracy.csv"
SESSIONS_FILENAME = "sessions.csv"


def load_public_id_mapping(db_path: str) -> dict[str, str]:
    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (PUBLIC_ID_TABLE,),
        ).fetchone()
        if table_exists is None:
            raise RuntimeError(
                f"Missing {PUBLIC_ID_TABLE!r} table in {db_path}. "
                "Run scripts/create_public_participant_ids.py first."
            )

        rows = conn.execute(
            f"""
            SELECT participant_id, public_participant_id
            FROM {PUBLIC_ID_TABLE}
            """
        ).fetchall()

    return {participant_id: public_id for participant_id, public_id in rows}


def load_session_metadata(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                id AS session_id,
                participant_id,
                iaf_hz,
                flicker_freq_hz
            FROM session
            """,
            conn,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where public CSV exports should be written"
    )
    parser.add_argument(
        "--include-only-participants",
        type=str,
        default=None,
        help="Comma-separated list of participant IDs to include (e.g., 'p001,p002')"
    )
    parser.add_argument(
        "--exclude-participants",
        type=str,
        default=None,
        help="Comma-separated list of participant IDs to exclude"
    )

    args = parser.parse_args()

    include_only = parse_participant_list(args.include_only_participants)
    exclude = parse_participant_list(args.exclude_participants)

    if include_only is not None and exclude is not None:
        parser.error("Cannot use both --include-only-participants and --exclude-participants")

    print(f"Loading trials from {args.db}...")
    if include_only:
        print(f"  Including only: {', '.join(include_only)}")
    if exclude:
        print(f"  Excluding: {', '.join(exclude)}")

    df = load_trials(args.db, include_only=include_only, exclude=exclude)
    print(f"Loaded {df.shape[0]} trials from {df['participant_id'].nunique()} participants "
          f"across {df['session_id'].nunique()} sessions")

    if df.empty:
        print("No data loaded. Nothing to export.")
        return

    df = add_day_index(df)

    # Per-block accuracy (0-100%) plus coarse block counts.
    block_acc = (
        df.groupby(["participant_id", "session_id", "start_ts", "block", "day_index", "cond"])
        .agg(
            n_trials=("correct", "size"),
            n_correct=("correct", "sum"),
            n_timeouts=("timed_out", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    block_acc["accuracy"] = block_acc["accuracy"] * 100

    session_cond = (
        df.groupby(["participant_id", "session_id", "start_ts", "day_index"])["cond"]
        .nunique()
        .reset_index(name="n_conditions")
    )
    multi_cond_sessions = session_cond[session_cond["n_conditions"] != 1]
    if not multi_cond_sessions.empty:
        raise RuntimeError("Expected exactly one condition per participant session.")

    session_export = (
        df.groupby(["participant_id", "session_id", "start_ts", "day_index"])["cond"]
        .first()
        .reset_index()
        .merge(
            load_session_metadata(args.db),
            on=["participant_id", "session_id"],
            how="left",
        )
    )

    public_id_mapping = load_public_id_mapping(args.db)
    missing_mapping = sorted(
        (set(block_acc["participant_id"]) | set(session_export["participant_id"])) - set(public_id_mapping)
    )
    if missing_mapping:
        raise RuntimeError(
            "Missing public participant IDs for: "
            + ", ".join(missing_mapping)
            + f". Recreate or repair {PUBLIC_ID_TABLE!r}."
        )

    block_acc["participant_id"] = block_acc["participant_id"].map(public_id_mapping)
    session_export["participant_id"] = session_export["participant_id"].map(public_id_mapping)

    os.makedirs(args.output_dir, exist_ok=True)
    block_output = os.path.join(args.output_dir, BLOCK_ACCURACY_FILENAME)
    sessions_output = os.path.join(args.output_dir, SESSIONS_FILENAME)

    block_acc[[
        "participant_id", "cond", "day_index", "block",
        "n_trials", "n_correct", "n_timeouts", "accuracy",
    ]].sort_values(
        ["participant_id", "day_index", "block"]
    ).round({"accuracy": 3}).to_csv(block_output, index=False)

    session_export[[
        "participant_id", "day_index", "cond", "iaf_hz", "flicker_freq_hz",
    ]].sort_values(
        ["participant_id", "day_index"]
    ).round({"iaf_hz": 1, "flicker_freq_hz": 1}).to_csv(sessions_output, index=False)

    print(f"\nExported {len(block_acc)} block rows to {block_output}")
    print(f"Exported {len(session_export)} session rows to {sessions_output}")


if __name__ == "__main__":
    main()
