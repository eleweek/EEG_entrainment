"""
Export replication study per-block accuracy to CSV.

Reads the SQLite study DB and writes one row per (participant, session, block):
  participant_id, cond, day_index, block, accuracy

Learning rates are not exported -- they are recomputed by analyze_data.py

The CSV is the input format consumed by `analyze_data.py --from-export`.
"""

from __future__ import annotations

import argparse
import os

from analyze_data import (
    add_day_index,
    load_trials,
    parse_participant_list,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "output",
        help="Output CSV path"
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

    # Per-block accuracy (0-100%)
    block_acc = (
        df.groupby(["participant_id", "session_id", "start_ts", "block", "day_index", "cond"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    block_acc["accuracy"] = block_acc["accuracy"] * 100

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    block_acc[["participant_id", "cond", "day_index", "block", "accuracy"]].sort_values(
        ["participant_id", "day_index", "block"]
    ).round({"accuracy": 3}).to_csv(args.output, index=False)

    print(f"\nExported {len(block_acc)} rows to {args.output}")


if __name__ == "__main__":
    main()
