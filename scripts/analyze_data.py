from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
import pingouin as pg

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats


# -----------------------------
# Model: accuracy = a + b*log(block)
# -----------------------------
def log_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.log(x)


@dataclass
class FitResult:
    a: float
    b: float
    a_se: float
    b_se: float
    r2: float
    n_points: int


def fit_learning_rate(blocks: np.ndarray, acc: np.ndarray) -> FitResult:
    """
    Fit accuracy = a + b*log(block) using scipy.optimize.curve_fit.

    blocks: array of block indices (>=1)
    acc: array of accuracies in [0,1]
    """
    blocks = np.asarray(blocks, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if len(blocks) != 8:
        raise ValueError(f"Unexpected number of blocks {blocks}. Expected: 8")

    # Initial guess: a ~= acc at block 1, b small
    p0 = (float(acc[0]), 0.01)

    popt, pcov = curve_fit(log_linear, blocks, acc, p0=p0)
    a_hat, b_hat = popt

    # Standard errors from covariance
    se = np.sqrt(np.diag(pcov))
    a_se, b_se = float(se[0]), float(se[1])

    # R^2
    y_hat = log_linear(blocks, a_hat, b_hat)
    ss_res = float(np.sum((acc - y_hat) ** 2))
    ss_tot = float(np.sum((acc - np.mean(acc)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return FitResult(a=float(a_hat), b=float(b_hat), a_se=a_se, b_se=b_se, r2=r2, n_points=len(blocks))


def load_trials(
    db_path: str,
    include_only: list[str] | None = None,
    exclude: list[str] | None = None
) -> pd.DataFrame:
    """
    Load trials from the SQLite DB, optionally filtering by participant IDs.

    Args:
        db_path: Path to the SQLite database
        include_only: If provided, only load these participant IDs
        exclude: If provided, exclude these participant IDs

    Returns columns:
      participant_id, session_id, start_ts, block, cond, correct
    """
    con = sqlite3.connect(db_path)
    try:
        # Build WHERE clause based on filters
        where_clauses = []
        params = []

        if include_only is not None and len(include_only) > 0:
            placeholders = ",".join("?" * len(include_only))
            where_clauses.append(f"s.participant_id IN ({placeholders})")
            params.extend(include_only)

        if exclude is not None and len(exclude) > 0:
            placeholders = ",".join("?" * len(exclude))
            where_clauses.append(f"s.participant_id NOT IN ({placeholders})")
            params.extend(exclude)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        q = f"""
        SELECT
            s.participant_id,
            s.id AS session_id,
            s.start_ts,
            t.trial_index,
            t.block,
            t.cond,
            t.correct
        FROM trial t
        JOIN session s ON s.id = t.session_id
        {where_sql}
        ORDER BY s.participant_id, s.start_ts, t.block, t.trial_index
        """
        df = pd.read_sql_query(q, con, params=params)
    finally:
        con.close()

    # Normalize types (safe even if df is empty)
    if not df.empty:
        df["block"] = df["block"].astype(int)
        df["correct"] = df["correct"].astype(int)
        df["start_ts"] = df["start_ts"].astype(float)

    return df



def add_day_index(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Add day_index per participant based on session start_ts order:
      earliest session => day_index=1, next => 2, etc.
    """
    sessions = (
        df_trials[["participant_id", "session_id", "start_ts"]]
        .drop_duplicates()
        .sort_values(["participant_id", "start_ts"])
        .copy()
    )
    sessions["day_index"] = sessions.groupby("participant_id").cumcount() + 1

    return df_trials.merge(
        sessions, on=["participant_id", "session_id", "start_ts"], how="left"
    )


def fit_slopes_per_session(block_acc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (pid, sid, day), sub in block_acc.groupby(["participant_id", "session_id", "day_index"], sort=True):
        sub = sub.sort_values("block")

        # condition is constant within the session/day
        cond = sub["cond"].iloc[0]

        fr = fit_learning_rate(
            sub["block"].to_numpy(float),
            sub["accuracy"].to_numpy(float),
        )

        rows.append({
            "participant_id": pid,
            "session_id": sid,
            "day_index": int(day),
            "cond": cond,
            "a": fr.a,
            "b": fr.b,
            "a_se": fr.a_se,
            "b_se": fr.b_se,
            "r2": fr.r2,
            "n_points": fr.n_points,
        })

    out = pd.DataFrame(rows).sort_values(["participant_id", "day_index"])

    if out.empty:
        raise RuntimeError("No slope fits produced (check block_acc).")

    return out

def cohens_dz(diffs: np.ndarray) -> float:
    # paired/within-subject effect size
    diffs = np.asarray(diffs, float)
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def run_h1_within_subject(slopes: pd.DataFrame) -> None:
    """
    H1: d_i = b_T - b_P, test d > 0.
    """
    counts = slopes.groupby(["participant_id", "cond"]).size()
    bad = counts[counts != 1]
    assert bad.empty, (
        "Expected exactly 1 slope per participant per condition.\n"
        f"These (participant,cond) have !=1 rows:\n{bad}"
    )

    b_by = slopes.pivot(index="participant_id", columns="cond", values="b")

    assert "P" in b_by.columns and "T" in b_by.columns, (
        f"Need both columns P and T. Got columns: {list(b_by.columns)}"
    )

    missing = b_by[b_by[["P", "T"]].isna().any(axis=1)]
    assert missing.empty, (
        "Some participants are missing P or T slopes:\n"
        f"{missing}"
    )

    diffs = (b_by["T"] - b_by["P"]).to_numpy(float)
    n = len(diffs)
    assert n >= 2, "Need at least 2 participants for a t-test."

    # SciPy one-sample t-test (one-sided)
    res = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")

    t_stat = float(res.statistic)
    p_one  = float(res.pvalue)
    df     = float(res.df)  # SciPy>=1.10 returns df on result object

    # SciPy 95% CI for mean(d) (two-sided CI; that's standard even for one-sided tests)
    ci = res.confidence_interval(confidence_level=0.95)
    ci_lo, ci_hi = float(ci.low), float(ci.high)

    m  = float(np.mean(diffs))     # mean difference
    dz = cohens_dz(diffs)          # effect size (SciPy doesn't provide this)

    print("\n=== H1 (confirmatory): within-subject T > P on slope b ===")
    print(f"N completers: {n}")
    print(f"mean(d=b_T-b_P) = {m:.6f}")
    print(f"t({df:.0f}) = {t_stat:.4f}, one-sided p = {p_one:.6g}")
    print(f"95% CI for mean(d): [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Cohen's dz: {dz:.4f}")


def permutation_pvalue_day1(xT: np.ndarray, xP: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> float:
    xT = np.asarray(xT, float)
    xP = np.asarray(xP, float)

    def stat(a, b, axis):
        # mean difference T - P
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    res = stats.permutation_test(
        data=(xT, xP),
        statistic=stat,
        permutation_type="independent",
        alternative="greater",     # T > P
        n_resamples=n_resamples,
        random_state=seed,
    )
    
    return float(res.pvalue)


def run_h2_between_groups_day1(slopes: pd.DataFrame, n_perm: int = 10000, seed: int = 0) -> None:
    day1 = slopes[slopes["day_index"] == 1].copy()

    # Assert one slope per participant on Day 1
    counts = day1.groupby("participant_id").size()
    bad = counts[counts != 1]
    assert bad.empty, (
        "Expected exactly 1 Day-1 slope per participant.\n"
        f"These participants have !=1 Day-1 rows:\n{bad}\n\n"
        "If you have duplicates, you probably grouped wrong upstream."
    )

    # Split groups by Day-1 condition
    xT = day1.loc[day1["cond"] == "T", "b"].to_numpy(float)
    xP = day1.loc[day1["cond"] == "P", "b"].to_numpy(float)

    assert len(xT) > 0 and len(xP) > 0, f"Need both groups present. nT={len(xT)} nP={len(xP)}"
    assert len(xT) >= 2 and len(xP) >= 2, f"Welch t-test is shaky with tiny groups. nT={len(xT)} nP={len(xP)}"

    # Welch t-test (one-sided: T > P)
    res = stats.ttest_ind(xT, xP, equal_var=False, alternative="greater")

    t_stat = float(res.statistic)
    p_one  = float(res.pvalue)
    df     = float(res.df)

    # 95% CI for mean difference (mean(xT) - mean(xP))
    ci = res.confidence_interval(confidence_level=0.95)
    ci_lo, ci_hi = float(ci.low), float(ci.high)

    mT, mP = float(np.mean(xT)), float(np.mean(xP))
    diff = mT - mP

    g = pg.compute_effsize(xT, xP, eftype="hedges")
    p_perm = permutation_pvalue_day1(xT, xP, n_resamples=n_perm, seed=seed)

    print("\n=== H2 (exploratory): Day-1 between-groups (T Day1 > P Day1) ===")
    print(f"n_T = {len(xT)}, n_P = {len(xP)}")
    print(f"mean(b)_T = {mT:.6f}, mean(b)_P = {mP:.6f}, diff(T-P) = {diff:.6f}")
    print(f"Welch t({df:.2f}) = {t_stat:.4f}, one-sided p = {p_one:.6g}")
    print(f"95% CI for diff(T-P) (SciPy): [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Hedges' g: {g:.4f}")
    print(f"Permutation p (one-sided, {n_perm} perms): {p_perm:.6g}")


def parse_participant_list(value: str | None) -> list[str] | None:
    """Parse comma-separated participant IDs like 'p001,p002'."""
    if value is None or value.strip() == "":
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze learning rate data from Glass pattern experiment"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="study.db",
        help="Path to SQLite database (default: study.db)"
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
        help="Comma-separated list of participant IDs to exclude (e.g., 'p001,p002')"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for H2 permutation test (default: 10000)"
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=42,
        help="Random seed for permutation test (default: 42)"
    )

    args = parser.parse_args()

    # Parse participant filters
    include_only = parse_participant_list(args.include_only_participants)
    exclude = parse_participant_list(args.exclude_participants)

    # Validate mutually exclusive options
    if include_only is not None and exclude is not None:
        parser.error("Cannot use both --include-only-participants and --exclude-participants")

    # Load data
    print(f"Loading trials from {args.db}...")
    if include_only:
        print(f"  Including only: {', '.join(include_only)}")
    if exclude:
        print(f"  Excluding: {', '.join(exclude)}")

    df = load_trials(args.db, include_only=include_only, exclude=exclude)

    print(f"Loaded {df.shape[0]} trials from {df['participant_id'].nunique()} participants "
          f"across {df['session_id'].nunique()} sessions")

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Add day index
    df = add_day_index(df)

    # Compute block accuracies
    block_acc = (
        df.groupby(["participant_id", "session_id", "start_ts", "block", "day_index", "cond"])["correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    # Fit learning rates
    slopes = fit_slopes_per_session(block_acc)

    # Print fitted slopes
    print("\n=== Fitted slopes ===")
    print(slopes.sort_values(["participant_id", "day_index", "cond"]).to_string(index=False))

    # Run hypothesis tests
    run_h1_within_subject(slopes)
    run_h2_between_groups_day1(slopes, n_perm=args.n_permutations, seed=args.permutation_seed)


if __name__ == "__main__":
    main()