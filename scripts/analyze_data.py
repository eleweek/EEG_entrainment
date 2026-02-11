from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
import pingouin as pg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg


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


def fit_learning_rate_ols(blocks: np.ndarray, acc: np.ndarray, min_points: int = 3) -> FitResult:
    """
    Fit accuracy = a + b*log(block) using OLS (L2 loss).

    blocks: array of block indices (>=1), can be fractional
    acc: array of accuracies in [0,1] or [0,100]
    min_points: minimum number of points required for fitting
    """
    blocks = np.asarray(blocks, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if len(blocks) < min_points:
        raise ValueError(f"Need at least {min_points} points, got {len(blocks)}")

    # Initial guess: a ~= acc at first block, b small
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


def fit_learning_rate_l1(blocks: np.ndarray, acc: np.ndarray, min_points: int = 3) -> FitResult:
    """
    Fit accuracy = a + b*log(block) using L1 loss (median regression).

    blocks: array of block indices (>=1), can be fractional
    acc: array of accuracies in [0,1] or [0,100]
    min_points: minimum number of points required for fitting
    """
    blocks = np.asarray(blocks, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if len(blocks) < min_points:
        raise ValueError(f"Need at least {min_points} points, got {len(blocks)}")

    # Design matrix: [1, log(block)]
    X = sm.add_constant(np.log(blocks))

    # Fit L1 (median regression)
    model = QuantReg(acc, X)
    result = model.fit(q=0.5)  # q=0.5 = median = L1

    a_hat, b_hat = result.params
    a_se, b_se = result.bse  # standard errors

    # Pseudo-R² (1 - sum|residuals| / sum|y - median(y)|)
    y_hat = result.fittedvalues
    resid_abs = np.sum(np.abs(acc - y_hat))
    total_abs = np.sum(np.abs(acc - np.median(acc)))
    r2 = 1.0 - resid_abs / total_abs if total_abs > 0 else np.nan

    return FitResult(a=float(a_hat), b=float(b_hat), a_se=float(a_se), b_se=float(b_se), r2=r2, n_points=len(blocks))


def fit_learning_rate(blocks: np.ndarray, acc: np.ndarray, method: str = "ols") -> FitResult:
    """
    Fit accuracy = a + b*log(block).

    method: "ols" (L2 loss) or "l1" (L1 loss / median regression)
    """
    if method == "ols":
        return fit_learning_rate_ols(blocks, acc)
    elif method == "l1":
        return fit_learning_rate_l1(blocks, acc)
    else:
        raise ValueError(f"Unknown fit method: {method}. Use 'ols' or 'l1'.")


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


def drop_first_n_trials(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Drop the first N trials from each session.

    This simulates a practice block - the original study had 50 practice trials
    before the main task began.
    """
    if n <= 0:
        return df

    def drop_first_n(session_df: pd.DataFrame) -> pd.DataFrame:
        # Sort by trial_index to ensure we drop the first N
        session_df = session_df.sort_values("trial_index")
        return session_df.iloc[n:]

    return df.groupby(["participant_id", "session_id"], group_keys=False).apply(drop_first_n)


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


def compute_sliding_window_accuracy(
    trials: pd.DataFrame,
    window_size: int = 100,
    step_size: int | None = None,
    n_blocks: int = 8,
) -> pd.DataFrame:
    """
    Compute accuracy using a sliding window over trials.

    Returns a DataFrame with fractional block positions (1.0 to n_blocks)
    and corresponding accuracies.

    Args:
        trials: DataFrame with columns [participant_id, session_id, day_index, cond, trial_index, correct]
        window_size: Number of trials in each window
        step_size: Step between windows (default: window_size // 8 for ~64 points per session)
        n_blocks: Number of blocks to map to (for x-axis scaling)

    Returns:
        DataFrame with columns [participant_id, session_id, day_index, cond, block, accuracy]
        where 'block' is a fractional value from 1.0 to n_blocks
    """
    if step_size is None:
        step_size = max(1, window_size // 8)

    rows = []

    for (pid, sid, day), session_trials in trials.groupby(
        ["participant_id", "session_id", "day_index"], sort=True
    ):
        session_trials = session_trials.sort_values("trial_index").reset_index(drop=True)
        n_trials = len(session_trials)
        cond = session_trials["cond"].iloc[0]

        # Slide window over trials
        for start in range(0, n_trials - window_size + 1, step_size):
            end = start + window_size
            window = session_trials.iloc[start:end]
            acc = window["correct"].mean() * 100  # Convert to percentage

            # Map center of window to fractional block position
            # Center of window in trial space (0 to n_trials-1)
            center = (start + end - 1) / 2
            # Map to block space (1.0 to n_blocks)
            # trial 0 -> block 1.0, trial n_trials-1 -> block n_blocks
            block_pos = 1.0 + (center / (n_trials - 1)) * (n_blocks - 1)

            rows.append({
                "participant_id": pid,
                "session_id": sid,
                "day_index": day,
                "cond": cond,
                "block": block_pos,
                "accuracy": acc,
            })

    return pd.DataFrame(rows)


def fit_slopes_per_session(block_acc: pd.DataFrame, method: str = "ols", drop_first_n_blocks: int = 0) -> pd.DataFrame:
    rows = []

    for (pid, sid, day), sub in block_acc.groupby(["participant_id", "session_id", "day_index"], sort=True):
        sub = sub.sort_values("block")

        # Drop first N blocks if requested (to exclude warm-up effects)
        if drop_first_n_blocks > 0:
            sub = sub[sub["block"] > drop_first_n_blocks].copy()
            # Renumber blocks so first remaining block is 1
            sub["block"] = sub["block"] - drop_first_n_blocks

        # condition is constant within the session/day
        cond = sub["cond"].iloc[0]

        fr = fit_learning_rate(
            sub["block"].to_numpy(float),
            sub["accuracy"].to_numpy(float),
            method=method,
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

    # One-sided 95% CI (lower bound only)
    ci_one = res.confidence_interval(confidence_level=0.95)
    ci_one_lo = float(ci_one.low)

    # Two-sided 95% CI (for effect size estimation)
    res_two = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
    ci_two = res_two.confidence_interval(confidence_level=0.95)
    ci_two_lo, ci_two_hi = float(ci_two.low), float(ci_two.high)

    m  = float(np.mean(diffs))     # mean difference
    sd = float(np.std(diffs, ddof=1))  # sample SD
    dz = cohens_dz(diffs)          # effect size (SciPy doesn't provide this)

    print("\n=== H1 (confirmatory): within-subject T > P on slope b ===")
    print(f"N completers: {n}")
    print(f"Difference (T-P): {m:.2f} ± {sd:.2f} (mean ± SD)")
    print(f"t({df:.0f}) = {t_stat:.4f}, one-sided p = {p_one:.6g}")
    print(f"95% CI (one-sided): [{ci_one_lo:.2f}, ∞)")
    print(f"95% CI (two-sided): [{ci_two_lo:.2f}, {ci_two_hi:.2f}]")
    print(f"Cohen's dz: {dz:.4f}")


def run_between_groups_test(
    slopes: pd.DataFrame,
    day_index: int,
    column: str,
    label: str,
    n_perm: int = 10000,
    seed: int = 0,
    two_sided: bool = False,
) -> None:
    """
    Generic between-groups comparison for a given day and column.

    Parameters:
        slopes: DataFrame with fitted slopes (must have day_index, cond, participant_id, and the column)
        day_index: Which day to filter (1 or 2)
        column: Which column to compare ("a" for intercept, "b" for slope)
        label: Label for print output (e.g., "H2", "H3")
        n_perm: Number of permutations for permutation test
        seed: Random seed for permutation test
        two_sided: If True, use two-sided test; if False, use one-sided (T > P)
    """
    day_data = slopes[slopes["day_index"] == day_index].copy()

    # Assert one row per participant on this day
    counts = day_data.groupby("participant_id").size()
    bad = counts[counts != 1]
    assert bad.empty, (
        f"Expected exactly 1 Day-{day_index} row per participant.\n"
        f"These participants have !=1 Day-{day_index} rows:\n{bad}"
    )

    # Split groups by condition on this day
    xT = day_data.loc[day_data["cond"] == "T", column].to_numpy(float)
    xP = day_data.loc[day_data["cond"] == "P", column].to_numpy(float)

    assert len(xT) > 0 and len(xP) > 0, f"Need both groups present. nT={len(xT)} nP={len(xP)}"
    assert len(xT) >= 2 and len(xP) >= 2, f"Welch t-test is shaky with tiny groups. nT={len(xT)} nP={len(xP)}"

    # Welch t-test
    alternative = "two-sided" if two_sided else "greater"
    res = stats.ttest_ind(xT, xP, equal_var=False, alternative=alternative)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    df = float(res.df)

    # 95% CI for mean difference (mean(xT) - mean(xP))
    ci = res.confidence_interval(confidence_level=0.95)
    ci_lo, ci_hi = float(ci.low), float(ci.high)

    # For one-sided tests, also compute two-sided CI for effect size estimation
    if not two_sided:
        res_two = stats.ttest_ind(xT, xP, equal_var=False, alternative="two-sided")
        ci_two = res_two.confidence_interval(confidence_level=0.95)
        ci_two_lo, ci_two_hi = float(ci_two.low), float(ci_two.high)

    mT, sdT = float(np.mean(xT)), float(np.std(xT, ddof=1))
    mP, sdP = float(np.mean(xP)), float(np.std(xP, ddof=1))
    diff = mT - mP

    g = pg.compute_effsize(xT, xP, eftype="hedges")

    # Permutation test
    def stat_fn(a, b, axis):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    perm_res = stats.permutation_test(
        data=(xT, xP),
        statistic=stat_fn,
        permutation_type="independent",
        alternative=alternative,
        n_resamples=n_perm,
        random_state=seed,
    )
    p_perm = float(perm_res.pvalue)

    col_label = "intercept a" if column == "a" else "slope b"
    test_type = "T ≠ P" if two_sided else "T > P"
    sided_label = "two-sided" if two_sided else "one-sided"
    print(f"\n=== {label}: Day-{day_index} between-groups on {col_label} ({test_type}) ===")
    print(f"n_T = {len(xT)}, n_P = {len(xP)}")
    print(f"T-match: {mT:.2f} ± {sdT:.2f} (mean ± SD)")
    print(f"P-match: {mP:.2f} ± {sdP:.2f} (mean ± SD)")
    print(f"Difference (T-P): {diff:.2f}")
    print(f"Welch t({df:.2f}) = {t_stat:.4f}, {sided_label} p = {p_val:.6g}")
    if two_sided:
        print(f"95% CI for diff(T-P): [{ci_lo:.2f}, {ci_hi:.2f}]")
    else:
        print(f"95% CI (one-sided): [{ci_lo:.2f}, ∞)")
        print(f"95% CI (two-sided): [{ci_two_lo:.2f}, {ci_two_hi:.2f}]")
    print(f"Hedges' g: {g:.4f}")
    print(f"Permutation p ({sided_label}, {n_perm} perms): {p_perm:.6g}")


def run_h2_between_groups_day1(slopes: pd.DataFrame, n_perm: int = 10000, seed: int = 0, two_sided: bool = False) -> None:
    """H2 (exploratory): Day-1 between-groups comparison on slope b."""
    run_between_groups_test(slopes, day_index=1, column="b", label="H2 (exploratory)", n_perm=n_perm, seed=seed, two_sided=two_sided)


def run_h3_between_groups_day1_intercept(slopes: pd.DataFrame, n_perm: int = 10000, seed: int = 0, two_sided: bool = False) -> None:
    """H3 (exploratory): Day-1 between-groups comparison on intercept a."""
    run_between_groups_test(slopes, day_index=1, column="a", label="H3 (exploratory)", n_perm=n_perm, seed=seed, two_sided=two_sided)


def run_h3_between_groups_day2_intercept(slopes: pd.DataFrame, n_perm: int = 10000, seed: int = 0, two_sided: bool = False) -> None:
    """H3 (exploratory): Day-2 between-groups comparison on intercept a."""
    run_between_groups_test(slopes, day_index=2, column="a", label="H3 (exploratory)", n_perm=n_perm, seed=seed, two_sided=two_sided)


def run_ancova_lr_controlling_offset(slopes: pd.DataFrame) -> None:
    """
    Mixed-effects ANCOVA: Does condition affect LR after controlling for offset and day?

    Model: b ~ cond + a + day + (1|participant_id)

    This answers: "If both conditions started at the same offset and day,
    which learning rate would be higher?"
    """
    print("\n=== Mixed-effects ANCOVA: LR ~ condition + offset + day + (1|participant) ===")

    # Prepare data - need numeric coding for condition
    data = slopes.copy()
    data["cond_T"] = (data["cond"] == "T").astype(int)  # T=1, P=0
    data["day2"] = (data["day_index"] == 2).astype(int)  # Day2=1, Day1=0

    # Center offset for easier interpretation
    data["a_centered"] = data["a"] - data["a"].mean()

    # Fit mixed-effects model: b ~ cond + a_centered + day + (1|participant_id)
    model = smf.mixedlm(
        "b ~ cond_T + a_centered + day2",
        data=data,
        groups=data["participant_id"]
    )
    result = model.fit()

    print(result.summary())

    # Extract key results for interpretation
    print("\n--- Interpretation ---")
    coef_cond = result.fe_params["cond_T"]
    pval_cond = result.pvalues["cond_T"]
    coef_offset = result.fe_params["a_centered"]
    pval_offset = result.pvalues["a_centered"]
    coef_day = result.fe_params["day2"]
    pval_day = result.pvalues["day2"]

    print(f"Condition effect (T vs P): {coef_cond:.4f} (p = {pval_cond:.4g})")
    print(f"  -> At the same offset/day, T-match LR is {coef_cond:.4f} {'higher' if coef_cond > 0 else 'lower'} than P-match")
    print(f"Offset effect: {coef_offset:.4f} (p = {pval_offset:.4g})")
    print(f"  -> For each 1% increase in offset, LR changes by {coef_offset:.4f}")
    print(f"Day effect (Day 2 vs Day 1): {coef_day:.4f} (p = {pval_day:.4g})")
    print(f"  -> Day 2 LR is {coef_day:.4f} {'higher' if coef_day > 0 else 'lower'} than Day 1")

    if pval_cond < 0.05:
        print("\nConclusion: Condition effect on LR is significant even after controlling for offset and day.")
    else:
        print("\nConclusion: Condition effect on LR is NOT significant after controlling for offset and day.")
        print("  -> The apparent LR difference may be driven by offset/day differences.")


def visualize_offset_vs_lr(slopes: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of offset (a) vs learning rate (b).

    - X = offset, Y = LR
    - Each participant's two points (Day 1, Day 2) are connected by an arrow
    - P-match = blue, T-match = red (by condition, not group)
    - Day 1 = dark shade, Day 2 = light shade
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colors by condition: P-match = blue, T-match = red
    # Day 1 = dark, Day 2 = light
    cond_colors = {
        ("P", 1): "#1f77b4",  # P-match, Day 1 (dark blue)
        ("P", 2): "#7fbfff",  # P-match, Day 2 (light blue)
        ("T", 1): "#d62728",  # T-match, Day 1 (dark red)
        ("T", 2): "#ff9896",  # T-match, Day 2 (light red)
    }

    # Plot each participant
    for pid in slopes["participant_id"].unique():
        p_data = slopes[slopes["participant_id"] == pid].sort_values("day_index")

        if len(p_data) == 2:
            # Draw arrow from Day 1 to Day 2
            d1 = p_data[p_data["day_index"] == 1].iloc[0]
            d2 = p_data[p_data["day_index"] == 2].iloc[0]
            ax.annotate(
                "",
                xy=(d2["a"], d2["b"]),  # arrow head (Day 2)
                xytext=(d1["a"], d1["b"]),  # arrow tail (Day 1)
                arrowprops=dict(
                    arrowstyle="->",
                    color="#cccccc",
                    lw=0.8,
                    shrinkA=4,  # shrink from tail (don't overlap dot)
                    shrinkB=4,  # shrink from head (don't overlap dot)
                ),
                zorder=1,
            )

        # Plot each day's point (color by condition, not group)
        for _, row in p_data.iterrows():
            day = int(row["day_index"])
            cond = row["cond"]  # T or P (the condition on THIS day)
            color = cond_colors.get((cond, day), "#888888")
            ax.scatter(
                row["a"],
                row["b"],
                c=color,
                s=50,
                zorder=2,
                edgecolors="white",
                linewidths=0.5,
            )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="P-match Day 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7fbfff", markersize=10, label="P-match Day 2"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=10, label="T-match Day 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff9896", markersize=10, label="T-match Day 2"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False)

    ax.set_xlabel("Offset (a)", fontsize=11)
    ax.set_ylabel("Learning Rate (b)", fontsize=11)
    ax.set_title("Offset vs Learning Rate by Participant", fontsize=12)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def visualize_group_average_both_days(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    drop_first_n_blocks: int = 0,
    use_sliding: bool = False,
    min_lr_1st_day: float | None = None,
) -> plt.Figure:
    """
    Plot average block accuracies for T-first and P-first groups on both days.

    - Day 1: blocks 1-8, Day 2: blocks 9-16 (continuous x-axis)
    - T-first in blue shades, P-first in green shades
    - Day 1 = dark, Day 2 = light
    - Fit log-linear curves separately for each day
    - Tufte-style: minimal, direct labeling

    Args:
        min_lr_1st_day: If set, exclude participants with day 1 LR below this threshold
    """
    # Filter participants by minimum LR on day 1 if specified
    title_suffix = ""
    if min_lr_1st_day is not None:
        # Exclude participants who have LR < min_lr_1st_day on day 1
        day1_slopes = slopes[slopes["day_index"] == 1]
        bad_pids = day1_slopes[day1_slopes["b"] < min_lr_1st_day]["participant_id"].unique()
        n_excluded = len(bad_pids)
        valid_pids = slopes[~slopes["participant_id"].isin(bad_pids)]["participant_id"].unique()
        block_acc = block_acc[block_acc["participant_id"].isin(valid_pids)].copy()
        slopes = slopes[slopes["participant_id"].isin(valid_pids)].copy()
        title_suffix = f" (excluded {n_excluded} participants with day 1 LR < {min_lr_1st_day})"

    # Drop first N blocks if requested and renumber (only for discrete blocks)
    if drop_first_n_blocks > 0 and not use_sliding:
        block_acc = block_acc[block_acc["block"] > drop_first_n_blocks].copy()
        block_acc["block"] = block_acc["block"] - drop_first_n_blocks
    # Determine each participant's first-day condition (group)
    day1_cond = (
        slopes[slopes["day_index"] == 1][["participant_id", "cond"]]
        .drop_duplicates()
        .set_index("participant_id")["cond"]
    )

    # Add group to block_acc
    acc_with_group = block_acc.copy()
    acc_with_group["group"] = acc_with_group["participant_id"].map(day1_cond)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colors: T-first = blue, P-first = green; Day 1 = dark, Day 2 = light
    colors = {
        ("P", 1): "#2d8a2d",  # P-first, Day 1 (dark green)
        ("P", 2): "#7dca7d",  # P-first, Day 2 (light green)
        ("T", 1): "#1f5f8a",  # T-first, Day 1 (dark blue)
        ("T", 2): "#6fb3d9",  # T-first, Day 2 (light blue)
    }

    # Store endpoints for direct labeling
    endpoints = {}

    for group in ["P", "T"]:
        for day in [1, 2]:
            # Filter data
            day_acc = acc_with_group[(acc_with_group["day_index"] == day) & (acc_with_group["group"] == group)]

            # Compute mean accuracy per block
            grp_means = day_acc.groupby("block")["accuracy"].mean().reset_index()
            blocks_original = grp_means["block"].values
            acc = grp_means["accuracy"].values

            # Shift Day 2 blocks for plotting (after Day 1's range)
            max_block = 8 - drop_first_n_blocks if not use_sliding else 8
            blocks_plot = blocks_original + (max_block if day == 2 else 0)

            color = colors[(group, day)]

            # Plot dots - small, unobtrusive
            ax.scatter(blocks_plot, acc, c=color, s=20, zorder=3)

            # Fit log-linear curve
            min_b = blocks_original.min()
            max_b = blocks_original.max()
            fit = fit_learning_rate(blocks_original, acc, method="ols")
            x_fit_original = np.linspace(min_b, max_b, 100)
            x_fit_plot = x_fit_original + (max_block if day == 2 else 0)
            y_fit = log_linear(x_fit_original, fit.a, fit.b)
            ax.plot(x_fit_plot, y_fit, color=color, linewidth=1.2, zorder=2)

            # Store endpoint for direct labeling
            endpoints[(group, day)] = (x_fit_plot[-1], y_fit[-1], fit.b)

    # Direct labeling (Tufte-style: no legend)
    # T-first labels above curves, P-first labels below
    label_names = {
        ("T", 1): "T-match",
        ("T", 2): "T→P-match",
        ("P", 1): "P-match",
        ("P", 2): "P→T-match",
    }
    for (group, day), (x, y, lr) in endpoints.items():
        color = colors[(group, day)]
        # T-first: label above (positive offset), P-first: label below (negative offset)
        y_offset = 2 if group == "T" else -2
        va = "bottom" if group == "T" else "top"
        ax.text(x - 3.5, y + y_offset, f"{label_names[(group, day)]}  LR={lr:.1f}",
                color=color, fontsize=8, va=va, ha="center")

    # Minimal day separator (at boundary between Day 1 and Day 2)
    n_blocks = 8 if use_sliding else 8 - drop_first_n_blocks
    ax.axvline(x=n_blocks + 0.5, color="#dddddd", linewidth=0.5, zorder=0)

    # Tufte-style axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Set up x-axis ticks
    day1_ticks = list(range(1, n_blocks + 1))
    day2_ticks = list(range(n_blocks + 1, 2 * n_blocks + 1))
    ax.set_xlim(0.5, 2 * n_blocks + 0.5)
    ax.set_xticks(day1_ticks + day2_ticks)
    ax.set_xticklabels(day1_ticks + day1_ticks, fontsize=8, color="black")
    ax.tick_params(axis="x", length=3, width=0.5)
    ax.set_xlabel("Block", fontsize=9, color="black")

    ax.set_ylabel("Accuracy (%)", fontsize=9, color="black")
    ax.tick_params(axis="y", colors="black", length=3, width=0.5)

    if title_suffix:
        ax.set_title(f"Group Average{title_suffix}", fontsize=10)

    fig.tight_layout()
    return fig


def visualize_learning_curves(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    drop_first_n_blocks: int = 0,
    use_sliding: bool = False,
    day1_only: bool = False,
) -> plt.Figure:
    """
    Create visualization of accuracy and learning rate fits for all participants.

    Layout: 4 columns per row (default) or 2 columns per row (day1_only=True)
      - Default: Columns 0-1: P-first (Day 1, Day 2), Columns 2-3: T-first (Day 1, Day 2)
      - day1_only: Column 0: P-first (Day 1), Column 1: T-first (Day 1)
    Two participants per row (one from each group).
    """
    # Drop first N blocks if requested and renumber (only for discrete blocks)
    if drop_first_n_blocks > 0 and not use_sliding:
        block_acc = block_acc[block_acc["block"] > drop_first_n_blocks].copy()
        block_acc["block"] = block_acc["block"] - drop_first_n_blocks

    n_blocks = 8 if use_sliding else 8 - drop_first_n_blocks

    # Determine each participant's first-day condition
    day1_cond = (
        slopes[slopes["day_index"] == 1][["participant_id", "cond"]]
        .drop_duplicates()
        .set_index("participant_id")["cond"]
    )

    # Split participants by their Day 1 condition
    p_first = sorted([pid for pid, c in day1_cond.items() if c == "P"])
    t_first = sorted([pid for pid, c in day1_cond.items() if c == "T"])

    n_rows = max(len(p_first), len(t_first))

    # Layout: 2 columns (day1_only) or 4 columns (default)
    if day1_only:
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 1.25 * n_rows), squeeze=False)
    else:
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 1.25 * n_rows), squeeze=False)

    # Compute global y-axis range from data with small padding
    all_acc = block_acc["accuracy"].values
    y_min = all_acc.min()
    y_max = all_acc.max()
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Muted colors
    data_color = "#000000"
    fit_color = "#888888"

    def plot_participant(ax, pid, day_idx, is_last_row):
        if pid is None:
            ax.set_visible(False)
            return

        p_data = block_acc[block_acc["participant_id"] == pid]
        p_slopes = slopes[slopes["participant_id"] == pid]

        day_data = p_data[p_data["day_index"] == day_idx]
        day_slope = p_slopes[p_slopes["day_index"] == day_idx]

        if day_data.empty:
            ax.set_visible(False)
            return

        blocks = day_data["block"].values
        acc = day_data["accuracy"].values

        # Plot accuracy points - small, direct
        ax.scatter(blocks, acc, s=4, color=data_color, zorder=3)

        # Plot fitted curve if we have slope data
        fit_annotation = ""
        if not day_slope.empty:
            a = day_slope["a"].iloc[0]
            b = day_slope["b"].iloc[0]
            r2 = day_slope["r2"].iloc[0]

            # Fit curve spans 1 to n_blocks
            x_fit = np.linspace(1, n_blocks, 100)
            y_fit = log_linear(x_fit, a, b)
            ax.plot(x_fit, y_fit, color=fit_color, linewidth=1, zorder=2)
            fit_annotation = f"  LR={b:.2f}  off={a:.2f}  R²={r2:.2f}"

        # Tufte-style: remove spines, keep only left and bottom
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        # Minimal ticks
        ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#000000")

        # Title with participant and fit info (day/condition shown in column headers)
        ax.set_title(f"{pid}{fit_annotation}",
                     fontsize=9, loc="left", color="#000000")

        ax.set_xlim(0.5, n_blocks + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(1, n_blocks + 1))

        if is_last_row:
            ax.set_xlabel("Block", fontsize=8, color="#000000")
        else:
            ax.set_xticklabels([])

    if day1_only:
        # Day 1 only: 2 columns (P-first, T-first)
        for row_idx in range(n_rows):
            pid_p = p_first[row_idx] if row_idx < len(p_first) else None
            pid_t = t_first[row_idx] if row_idx < len(t_first) else None
            is_last_row = row_idx == n_rows - 1

            # Column 0: P-first Day 1
            plot_participant(axes[row_idx, 0], pid_p, 1, is_last_row)
            # Column 1: T-first Day 1
            plot_participant(axes[row_idx, 1], pid_t, 1, is_last_row)

            # Y-axis labels only on leftmost column
            axes[row_idx, 0].set_ylabel("Accuracy", fontsize=8, color="#000000")
            axes[row_idx, 1].set_yticklabels([])

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.subplots_adjust(wspace=0.15, hspace=0.4)

        # Add group headers at the top
        fig.text(0.25, 0.98, "P-match (Day 1)", ha="center", va="top", fontsize=12, color="#000000")
        fig.text(0.75, 0.98, "T-match (Day 1)", ha="center", va="top", fontsize=12, color="#000000")

    else:
        # Default: 4 columns (Day 1 + Day 2 for both groups)
        for row_idx in range(n_rows):
            # P-first participant (columns 0, 1)
            pid_p = p_first[row_idx] if row_idx < len(p_first) else None
            # T-first participant (columns 2, 3)
            pid_t = t_first[row_idx] if row_idx < len(t_first) else None

            is_last_row = row_idx == n_rows - 1

            # P-first: Day 1 (col 0), Day 2 (col 1)
            plot_participant(axes[row_idx, 0], pid_p, 1, is_last_row)
            plot_participant(axes[row_idx, 1], pid_p, 2, is_last_row)

            # T-first: Day 1 (col 2), Day 2 (col 3)
            plot_participant(axes[row_idx, 2], pid_t, 1, is_last_row)
            plot_participant(axes[row_idx, 3], pid_t, 2, is_last_row)

            # Y-axis labels only on leftmost of each group
            for col_idx in range(4):
                ax = axes[row_idx, col_idx]
                if col_idx in (0, 2):
                    ax.set_ylabel("Accuracy", fontsize=8, color="#000000")
                else:
                    ax.set_yticklabels([])

        # Adjust layout with gap between the two groups
        fig.tight_layout(rect=(0, 0, 0.98, 0.95))
        fig.subplots_adjust(wspace=0.15, hspace=0.4)
        # Add extra space between columns 1 and 2 (between P-first and T-first)
        for row_idx in range(n_rows):
            for col_idx in [2, 3]:
                ax = axes[row_idx, col_idx]
                pos = ax.get_position()
                ax.set_position([pos.x0 + 0.03, pos.y0, pos.width, pos.height])

        # Add group headers at the top
        fig.text(0.25, 0.99, "P-match first", ha="center", va="top", fontsize=13, color="#000000")
        fig.text(0.75, 0.99, "T-match first", ha="center", va="top", fontsize=13, color="#000000")

        # Add column headers (Day + condition) above each column
        col_headers = ["Day 1, P-match", "Day 2, T-match", "Day 1, T-match", "Day 2, P-match"]
        for col_idx, header in enumerate(col_headers):
            ax = axes[0, col_idx]
            pos = ax.get_position()
            fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.04, header,
                     ha="center", va="bottom", fontsize=11, color="#000000")

    return fig


def parse_participant_list(value: str | None) -> list[str] | None:
    """Parse comma-separated participant IDs like 'p001,p002'."""
    if value is None or value.strip() == "":
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


# -----------------------------
# Original paper data adapters
# -----------------------------

def load_original_paper_data(
    data_dir: str,
    groups: dict[int, str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and convert original paper data (Michael et al., 2023) to our format.

    The paper data has:
    - AccPerLat.csv: accuracy per block, 3 rows per subject (3 latency conditions)
    - groupID: 1=Peak-Match (P), 2=Trough-Match (T), 3=Trough-NonMatch, 4=Control

    Args:
        data_dir: Path to directory containing AccPerLat.csv
        groups: Dict mapping groupID to condition label. Default: {1: "P", 2: "T"} for P-match vs T-match

    Returns:
        block_acc: DataFrame with [participant_id, cond, block, accuracy]
        slopes: DataFrame with fitted slopes per participant
    """
    import os

    if groups is None:
        groups = {1: "P", 2: "T"}  # Default: P-match vs T-match

    acc_path = os.path.join(data_dir, "AccPerLat.csv")
    df_acc = pd.read_csv(acc_path)

    blocks = ['block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'block8']

    # Assign subject IDs (they cycle 1-20 within each group, 3 latencies each)
    df_acc = df_acc.copy()
    df_acc['assumed_subID'] = -1

    for gid in sorted(df_acc['groupID'].unique()):
        group_mask = df_acc['groupID'] == gid
        group_rows = df_acc[group_mask].index
        for idx, row_idx in enumerate(group_rows):
            slot_in_latency = idx % 20
            subject_num = slot_in_latency + 1
            df_acc.loc[row_idx, 'assumed_subID'] = subject_num

    # Convert to long format, averaging across latencies
    rows = []
    for gid, cond in groups.items():
        group_data = df_acc[df_acc['groupID'] == gid]

        for subj in range(1, 21):
            subj_data = group_data[group_data['assumed_subID'] == subj]
            if len(subj_data) != 3:
                continue

            # Average accuracy across 3 latencies for each block
            avg_accs = subj_data[blocks].mean(axis=0).values * 100  # Convert to %

            pid = f"orig_{cond}_{subj:02d}"

            for block_idx, acc in enumerate(avg_accs, start=1):
                rows.append({
                    "participant_id": pid,
                    "session_id": f"{pid}_s1",
                    "day_index": 1,
                    "cond": cond,
                    "block": block_idx,
                    "accuracy": acc,
                })

    block_acc = pd.DataFrame(rows)

    # Fit slopes
    slope_rows = []
    computed_lr_map = {}  # participant_id -> computed learning rate

    for pid in block_acc["participant_id"].unique():
        p_data = block_acc[block_acc["participant_id"] == pid].sort_values("block")
        cond = p_data["cond"].iloc[0]

        blocks_arr = p_data["block"].to_numpy(float)
        accuracy_arr = p_data["accuracy"].to_numpy(float)

        fr = fit_learning_rate(blocks_arr, accuracy_arr, method="ols")
        computed_lr_map[pid] = fr.b

        slope_rows.append({
            "participant_id": pid,
            "session_id": f"{pid}_s1",
            "day_index": 1,
            "cond": cond,
            "a": fr.a,
            "b": fr.b,
            "a_se": fr.a_se,
            "b_se": fr.b_se,
            "r2": fr.r2,
            "n_points": fr.n_points,
        })

    slopes = pd.DataFrame(slope_rows)

    # Comprehensive validation: compare with provided learning rates if available
    print("\n" + "=" * 60)
    print("VALIDATING LEARNING RATE COMPUTATION")
    print("=" * 60)

    # Check for groupLR_forLMM.csv (provided learning rates)
    lr_provided_path = os.path.join(data_dir, "groupLR_forLMM.csv")

    if os.path.exists(lr_provided_path):
        df_lr_provided = pd.read_csv(lr_provided_path)
        print(f"\nFound provided learning rates: {lr_provided_path}")
        print(f"Columns: {list(df_lr_provided.columns)}")

        # Map groupID to phase/match for comparison
        # groupID: 1=Peak-Match (phase=1, match=1), 2=Trough-Match (phase=2, match=1),
        #          3=Trough-NonMatch (phase=2, match=2), 4=Control
        group_mapping = {
            1: (1, 1, "P", "Peak-Match"),
            2: (2, 1, "T", "Trough-Match"),
            3: (2, 2, "TN", "Trough-NonMatch"),
            4: (None, None, "C", "Control"),
        }

        # For validation, we need to load ALL groups, not just the requested ones
        # Build a complete dataset for validation
        all_validation_slopes = []
        for gid in [1, 2, 3, 4]:
            if gid in groups:
                # Already computed, use existing data
                cond = groups[gid]
                group_slopes = slopes[slopes["cond"] == cond].copy()
                group_slopes["groupID"] = gid
                all_validation_slopes.append(group_slopes)
            else:
                # Need to compute this group for validation
                phase, match, cond_code, label = group_mapping.get(gid, (None, None, None, None))
                if phase is None:
                    continue  # Skip control if no phase/match

                group_data = df_acc[df_acc['groupID'] == gid]
                if len(group_data) == 0:
                    continue

                # Compute slopes for this group
                temp_slopes = []
                for subj in range(1, 21):
                    subj_data = group_data[group_data['assumed_subID'] == subj]
                    if len(subj_data) != 3:
                        continue

                    avg_accs = subj_data[blocks].mean(axis=0).values * 100
                    pid = f"validation_{cond_code}_{subj:02d}"

                    # Create temporary block_acc data
                    temp_blocks = np.arange(1, 9)
                    fr = fit_learning_rate(temp_blocks, avg_accs, method="ols")

                    temp_slopes.append({
                        "participant_id": pid,
                        "groupID": gid,
                        "cond": cond_code,
                        "b": fr.b,
                    })

                if temp_slopes:
                    temp_df = pd.DataFrame(temp_slopes)
                    all_validation_slopes.append(temp_df)

        if all_validation_slopes:
            validation_slopes_df = pd.concat(all_validation_slopes, ignore_index=True)

            print("\n" + "-" * 60)
            print("Comparing computed vs. provided learning rates:")
            print("-" * 60)

            mismatches = []
            tolerance = 2.0  # Allow up to 2.0 difference in mean LR

            for gid in [1, 2, 3, 4]:
                if gid not in group_mapping:
                    continue

                phase, match, cond_code, label = group_mapping[gid]

                # Get our computed learning rates for this group
                computed_lr = validation_slopes_df[validation_slopes_df["groupID"] == gid]["b"].values

                if len(computed_lr) == 0:
                    continue  # Skip if we don't have data for this group

                # Get provided learning rates for this group (if available)
                if phase is not None and match is not None:
                    provided_lr = df_lr_provided[
                        (df_lr_provided['phase'] == phase) &
                        (df_lr_provided['match'] == match)
                    ]['LR'].values
                else:
                    # Control group might not be in the provided file
                    provided_lr = np.array([])

                if len(provided_lr) == 0:
                    print(f"\n{label} (Group {gid}):")
                    print("  No provided LR found in groupLR_forLMM.csv")
                    continue

                # Compute statistics
                provided_mean = provided_lr.mean()
                computed_mean = computed_lr.mean()
                diff = abs(provided_mean - computed_mean)

                # Sort both arrays for correlation
                provided_sorted = np.sort(provided_lr)
                computed_sorted = np.sort(computed_lr)

                # Compute correlation (if same length)
                if len(provided_lr) == len(computed_lr):
                    corr = np.corrcoef(provided_sorted, computed_sorted)[0, 1]
                else:
                    corr = np.nan

                print(f"\n{label} (Group {gid}):")
                print(f"  Provided LR:  {provided_mean:7.3f} ± {provided_lr.std():6.3f} (n={len(provided_lr)})")
                print(f"  Computed LR:  {computed_mean:7.3f} ± {computed_lr.std():6.3f} (n={len(computed_lr)})")
                print(f"  |Difference|: {diff:7.3f}")
                if not np.isnan(corr):
                    print(f"  Correlation:  {corr:7.3f}", end="")

            if diff > tolerance:
                print(" ✗ MISMATCH!")
                mismatches.append({
                    'group': label,
                    'provided': provided_mean,
                    'computed': computed_mean,
                    'diff': diff,
                    'corr': corr
                })
            else:
                print(" ✓ OK")

        # Show example fit for one subject
        if len(computed_lr_map) > 0:
            example_pid = list(computed_lr_map.keys())[0]
            example_data = block_acc[block_acc["participant_id"] == example_pid].sort_values("block")
            example_slopes = slopes[slopes["participant_id"] == example_pid].iloc[0]

            print("\n" + "-" * 60)
            print(f"Example fit: {example_pid}")
            print("-" * 60)
            print(f"Fitted curve: y = {example_slopes['a']:.2f} + {example_slopes['b']:.2f}*log(x)")
            print(f"Learning rate (b) = {example_slopes['b']:.3f}")
            print(f"R² = {example_slopes['r2']:.3f}")
            print("\nBlock | Accuracy | Fitted")
            print("-" * 35)
            for _, row in example_data.iterrows():
                block_num = row['block']
                actual = row['accuracy']
                fitted = example_slopes['a'] + example_slopes['b'] * np.log(block_num)
                print(f"  {int(block_num)}   |  {actual:6.2f}  | {fitted:6.2f}")

        if mismatches:
            print("\n" + "=" * 60)
            print("⚠ VALIDATION WARNING")
            print("=" * 60)
            print(f"\nFound {len(mismatches)} group(s) with learning rate mismatch:")
            for mm in mismatches:
                print(f"  {mm['group']}: provided={mm['provided']:.3f}, "
                     f"computed={mm['computed']:.3f}, diff={mm['diff']:.3f}")
            print(f"\nTolerance: {tolerance}")
            print("\nPossible reasons:")
            print("  1. Different fitting method (OLS vs. curve_fit)")
            print("  2. Subject ID assignment differences")
            print("  3. Different preprocessing or filtering")
        else:
            print("\n" + "=" * 60)
            print("✓ VALIDATION PASSED")
            print("=" * 60)
            print(f"All groups match within tolerance ({tolerance})")
    else:
        print("\nNo groupLR_forLMM.csv found in data directory.")
        print("Skipping validation. Proceeding with computed learning rates only.")

    return block_acc, slopes


def load_provided_learning_rates(data_dir: str) -> pd.DataFrame | None:
    """
    Load provided learning rates from groupLR_forLMM.csv.

    Returns DataFrame with columns: [cond, b] where b is the learning rate
    Returns None if file doesn't exist.
    """
    import os

    lr_path = os.path.join(data_dir, "groupLR_forLMM.csv")
    if not os.path.exists(lr_path):
        return None

    df_lr = pd.read_csv(lr_path)

    # Map phase/match to condition codes
    # phase=1, match=1 -> P-match
    # phase=2, match=1 -> T-match
    # phase=2, match=2 -> T-nonmatch

    rows = []
    for _, row in df_lr.iterrows():
        phase = int(row['phase'])
        match = int(row['match'])
        lr = float(row['LR'])
        subid = int(row['subID'])

        if phase == 1 and match == 1:
            cond = "P"
            pid = f"orig_P_{((subid-1) % 20) + 1:02d}"
        elif phase == 2 and match == 1:
            cond = "T"
            pid = f"orig_T_{((subid-21) % 20) + 1:02d}"
        elif phase == 2 and match == 2:
            cond = "TN"
            pid = f"orig_TN_{((subid-41) % 20) + 1:02d}"
        else:
            continue

        rows.append({
            "participant_id": pid,
            "cond": cond,
            "b": lr,
        })

    return pd.DataFrame(rows)


def visualize_lr_strip_plot(
    orig_p_slopes: pd.DataFrame,
    orig_t_slopes: pd.DataFrame,
    orig_tn_slopes: pd.DataFrame,
    orig_c_slopes: pd.DataFrame,
    rep_p_slopes: pd.DataFrame,
    rep_t_slopes: pd.DataFrame,
) -> plt.Figure:
    """
    Dot plot showing individual learning rates for each group.
    Dots are binned and stacked vertically to create a histogram-like appearance.

    Args:
        orig_p_slopes: Original P-match slopes
        orig_t_slopes: Original T-match slopes
        orig_tn_slopes: Original T-nonmatch slopes
        orig_c_slopes: Original Control slopes
        rep_p_slopes: Replication P-match slopes (day 1)
        rep_t_slopes: Replication T-match slopes (day 1)
    """
    # Merge all original conditions
    all_orig_values = np.concatenate([
        orig_p_slopes["b"].values,
        orig_t_slopes["b"].values,
        orig_tn_slopes["b"].values,
        orig_c_slopes["b"].values
    ])

    fig, ax = plt.subplots(figsize=(10, 7))

    # section=0 -> Original, section=1 -> Replication
    groups_info = [
        ("Original pooled", all_orig_values, "#333333", 0),
        ("Arrhythmic Control", orig_c_slopes["b"].values, "#666666", 0),
        ("T-nonmatch", orig_tn_slopes["b"].values, "#9370DB", 0),
        ("P-match", orig_p_slopes["b"].values, "#2d8a2d", 0),
        ("T-match", orig_t_slopes["b"].values, "#1f5f8a", 0),
        ("P-match", rep_p_slopes["b"].values, "#2d8a2d", 1),
        ("T-match", rep_t_slopes["b"].values, "#1f5f8a", 1),
    ]

    dot_size = 40  # Size of each dot (matplotlib scatter 's' parameter)
    dot_spacing = 0.10  # Vertical spacing between layers
    # Calculate overlap threshold based on dot radius
    # For scatter plot, s is area in points^2, so radius ≈ sqrt(s)
    # In data coordinates, diameter is approximately sqrt(dot_size) / 50
    dot_radius_data = np.sqrt(dot_size) / 50  # Approximate conversion to data units
    overlap_threshold = 2 * dot_radius_data  # Two dots touch when centers are 2*radius apart

    # Compute y positions: tighter spacing, with extra gaps between sections
    strip_spacing = 0.7
    section_gap = 1.2  # gap between "All conditions" and the rest of Original
    section_break = 1.4  # gap between Original and Replication sections
    y_positions = []
    for i in range(len(groups_info)):
        if i == 0:
            y_positions.append(0)
        elif i == 1:
            y_positions.append(y_positions[-1] + section_gap)
        elif groups_info[i][3] != groups_info[i - 1][3]:
            # Section change (Original -> Replication)
            y_positions.append(y_positions[-1] + section_break)
        else:
            y_positions.append(y_positions[-1] + strip_spacing)

    for group_idx, (label, values, color, section) in enumerate(groups_info):
        y_baseline = y_positions[group_idx]

        # Greedy layering: sort values first, then place each dot in the lowest layer where it fits
        # Processing in sorted order (left to right) creates a tidier layout
        sorted_indices = np.argsort(values)
        layers = {}  # layer_num -> sorted list of x positions already placed in that layer
        dot_layers = {}  # index -> layer assignment

        for idx in sorted_indices:
            val = values[idx]
            # Try each layer starting from 0
            layer = 0
            while True:
                if layer not in layers:
                    layers[layer] = []

                # Check if this value fits in current layer (no overlap)
                fits = True
                for existing_x in layers[layer]:
                    if abs(val - existing_x) < overlap_threshold:
                        fits = False
                        break

                if fits:
                    # Insert in sorted order for tidier bookkeeping
                    import bisect
                    bisect.insort(layers[layer], val)
                    dot_layers[idx] = layer
                    break
                else:
                    layer += 1

        # Plot all dots at their exact positions with assigned layers
        for idx, val in enumerate(values):
            layer = dot_layers[idx]
            y_pos = y_baseline + layer * dot_spacing
            ax.scatter(val, y_pos, s=dot_size, color=color, alpha=0.7,
                      edgecolors='white', linewidth=0.5, zorder=5)

        # Add mean line starting from baseline
        mean = np.mean(values)
        median = np.median(values)
        max_layer = max(dot_layers.values()) if dot_layers else 0
        line_top = y_baseline + max_layer * dot_spacing + 0.15
        ax.vlines(mean, y_baseline - 0.05, line_top,
                 color=color, linewidth=1, zorder=10)
        ax.vlines(median, y_baseline - 0.05, line_top,
                 color=color, linewidth=1, linestyle='--', zorder=10)
        is_first = (group_idx == 0)
        is_last = (group_idx == len(groups_info) - 1)
        mean_label = f"mean={mean:.2f}" if is_first or is_last else f"{mean:.2f}"
        ax.text(mean, y_baseline - 0.15, mean_label,
                color=color, fontsize=9, ha="center", va="top", zorder=10)

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

    # Tufte-style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Set y-axis labels and limits
    ax.set_yticks(y_positions)
    ax.set_yticklabels([g[0] for g in groups_info], fontsize=9)
    ax.set_ylim(-0.5, y_positions[-1] + 0.8)

    # Add section subheadings to the left of y-axis, right-aligned with tick labels
    orig_indices = [i for i, g in enumerate(groups_info) if g[3] == 0]
    rep_indices = [i for i, g in enumerate(groups_info) if g[3] == 1]

    # Use axes transform for x (aligned with tick labels) and data transform for y
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    tick_x = -0.01  # just left of the tick labels
    ax.text(tick_x, y_positions[orig_indices[-1]] + 0.4, "Original Study",
            transform=trans, fontsize=11, fontweight="bold", ha="right", va="bottom", color="black")
    ax.text(tick_x, y_positions[rep_indices[-1]] + 0.4, "Replication, Day 1",
            transform=trans, fontsize=11, fontweight="bold", ha="right", va="bottom", color="black")

    ax.tick_params(axis="both", colors="black", length=3, width=0.5)
    ax.set_xlabel("Learning Rates by Group", fontsize=11, fontweight="bold", color="black", labelpad=10)
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig("strip_plot.png", dpi=300, bbox_inches="tight")
    return fig


def visualize_original_individual_curves(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    max_per_group: int = 6,
    cols_per_group: int = 3,
    seed: int = 42,
    cond_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """
    Visualize individual learning curves from original paper data.

    Layout: cols_per_group columns for condition 1 (left) + cols_per_group columns for condition 2 (right),
    limited to max_per_group random samples per condition.

    Args:
        block_acc: Block accuracy data
        slopes: Fitted slopes data
        max_per_group: Maximum participants to show per condition
        cols_per_group: Number of columns per condition
        seed: Random seed for sampling
        cond_labels: Dict mapping condition codes to display labels. Default: {"P": "P-match", "T": "T-match"}
    """
    np.random.seed(seed)

    if cond_labels is None:
        cond_labels = {"P": "P-match", "T": "T-match"}

    # Get the two conditions from the data
    conditions = sorted(slopes["cond"].unique())
    if len(conditions) != 2:
        raise ValueError(f"Expected 2 conditions, got {len(conditions)}: {conditions}")

    cond1, cond2 = conditions

    # Split by condition
    cond1_pids = slopes[slopes["cond"] == cond1]["participant_id"].unique()
    cond2_pids = slopes[slopes["cond"] == cond2]["participant_id"].unique()

    # Sample if needed
    if len(cond1_pids) > max_per_group:
        cond1_pids = np.random.choice(cond1_pids, max_per_group, replace=False)
    if len(cond2_pids) > max_per_group:
        cond2_pids = np.random.choice(cond2_pids, max_per_group, replace=False)

    # Layout: cols_per_group for cond1 + cols_per_group for cond2 per row
    n_cols = cols_per_group * 2
    n_rows = max(math.ceil(len(cond1_pids) / cols_per_group), math.ceil(len(cond2_pids) / cols_per_group))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.5 * n_rows), squeeze=False)

    # Global y-axis range across displayed participants only
    displayed_pids = list(cond1_pids) + list(cond2_pids)
    displayed_acc = block_acc[block_acc["participant_id"].isin(displayed_pids)]["accuracy"].values
    y_min, y_max = displayed_acc.min(), displayed_acc.max()
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    data_color = "#000000"
    fit_color = "#888888"

    def plot_one(ax, pid, is_last_row, show_ylabel):
        if pid is None:
            ax.set_visible(False)
            return

        p_data = block_acc[block_acc["participant_id"] == pid]
        p_slope = slopes[slopes["participant_id"] == pid]

        blocks = p_data["block"].values
        acc = p_data["accuracy"].values

        ax.scatter(blocks, acc, s=4, color=data_color, zorder=3)

        fit_annotation = ""
        if not p_slope.empty:
            a = p_slope["a"].iloc[0]
            b = p_slope["b"].iloc[0]
            r2 = p_slope["r2"].iloc[0]

            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, a, b)
            ax.plot(x_fit, y_fit, color=fit_color, linewidth=1, zorder=2)
            fit_annotation = f"LR={b:.2f}  off={a:.2f}  R²={r2:.2f}"

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#000000")
        ax.set_title(fit_annotation, fontsize=9, loc="left", color="#000000")

        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(1, 9))

        if is_last_row:
            ax.set_xlabel("Block", fontsize=8, color="#000000")
        else:
            ax.set_xticklabels([])

        if show_ylabel:
            ax.set_ylabel("Accuracy", fontsize=8, color="#000000")
        else:
            ax.set_yticklabels([])

    for row_idx in range(n_rows):
        is_last = row_idx == n_rows - 1

        # Condition 1: columns 0 to cols_per_group-1
        for col in range(cols_per_group):
            idx = row_idx * cols_per_group + col
            pid = cond1_pids[idx] if idx < len(cond1_pids) else None
            plot_one(axes[row_idx, col], pid, is_last, show_ylabel=(col == 0))

        # Condition 2: columns cols_per_group to n_cols-1
        for col in range(cols_per_group):
            idx = row_idx * cols_per_group + col
            pid = cond2_pids[idx] if idx < len(cond2_pids) else None
            plot_one(axes[row_idx, cols_per_group + col], pid, is_last, show_ylabel=(col == 0))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    # Add extra space between condition 1 and condition 2 columns
    for row_idx in range(n_rows):
        for col_idx in range(cols_per_group, n_cols):
            ax = axes[row_idx, col_idx]
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.01, pos.y0, pos.width, pos.height])

    # Column headers (centered over each group's columns)
    fig.text(0.25, 0.98, cond_labels[cond1], ha="center", va="top", fontsize=12, color="#000000")
    fig.text(0.75, 0.98, cond_labels[cond2], ha="center", va="top", fontsize=12, color="#000000")

    fig.savefig(f"original learning rates {cond_labels[cond1]} vs {cond_labels[cond2]}.png", dpi=150, bbox_inches="tight")
    return fig


def visualize_original_aggregate(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    min_lr_1st_day: float | None = None,
    show_side_by_side: bool = True,
    replication_data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    control_data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    tn_data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    show_control: bool = False,
    show_filtered: bool = True,
    y_lim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Visualize aggregate learning curves for P-match vs T-match from original paper data.

    Single day, two or three conditions overlaid.

    Args:
        min_lr_1st_day: If set, exclude participants with LR below this threshold (e.g., 0 to exclude negative LRs)
        show_side_by_side: If True, show both unfiltered and filtered data side-by-side
        replication_data: Optional tuple of (block_acc, slopes) from replication study (day 1 only)
        control_data: Optional tuple of (block_acc, slopes) for arrhythmic control group
        show_control: If True, show control line on aggregate charts (default: False)
        show_filtered: If True, show the middle filtered chart in side-by-side view (default: True)
    """
    colors = {
        "P": "#2d8a2d",  # Green for P-match
        "T": "#1f5f8a",  # Blue for T-match
        "TN": "#9370DB",  # Purple for T-nonMatch
        "C": "#666666",  # Grey for Control
    }
    labels = {
        "P": "P-match",
        "T": "T-match",
        "TN": "T-nonMatch",
        "C": "Arrhythmic Control",
    }

    def plot_aggregate(ax, block_acc_data, slopes_data, title, y_lim=None, control_data_inner=None, tn_data_inner=None):
        """Helper to plot one aggregate view"""
        endpoints = {}

        for cond in ["P", "T"]:
            cond_acc = block_acc_data[block_acc_data["cond"] == cond]
            grp_means = cond_acc.groupby("block")["accuracy"].mean().reset_index()
            blocks = grp_means["block"].values
            acc = grp_means["accuracy"].values

            color = colors[cond]

            # Plot dots
            ax.scatter(blocks, acc, c=color, s=30, zorder=3)

            # Fit and plot curve
            fit = fit_learning_rate(blocks, acc, method="ols")
            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, fit.a, fit.b)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, zorder=2)

            endpoints[cond] = (x_fit[-1], y_fit[-1], fit.b)

        # Add T-nonMatch if provided
        if tn_data_inner is not None:
            tn_block_acc, tn_slopes = tn_data_inner
            tn_acc = tn_block_acc[tn_block_acc["cond"] == "TN"]
            grp_means = tn_acc.groupby("block")["accuracy"].mean().reset_index()
            blocks = grp_means["block"].values
            acc = grp_means["accuracy"].values

            color = colors["TN"]

            # Plot dots
            ax.scatter(blocks, acc, c=color, s=30, zorder=3)

            # Fit and plot curve
            fit = fit_learning_rate(blocks, acc, method="ols")
            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, fit.a, fit.b)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, zorder=2)

            endpoints["TN"] = (x_fit[-1], y_fit[-1], fit.b)

        # Add control if provided
        if control_data_inner is not None:
            ctrl_block_acc, ctrl_slopes = control_data_inner
            ctrl_acc = ctrl_block_acc[ctrl_block_acc["cond"] == "C"]
            grp_means = ctrl_acc.groupby("block")["accuracy"].mean().reset_index()
            blocks = grp_means["block"].values
            acc = grp_means["accuracy"].values

            color = colors["C"]

            # Plot dots
            ax.scatter(blocks, acc, c=color, s=30, zorder=3)

            # Fit and plot curve
            fit = fit_learning_rate(blocks, acc, method="ols")
            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, fit.a, fit.b)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, zorder=2)

            endpoints["C"] = (x_fit[-1], y_fit[-1], fit.b)

        # Direct labeling: T-match on top, TN below T, P-match on bottom, Control in middle
        for cond, (x, y, lr) in endpoints.items():
            color = colors[cond]
            if cond == "T":
                # T-match: position above the curve
                y_offset = 2.0
                va = "bottom"
            elif cond == "TN":
                # T-nonMatch: position between T and P
                y_offset = 0.5
                va = "center"
            elif cond == "C":
                # Control: position in middle/right side
                y_offset = 0.0
                va = "center"
            else:  # P-match
                # P-match: position below the curve
                y_offset = -2.0
                va = "top"
            ax.text(x - 0.5, y + y_offset, f"{labels[cond]}  LR={lr:.1f}",
                    color=color, fontsize=9, va=va, ha="center")

        # Tufte-style axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.set_xlim(0.5, 8.5)
        ax.set_xticks(range(1, 9))
        ax.tick_params(axis="x", length=3, width=0.5)
        ax.set_xlabel("Block", fontsize=10, color="black")

        ax.set_ylabel("Accuracy (%)", fontsize=10, color="black")
        ax.tick_params(axis="y", colors="black", length=3, width=0.5)

        # Set y-axis limits if provided
        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.set_title(title, fontsize=11)

    # Show side-by-side comparison if requested
    if show_side_by_side and min_lr_1st_day is not None:
        # Determine number of subplots based on what we're showing
        n_plots = 1  # Always show "all participants"
        if show_filtered:
            n_plots += 1
        if replication_data is not None:
            n_plots += 1

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]  # Make it iterable

        ax_idx = 0  # Track which axis we're using

        # Compute shared y-axis range from all data (original + replication + control if provided)
        all_means = block_acc.groupby(["cond", "block"])["accuracy"].mean()
        y_min = all_means.min()
        y_max = all_means.max()

        if replication_data is not None:
            rep_block_acc, rep_slopes = replication_data
            # Filter to day 1 only
            rep_day1 = rep_block_acc[rep_block_acc["day_index"] == 1]
            rep_means = rep_day1.groupby(["cond", "block"])["accuracy"].mean()
            y_min = min(y_min, rep_means.min())
            y_max = max(y_max, rep_means.max())

        if control_data is not None and show_control:
            ctrl_block_acc, ctrl_slopes = control_data
            ctrl_means = ctrl_block_acc.groupby(["cond", "block"])["accuracy"].mean()
            y_min = min(y_min, ctrl_means.min())
            y_max = max(y_max, ctrl_means.max())

        y_padding = (y_max - y_min) * 0.1
        y_lim = (y_min - y_padding, y_max + y_padding)

        # Decide whether to show control and TN
        ctrl_data_to_show = control_data if show_control else None
        tn_data_to_show = tn_data  # Always pass if provided

        # Left: All data from original paper
        plot_aggregate(axes[ax_idx], block_acc, slopes, "Original: All participants", y_lim=y_lim, control_data_inner=ctrl_data_to_show, tn_data_inner=tn_data_to_show)
        ax_idx += 1

        # Middle: Filtered data from original paper (optional)
        if show_filtered:
            valid_pids = slopes[slopes["b"] >= min_lr_1st_day]["participant_id"].unique()
            block_acc_filtered = block_acc[block_acc["participant_id"].isin(valid_pids)]
            slopes_filtered = slopes[slopes["participant_id"].isin(valid_pids)]
            n_excluded = len(slopes["participant_id"].unique()) - len(valid_pids)
            plot_aggregate(axes[ax_idx], block_acc_filtered, slopes_filtered,
                          f"Original: Filtered (LR ≥ {min_lr_1st_day}, n={n_excluded} excluded)", y_lim=y_lim, control_data_inner=ctrl_data_to_show, tn_data_inner=tn_data_to_show)
            ax_idx += 1

        # Right: Replication day 1 (if provided) - no control or TN lines here
        if replication_data is not None:
            rep_block_acc, rep_slopes = replication_data
            # Filter to day 1 only
            rep_day1_acc = rep_block_acc[rep_block_acc["day_index"] == 1].copy()
            rep_day1_slopes = rep_slopes[rep_slopes["day_index"] == 1].copy()
            plot_aggregate(axes[ax_idx], rep_day1_acc, rep_day1_slopes, "Replication: Day 1", y_lim=y_lim, control_data_inner=None, tn_data_inner=None)
            ax_idx += 1


        fig.suptitle("Learning Rate Comparison: Original vs Replication", fontsize=14, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 1))
    else:
        # Original single-panel behavior
        fig_size = figsize if figsize is not None else (8, 5)
        fig, ax = plt.subplots(figsize=fig_size)

        # Filter participants by minimum LR if specified
        if min_lr_1st_day is not None:
            valid_pids = slopes[slopes["b"] >= min_lr_1st_day]["participant_id"].unique()
            block_acc = block_acc[block_acc["participant_id"].isin(valid_pids)]
            n_excluded = len(slopes) - len(valid_pids)
            title_suffix = f" (excluded {n_excluded} with LR < {min_lr_1st_day})"
        else:
            title_suffix = ""

        ctrl_data_to_show = control_data if show_control else None
        tn_data_to_show = tn_data  # Always pass if provided
        plot_aggregate(ax, block_acc, slopes, f"Original Paper Data: P-match vs T-match{title_suffix}", y_lim=y_lim, control_data_inner=ctrl_data_to_show, tn_data_inner=tn_data_to_show)
        fig.tight_layout()

    return fig


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
    parser.add_argument(
        "--fit-method",
        type=str,
        choices=["ols", "l1"],
        default="ols",
        help="Fitting method: 'ols' (L2 loss, default) or 'l1' (L1 loss / median regression)"
    )
    parser.add_argument(
        "--drop-first-n-blocks",
        type=int,
        default=0,
        help="Drop the first N blocks from each session before fitting (to exclude warm-up effects)"
    )
    parser.add_argument(
        "--drop-first-n-trials",
        type=int,
        default=0,
        help="Drop the first N trials from each session (to simulate practice trials, e.g., 50)"
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=0,
        metavar="SIZE",
        help="Use sliding window of SIZE trials instead of fixed blocks (e.g., 100). Produces ~64 points per session."
    )
    parser.add_argument(
        "--original-data-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Load and visualize original paper data (Michael et al., 2023) from DIR instead of study DB"
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

    # Drop first N trials if requested (simulates practice block)
    if args.drop_first_n_trials > 0:
        print(f"  Dropping first {args.drop_first_n_trials} trials from each session")
        df = drop_first_n_trials(df, args.drop_first_n_trials)
        print(f"  {df.shape[0]} trials remaining")

    # Add day index
    df = add_day_index(df)

    # Compute block accuracies (scaled to 0-100%)
    if args.sliding_window > 0:
        print(f"  Using sliding window of {args.sliding_window} trials")
        block_acc = compute_sliding_window_accuracy(df, window_size=args.sliding_window)
        use_sliding = True
    else:
        block_acc = (
            df.groupby(["participant_id", "session_id", "start_ts", "block", "day_index", "cond"])["correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        block_acc["accuracy"] = block_acc["accuracy"] * 100
        use_sliding = False

    # Fit learning rates
    print(f"Fitting with method: {args.fit_method}")
    if args.drop_first_n_blocks > 0 and not use_sliding:
        print(f"  Dropping first {args.drop_first_n_blocks} block(s) from each session")
    slopes = fit_slopes_per_session(block_acc, method=args.fit_method, drop_first_n_blocks=0 if use_sliding else args.drop_first_n_blocks)

    # Handle original paper data mode (after replication data loaded, so we can compare)
    if args.original_data_dir is not None:
        print(f"\nLoading original paper data from {args.original_data_dir}...")

        # Load P-match vs T-match
        orig_block_acc, orig_slopes = load_original_paper_data(args.original_data_dir)
        print(f"Loaded {len(orig_slopes)} participants (P-match and T-match)")

        # Print fitted slopes
        print("\n=== Original Paper Fitted slopes (P-match vs T-match) ===")
        print(orig_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

        # Load T-match vs Control (needed for aggregate chart)
        print("\nLoading T-match vs Control comparison...")
        tc_block_acc, tc_slopes = load_original_paper_data(args.original_data_dir, groups={2: "T", 4: "C"})
        print(f"Loaded {len(tc_slopes)} participants (T-match and Control)")

        # Extract control data for aggregate chart
        control_block_acc = tc_block_acc[tc_block_acc["cond"] == "C"]
        control_slopes = tc_slopes[tc_slopes["cond"] == "C"]

        # Load T-nonMatch (needed for specific aggregate chart)
        print("\nLoading T-nonMatch...")
        tn_block_acc, tn_slopes = load_original_paper_data(args.original_data_dir, groups={3: "TN"})
        print(f"Loaded {len(tn_slopes)} participants (T-nonMatch)")

        # Visualize P-match vs T-match individual curves
        visualize_original_individual_curves(orig_block_acc, orig_slopes, max_per_group=20, cols_per_group=3)

        # Show original data only with paper's y-axis range (0.55-0.7)
        print("\nGenerating aggregate chart for original data only (y-axis: 55-70%)...")
        visualize_original_aggregate(orig_block_acc, orig_slopes,
                                     show_side_by_side=False, replication_data=None,
                                     control_data=(control_block_acc, control_slopes),
                                     tn_data=(tn_block_acc, tn_slopes),
                                     show_control=False, show_filtered=False,
                                     y_lim=(55, 70), figsize=(6, 6))

        # Show comparison with replication data (2 columns: original all + replication)
        visualize_original_aggregate(orig_block_acc, orig_slopes, min_lr_1st_day=-1,
                                     show_side_by_side=True, replication_data=(block_acc, slopes),
                                     control_data=(control_block_acc, control_slopes),
                                     show_control=False, show_filtered=False)

        # Print fitted slopes
        print("\n=== Original Paper Fitted slopes (T-match vs Control) ===")
        print(tc_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

        # Visualize T-match vs Control
        visualize_original_individual_curves(tc_block_acc, tc_slopes, max_per_group=20, cols_per_group=3,
                                            cond_labels={"T": "T-match", "C": "Arrhythmic Control"})

        # Load P-match vs Control
        print("\nLoading P-match vs Control comparison...")
        pc_block_acc, pc_slopes = load_original_paper_data(args.original_data_dir, groups={1: "P", 4: "C"})
        print(f"Loaded {len(pc_slopes)} participants (P-match and Control)")

        # Print fitted slopes
        print("\n=== Original Paper Fitted slopes (P-match vs Control) ===")
        print(pc_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

        # Visualize P-match vs Control
        visualize_original_individual_curves(pc_block_acc, pc_slopes, max_per_group=20, cols_per_group=3,
                                            cond_labels={"P": "P-match", "C": "Arrhythmic Control"})

        # Load T-nonmatch data for strip plot
        print("\nLoading T-nonmatch data...")
        tn_block_acc, tn_slopes = load_original_paper_data(args.original_data_dir, groups={3: "TN"})
        print(f"Loaded {len(tn_slopes)} participants (T-nonmatch)")

        # Prepare data for strip plot - use PROVIDED learning rates for original data
        print("\nLoading provided learning rates from groupLR_forLMM.csv...")
        provided_slopes = load_provided_learning_rates(args.original_data_dir)

        if provided_slopes is None:
            raise FileNotFoundError(
                f"groupLR_forLMM.csv not found in {args.original_data_dir}. "
                "This file is required for the strip plot to use the paper's official learning rates."
            )

        print("Using provided learning rates for original data (from groupLR_forLMM.csv)")
        p_slopes_only = provided_slopes[provided_slopes["cond"] == "P"]
        t_slopes_only = provided_slopes[provided_slopes["cond"] == "T"]
        tn_slopes_only = provided_slopes[provided_slopes["cond"] == "TN"]
        # Control not in groupLR_forLMM.csv, use computed
        c_slopes_only = tc_slopes[tc_slopes["cond"] == "C"]

        # Strip plot of individual learning rates by group
        print("\nGenerating learning rate strip plot...")
        rep_day1_slopes_for_strip = slopes[slopes["day_index"] == 1]
        rep_p_slopes_only = rep_day1_slopes_for_strip[rep_day1_slopes_for_strip["cond"] == "P"]
        rep_t_slopes_only = rep_day1_slopes_for_strip[rep_day1_slopes_for_strip["cond"] == "T"]
        visualize_lr_strip_plot(
            p_slopes_only, t_slopes_only, tn_slopes_only, c_slopes_only,
            rep_p_slopes_only, rep_t_slopes_only
        )

        # T-tests: Compare replication vs original learning rates
        print("\n=== T-tests: Replication vs Original Learning Rates ===")
        from scipy import stats

        # Filter to day 1 only for replication
        rep_day1_slopes = slopes[slopes["day_index"] == 1]

        # T-match comparison
        orig_t_lr = orig_slopes[orig_slopes["cond"] == "T"]["b"].values
        rep_t_lr = rep_day1_slopes[rep_day1_slopes["cond"] == "T"]["b"].values
        t_result = stats.ttest_ind(rep_t_lr, orig_t_lr, equal_var=False)
        print(f"\nT-match: Replication (n={len(rep_t_lr)}, mean={rep_t_lr.mean():.3f}) vs Original (n={len(orig_t_lr)}, mean={orig_t_lr.mean():.3f})")
        print(f"  t({t_result.df:.1f}) = {t_result.statistic:.3f}, p = {t_result.pvalue:.4f}")

        # P-match comparison (all participants)
        orig_p_lr = orig_slopes[orig_slopes["cond"] == "P"]["b"].values
        rep_p_lr = rep_day1_slopes[rep_day1_slopes["cond"] == "P"]["b"].values
        p_result = stats.ttest_ind(rep_p_lr, orig_p_lr, equal_var=False)
        print(f"\nP-match (all): Replication (n={len(rep_p_lr)}, mean={rep_p_lr.mean():.3f}) vs Original (n={len(orig_p_lr)}, mean={orig_p_lr.mean():.3f})")
        print(f"  t({p_result.df:.1f}) = {p_result.statistic:.3f}, p = {p_result.pvalue:.4f}")

        # P-match comparison (filtered: LR >= -1 on day 1)
        orig_p_filtered_lr = orig_slopes[(orig_slopes["cond"] == "P") & (orig_slopes["b"] >= -1)]["b"].values
        rep_p_filtered_lr = rep_day1_slopes[(rep_day1_slopes["cond"] == "P") & (rep_day1_slopes["b"] >= -1)]["b"].values
        pf_result = stats.ttest_ind(rep_p_filtered_lr, orig_p_filtered_lr, equal_var=False)
        print(f"\nP-match (filtered LR≥-1): Replication (n={len(rep_p_filtered_lr)}, mean={rep_p_filtered_lr.mean():.3f}) vs Original (n={len(orig_p_filtered_lr)}, mean={orig_p_filtered_lr.mean():.3f})")
        print(f"  t({pf_result.df:.1f}) = {pf_result.statistic:.3f}, p = {pf_result.pvalue:.4f}")

        # Mixed-effects ANCOVA: P-match replication vs original controlling for offset
        print("\n=== ANCOVA: P-match (all) LR ~ study + offset ===")
        print("Question: Is replication P-match LR higher than original, after controlling for offset?")

        # Combine original and replication P-match data (all participants)
        orig_p_data = orig_slopes[orig_slopes["cond"] == "P"].copy()
        orig_p_data["study"] = "original"

        rep_p_data = rep_day1_slopes[rep_day1_slopes["cond"] == "P"].copy()
        rep_p_data["study"] = "replication"

        combined_p = pd.concat([orig_p_data, rep_p_data], ignore_index=True)
        combined_p["study_replication"] = (combined_p["study"] == "replication").astype(int)
        combined_p["a_centered"] = combined_p["a"] - combined_p["a"].mean()

        # Fit model: LR ~ study + offset
        # Note: No random effects needed since each participant is in only one study
        import statsmodels.formula.api as smf
        model = smf.ols("b ~ study_replication + a_centered", data=combined_p)
        result = model.fit()

        print(result.summary())

        coef_study = result.params["study_replication"]
        pval_study = result.pvalues["study_replication"]
        coef_offset = result.params["a_centered"]
        pval_offset = result.pvalues["a_centered"]

        print("\n--- Interpretation ---")
        print(f"Study effect (replication vs original): {coef_study:.4f} (p = {pval_study:.4f})")
        print(f"  -> At the same offset, replication P-match LR is {coef_study:.4f} {'higher' if coef_study > 0 else 'lower'} than original")
        print(f"Offset effect: {coef_offset:.4f} (p = {pval_offset:.4f})")
        print(f"  -> For each 1% increase in offset, LR changes by {coef_offset:.4f}")

        if pval_study < 0.05:
            print("\nConclusion: Replication effect is significant even after controlling for offset.")
        else:
            print("\nConclusion: Replication effect is NOT significant after controlling for offset.")

        # ANCOVA on filtered data (LR >= -1)
        print("\n=== ANCOVA: P-match (filtered LR≥-1) LR ~ study + offset ===")
        print("Question: Same as above, but excluding participants with LR < -1")

        # Combine filtered P-match data
        orig_p_filtered_data = orig_slopes[(orig_slopes["cond"] == "P") & (orig_slopes["b"] >= -1)].copy()
        orig_p_filtered_data["study"] = "original"

        rep_p_filtered_data = rep_day1_slopes[(rep_day1_slopes["cond"] == "P") & (rep_day1_slopes["b"] >= -1)].copy()
        rep_p_filtered_data["study"] = "replication"

        combined_p_filtered = pd.concat([orig_p_filtered_data, rep_p_filtered_data], ignore_index=True)
        combined_p_filtered["study_replication"] = (combined_p_filtered["study"] == "replication").astype(int)
        combined_p_filtered["a_centered"] = combined_p_filtered["a"] - combined_p_filtered["a"].mean()

        # Fit model on filtered data
        model_filtered = smf.ols("b ~ study_replication + a_centered", data=combined_p_filtered)
        result_filtered = model_filtered.fit()

        print(result_filtered.summary())

        coef_study_f = result_filtered.params["study_replication"]
        pval_study_f = result_filtered.pvalues["study_replication"]
        coef_offset_f = result_filtered.params["a_centered"]
        pval_offset_f = result_filtered.pvalues["a_centered"]

        print("\n--- Interpretation ---")
        print(f"Study effect (replication vs original): {coef_study_f:.4f} (p = {pval_study_f:.4f})")
        print(f"  -> At the same offset, replication P-match LR is {coef_study_f:.4f} {'higher' if coef_study_f > 0 else 'lower'} than original")
        print(f"Offset effect: {coef_offset_f:.4f} (p = {pval_offset_f:.4f})")
        print(f"  -> For each 1% increase in offset, LR changes by {coef_offset_f:.4f}")

        if pval_study_f < 0.05:
            print("\nConclusion: Replication effect is significant even after controlling for offset (filtered data).")
        else:
            print("\nConclusion: Replication effect is NOT significant after controlling for offset (filtered data).")

        plt.show()
        return

    # Print fitted slopes
    print("\n=== Fitted slopes ===")
    print(slopes.sort_values(["participant_id", "day_index", "cond"]).to_string(index=False))

    # Run hypothesis tests
    run_h1_within_subject(slopes)
    run_h2_between_groups_day1(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=False)
    run_h3_between_groups_day1_intercept(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=True)
    run_h3_between_groups_day2_intercept(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=True)
    run_ancova_lr_controlling_offset(slopes)

    # Visualize
    drop_n = 0 if use_sliding else args.drop_first_n_blocks
    visualize_learning_curves(block_acc, slopes, drop_first_n_blocks=drop_n, use_sliding=use_sliding)
    visualize_learning_curves(block_acc, slopes, drop_first_n_blocks=drop_n, use_sliding=use_sliding, day1_only=True)  # Day 1 only comparison
    visualize_offset_vs_lr(slopes)
    visualize_group_average_both_days(block_acc, slopes, drop_first_n_blocks=drop_n, use_sliding=use_sliding)
    visualize_group_average_both_days(block_acc, slopes, drop_first_n_blocks=drop_n, use_sliding=use_sliding, min_lr_1st_day=0)  # Exclude negative day 1 LRs
    plt.show()


if __name__ == "__main__":
    main()