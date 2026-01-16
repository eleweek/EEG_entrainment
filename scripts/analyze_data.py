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


def visualize_group_average_both_days(block_acc: pd.DataFrame, slopes: pd.DataFrame, drop_first_n_blocks: int = 0, use_sliding: bool = False) -> plt.Figure:
    """
    Plot average block accuracies for T-first and P-first groups on both days.

    - Day 1: blocks 1-8, Day 2: blocks 9-16 (continuous x-axis)
    - T-first in blue shades, P-first in green shades
    - Day 1 = dark, Day 2 = light
    - Fit log-linear curves separately for each day
    - Tufte-style: minimal, direct labeling
    """
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

    fig.tight_layout()
    return fig


def visualize_learning_curves(block_acc: pd.DataFrame, slopes: pd.DataFrame, drop_first_n_blocks: int = 0, use_sliding: bool = False) -> plt.Figure:
    """
    Create visualization of accuracy and learning rate fits for all participants.

    Layout: 4 columns per row
      - Columns 0-1: P-first participant (Day 1, Day 2)
      - Columns 2-3: T-first participant (Day 1, Day 2)
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

    # 4 columns: P-first Day1, P-first Day2, T-first Day1, T-first Day2
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 1.25 * n_rows), squeeze=False)

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

def load_original_paper_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and convert original paper data (Michael et al., 2023) to our format.

    The paper data has:
    - AccPerLat.csv: accuracy per block, 3 rows per subject (3 latency conditions)
    - groupID: 1=Peak-Match (P), 2=Trough-Match (T), 3=Trough-NonMatch, 4=Control

    We focus on P-match (groupID=1) vs T-match (groupID=2).

    Returns:
        block_acc: DataFrame with [participant_id, cond, block, accuracy]
        slopes: DataFrame with fitted slopes per participant
    """
    import os

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
    # Focus on P-match (1) and T-match (2) only
    rows = []
    for gid in [1, 2]:  # P-match, T-match
        cond = "P" if gid == 1 else "T"
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
    for pid in block_acc["participant_id"].unique():
        p_data = block_acc[block_acc["participant_id"] == pid]
        cond = p_data["cond"].iloc[0]

        fr = fit_learning_rate(
            p_data["block"].to_numpy(float),
            p_data["accuracy"].to_numpy(float),
            method="ols",
        )
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

    return block_acc, slopes


def visualize_original_individual_curves(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    max_per_group: int = 6,
    cols_per_group: int = 3,
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize individual learning curves from original paper data.

    Layout: cols_per_group columns for P-match (left) + cols_per_group columns for T-match (right),
    limited to max_per_group random samples per condition.
    """
    np.random.seed(seed)

    # Split by condition
    p_pids = slopes[slopes["cond"] == "P"]["participant_id"].unique()
    t_pids = slopes[slopes["cond"] == "T"]["participant_id"].unique()

    # Sample if needed
    if len(p_pids) > max_per_group:
        p_pids = np.random.choice(p_pids, max_per_group, replace=False)
    if len(t_pids) > max_per_group:
        t_pids = np.random.choice(t_pids, max_per_group, replace=False)

    # Layout: cols_per_group P-match + cols_per_group T-match per row
    n_cols = cols_per_group * 2
    n_rows = max(math.ceil(len(p_pids) / cols_per_group), math.ceil(len(t_pids) / cols_per_group))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.5 * n_rows), squeeze=False)

    # Global y-axis range across displayed participants only
    displayed_pids = list(p_pids) + list(t_pids)
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

        ax.scatter(blocks, acc, s=12, color=data_color, zorder=3)

        fit_annotation = ""
        if not p_slope.empty:
            a = p_slope["a"].iloc[0]
            b = p_slope["b"].iloc[0]
            r2 = p_slope["r2"].iloc[0]

            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, a, b)
            ax.plot(x_fit, y_fit, color=fit_color, linewidth=1, zorder=2)
            fit_annotation = f"  LR={b:.2f}  R²={r2:.2f}"

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#000000")
        ax.set_title(f"{pid}{fit_annotation}", fontsize=9, loc="left", color="#000000")

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

        # P-match: columns 0 to cols_per_group-1
        for col in range(cols_per_group):
            p_idx = row_idx * cols_per_group + col
            pid = p_pids[p_idx] if p_idx < len(p_pids) else None
            plot_one(axes[row_idx, col], pid, is_last, show_ylabel=(col == 0))

        # T-match: columns cols_per_group to n_cols-1
        for col in range(cols_per_group):
            t_idx = row_idx * cols_per_group + col
            pid = t_pids[t_idx] if t_idx < len(t_pids) else None
            plot_one(axes[row_idx, cols_per_group + col], pid, is_last, show_ylabel=(col == 0))

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Column headers (centered over each group's columns)
    fig.text(0.25, 0.98, "P-match", ha="center", va="top", fontsize=12, color="#000000")
    fig.text(0.75, 0.98, "T-match", ha="center", va="top", fontsize=12, color="#000000")

    return fig


def visualize_original_aggregate(block_acc: pd.DataFrame, slopes: pd.DataFrame) -> plt.Figure:
    """
    Visualize aggregate learning curves for P-match vs T-match from original paper data.

    Single day, two conditions overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "P": "#2d8a2d",  # Green for P-match
        "T": "#1f5f8a",  # Blue for T-match
    }
    labels = {
        "P": "P-match (Peak)",
        "T": "T-match (Trough)",
    }

    endpoints = {}

    for cond in ["P", "T"]:
        cond_acc = block_acc[block_acc["cond"] == cond]
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

    # Direct labeling
    for cond, (x, y, lr) in endpoints.items():
        color = colors[cond]
        y_offset = 1.5 if cond == "T" else -1.5
        va = "bottom" if cond == "T" else "top"
        ax.text(x + 0.2, y + y_offset, f"{labels[cond]}  LR={lr:.1f}",
                color=color, fontsize=9, va=va, ha="left")

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

    ax.set_title("Original Paper Data: P-match vs T-match (Michael et al., 2023)", fontsize=11)

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

    # Handle original paper data mode
    if args.original_data_dir is not None:
        print(f"Loading original paper data from {args.original_data_dir}...")
        block_acc, slopes = load_original_paper_data(args.original_data_dir)
        print(f"Loaded {len(slopes)} participants (P-match and T-match)")

        # Print fitted slopes
        print("\n=== Fitted slopes ===")
        print(slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

        # Visualize
        visualize_original_individual_curves(block_acc, slopes, max_per_group=18, cols_per_group=3)
        visualize_original_aggregate(block_acc, slopes)
        plt.show()
        return

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
    visualize_offset_vs_lr(slopes)
    visualize_group_average_both_days(block_acc, slopes, drop_first_n_blocks=drop_n, use_sliding=use_sliding)
    plt.show()


if __name__ == "__main__":
    main()