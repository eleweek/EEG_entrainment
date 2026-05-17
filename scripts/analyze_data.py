from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
import pingouin as pg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg


OUTPUT_DIR = "_generated_charts"
PUBLIC_ID_TABLE = "participant_public_id_mapping"


def _save(fig: plt.Figure, filename: str, dpi: int = 450) -> None:
    """Save a figure into OUTPUT_DIR (created on first use)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stem, _ = os.path.splitext(filename)
    svg_path = os.path.join(OUTPUT_DIR, f"{stem}.svg")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg_path, format="svg", bbox_inches="tight")

    webp_path = os.path.join(OUTPUT_DIR, f"{stem}.webp")
    fig.savefig(webp_path, format="webp", dpi=dpi, bbox_inches="tight")


def _fmt_lr(lr: float, digits: int = 2) -> str:
    """Format a learning rate, collapsing tiny negatives so '-0.0' never appears."""
    s = f"{lr:.{digits}f}"
    if s == f"-{0.0:.{digits}f}":  # -0.0, -0.00, ...
        s = f"{0.0:.{digits}f}"
    return s


def _accuracy_formatter() -> PercentFormatter:
    """Formatter for axes that display accuracy values on a 0-100 scale."""
    return PercentFormatter(xmax=100, decimals=0)


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


def load_public_participant_id_mapping(db_path: str) -> dict[str, str]:
    """Load internal -> public participant ID mapping from the SQLite DB."""
    with sqlite3.connect(db_path) as con:
        table_exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (PUBLIC_ID_TABLE,),
        ).fetchone()
        if table_exists is None:
            raise ValueError(
                f"Missing {PUBLIC_ID_TABLE!r} table in {db_path}. "
                "Run scripts/create_public_participant_ids.py first, "
                "or pass --use-internal-ids for private/debug charts."
            )

        rows = con.execute(
            f"""
            SELECT participant_id, public_participant_id
            FROM {PUBLIC_ID_TABLE}
            """
        ).fetchall()

    return {participant_id: public_id for participant_id, public_id in rows}


def replace_with_public_participant_ids(block_acc: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """Replace internal participant IDs with stable public IDs for plotting/export-like output."""
    public_id_mapping = load_public_participant_id_mapping(db_path)
    missing = sorted(set(block_acc["participant_id"]) - set(public_id_mapping))
    if missing:
        raise ValueError(
            "Missing public participant IDs for: "
            + ", ".join(missing)
            + f". Recreate or repair {PUBLIC_ID_TABLE!r}, "
            + "or pass --use-internal-ids for private/debug charts."
        )

    block_acc = block_acc.copy()
    block_acc["participant_id"] = block_acc["participant_id"].map(public_id_mapping)
    return block_acc


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


def fit_slopes_per_session(block_acc: pd.DataFrame, method: str = "ols") -> pd.DataFrame:
    """Fit one log-linear curve per (participant_id, day_index).

    Assumes a single session per (participant, day). If `session_id` is
    present, raises ValueError when that assumption is violated.
    """
    rows = []
    has_session_id = "session_id" in block_acc.columns

    for (pid, day), sub in block_acc.groupby(["participant_id", "day_index"], sort=True):
        sub = sub.sort_values("block")

        if has_session_id:
            unique_sids = sub["session_id"].unique()
            if len(unique_sids) != 1:
                raise ValueError(
                    f"Expected exactly one session_id for (participant_id={pid!r}, "
                    f"day_index={day}), got {len(unique_sids)}: {list(unique_sids)}"
                )
            sid = unique_sids[0]
        else:
            sid = None

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

    col_label = {"a": "intercept a", "b": "slope b", "accuracy": "initial accuracy"}.get(column, column)
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


def run_h3_between_groups_day1_initial_accuracy(
    block_acc: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = 0,
    two_sided: bool = False,
) -> None:
    """H3 variant: Day-1 between-groups comparison on initial accuracy (block 1)."""
    initial = (
        block_acc[block_acc["block"] == 1]
        [["participant_id", "day_index", "cond", "accuracy"]]
        .copy()
    )
    run_between_groups_test(
        initial,
        day_index=1,
        column="accuracy",
        label="H3 initial accuracy (exploratory)",
        n_perm=n_perm,
        seed=seed,
        two_sided=two_sided,
    )


def visualize_replication_aggregate_both_days(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
) -> plt.Figure:
    """
    Plot average block accuracies for T-first and P-first groups on both days.

    - Day 1: blocks 1-8, Day 2: blocks 9-16 (continuous x-axis)
    - T-first in blue shades, P-first in green shades
    - Day 1 = dark, Day 2 = a bit lighter
    - Fit log-linear curves separately for each day
    - Tufte-style: minimal, direct labeling
    """
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

    # Colors: T-first = blue, P-first = green; Day 1 = dark, Day 2 = a bit lighter
    colors = {
        ("P", 1): "#2d8a2d",  # P-first, Day 1 (dark green)
        ("P", 2): "#6cc16c",  # P-first, Day 2 (slightly lighter green)
        ("T", 1): "#1f5f8a",  # T-first, Day 1 (dark blue)
        ("T", 2): "#5a9bc9",  # T-first, Day 2 (slightly lighter blue)
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
            max_block = 8
            blocks_plot = blocks_original + (max_block if day == 2 else 0)

            color = colors[(group, day)]

            # Plot dots - small, unobtrusive
            ax.scatter(blocks_plot, acc, c=color, s=16, zorder=3)

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
    # Right-edge of each label aligns with the rightmost edge of the dot at the
    # endpoint, not its center. dot_size=16 in scatter -> radius ~= sqrt(16/pi) pts.
    from matplotlib.transforms import offset_copy
    dot_radius_pts = float(np.sqrt(16 / np.pi))
    label_trans = offset_copy(ax.transData, fig=fig, x=dot_radius_pts, units="points")

    for (group, day), (x, y, lr) in endpoints.items():
        color = colors[(group, day)]
        # T-first: label above (positive offset), P-first: label below (negative offset)
        y_offset = 2 if group == "T" else -2
        va = "bottom" if group == "T" else "top"
        ax.text(x, y + y_offset, f"{label_names[(group, day)]}  LR={_fmt_lr(lr)}",
                color=color, fontsize=9, va=va, ha="right",
                transform=label_trans)

    # Minimal day separator (at boundary between Day 1 and Day 2)
    n_blocks = 8
    ax.axvline(x=n_blocks + 0.5, color="#dddddd", linewidth=0.5, zorder=0)

    # Day-section headers, inline at top-left of each half (spaghetti-style)
    ax.text(1, 1, "Replication, Day 1", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=11, fontweight="bold", color="black")
    ax.text(n_blocks + 1, 1, "Day 2", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=11, fontweight="bold", color="black")

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

    ax.set_ylabel("Accuracy", fontsize=9, color="black")
    ax.tick_params(axis="y", colors="black", length=3, width=0.5)
    ax.yaxis.set_major_formatter(_accuracy_formatter())

    fig.tight_layout()
    _save(fig, "replication_aggregate_both_days.png")
    return fig


def visualize_learning_curves(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
) -> plt.Figure:
    """
    Create visualization of accuracy and learning rate fits for all participants.

    Layout: 4 columns per row:
      - Columns 0-1: T-first (Day 1, Day 2)
      - Columns 2-3: P-first (Day 1, Day 2)
    Two participants per row (one from each group).
    """
    n_blocks = 8

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

    panel_size = 2.5
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)

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
            fit_annotation = f"LR={b:.2f}   off={a:.2f}   R²={r2:.2f}"

        # Tufte-style: remove spines, keep only left and bottom
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        # Minimal ticks
        ax.tick_params(axis="both", which="both", length=3, width=0.5, colors="#000000")

        ax.set_xlim(0.5, n_blocks + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(1, n_blocks + 1))
        ax.set_box_aspect(1)
        ax.text(0, -0.05, f"{pid}{fit_annotation}", transform=ax.transAxes,
                fontsize=8, ha="left", va="top", color="#000000", clip_on=False)

        if is_last_row:
            ax.set_xlabel("Block", fontsize=8, color="#000000", labelpad=2)
        else:
            ax.set_xticklabels([])

    for row_idx in range(n_rows):
        # T-first participant (columns 0, 1)
        pid_t = t_first[row_idx] if row_idx < len(t_first) else None
        # P-first participant (columns 2, 3)
        pid_p = p_first[row_idx] if row_idx < len(p_first) else None

        is_last_row = row_idx == n_rows - 1

        # T-first: Day 1 (col 0), Day 2 (col 1)
        plot_participant(axes[row_idx, 0], pid_t, 1, is_last_row)
        plot_participant(axes[row_idx, 1], pid_t, 2, is_last_row)

        # P-first: Day 1 (col 2), Day 2 (col 3)
        plot_participant(axes[row_idx, 2], pid_p, 1, is_last_row)
        plot_participant(axes[row_idx, 3], pid_p, 2, is_last_row)

        # Y tick labels on the leftmost column of every row.
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx != 0:
                ax.set_yticklabels([])

    # Drop x-tick labels and the "Block" xlabel everywhere — the inline
    # participant caption already grounds each panel.
    for ax in axes.flat:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    # Y-axis label only on the top-left chart
    for ax in axes.flat:
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("Accuracy", fontsize=8, color="#000000")
    # Percent formatter on every axis that still shows y-tick labels.
    for row_idx in range(n_rows):
        axes[row_idx, 0].yaxis.set_major_formatter(_accuracy_formatter())

    fig.tight_layout(rect=(0, 0.035, 0.98, 0.965))
    fig.subplots_adjust(wspace=0.08, hspace=0.24)
    # Add extra space between columns 1 and 2 (between T-first and P-first)
    for row_idx in range(n_rows):
        for col_idx in [2, 3]:
            ax = axes[row_idx, col_idx]
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height])

    # Column headers just above each chart (subordinate to group headers)
    col_headers = ["Day 1, T-match", "Day 2, P-match", "Day 1, P-match", "Day 2, T-match"]
    axes_top = max(axes[0, col_idx].get_position().y1 for col_idx in range(4))
    for col_idx, header in enumerate(col_headers):
        ax = axes[0, col_idx]
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, axes_top + 0.002, header,
                 ha="center", va="bottom", fontsize=11, color="#000000")

    # Group headers above the column headers, centered over the actual axes groups.
    t_group_pos0 = axes[0, 0].get_position()
    t_group_pos1 = axes[0, 1].get_position()
    p_group_pos0 = axes[0, 2].get_position()
    p_group_pos1 = axes[0, 3].get_position()
    t_group_center = (t_group_pos0.x0 + t_group_pos1.x1) / 2
    p_group_center = (p_group_pos0.x0 + p_group_pos1.x1) / 2
    group_header_y = axes_top + 0.028
    fig.text(t_group_center, group_header_y, "T-match first", ha="center", va="top",
             fontsize=13, fontweight="bold", color="#000000")
    fig.text(p_group_center, group_header_y, "P-match first", ha="center", va="top",
             fontsize=13, fontweight="bold", color="#000000")

    _save(fig, "learning_curves_per_participant.png")
    return fig


def parse_participant_list(value: str | None) -> list[str] | None:
    """Parse comma-separated participant IDs like 'p001,p002'."""
    if value is None or value.strip() == "":
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def load_replication_data(
    db_path: str | None,
    from_export_path: str | None,
    include_only: list[str] | None = None,
    exclude: list[str] | None = None,
    fit_method: str = "ols",
    use_internal_ids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load replication block accuracy + per-session log-linear fits.

    Source is either an exported accuracy CSV (from_export_path) or the SQLite
    study DB (db_path). Exactly one must be provided. Raises ValueError if
    the source resolves to zero rows after any include/exclude filtering.
    """
    if (db_path is None) == (from_export_path is None):
        raise ValueError("Provide exactly one of db_path or from_export_path")
    if from_export_path is not None and use_internal_ids:
        raise ValueError("--use-internal-ids cannot be used with --from-export")

    if from_export_path is not None:
        print(f"Loading exported accuracy data from {from_export_path}...")
        block_acc = pd.read_csv(from_export_path)

        if include_only is not None:
            block_acc = block_acc[block_acc["participant_id"].isin(include_only)]
        if exclude is not None:
            block_acc = block_acc[~block_acc["participant_id"].isin(exclude)]

        if block_acc.empty:
            raise ValueError(f"No accuracy rows in {from_export_path} after filtering")

        print(f"Loaded {len(block_acc)} accuracy rows from {block_acc['participant_id'].nunique()} participants")
    else:
        print(f"Loading trials from {db_path}...")
        if include_only:
            print(f"  Including only: {', '.join(include_only)}")
        if exclude:
            print(f"  Excluding: {', '.join(exclude)}")

        df = load_trials(db_path, include_only=include_only, exclude=exclude)
        print(f"Loaded {df.shape[0]} trials from {df['participant_id'].nunique()} participants "
              f"across {df['session_id'].nunique()} sessions")

        if df.empty:
            raise ValueError(f"No trials in {db_path} after filtering")

        df = add_day_index(df)
        block_acc = (
            df.groupby(["participant_id", "session_id", "start_ts", "block", "day_index", "cond"])["correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        block_acc["accuracy"] = block_acc["accuracy"] * 100

        if use_internal_ids:
            print("Using internal participant IDs for charts.")
        else:
            print("Using public participant IDs for charts.")
            block_acc = replace_with_public_participant_ids(block_acc, db_path)

    print(f"Fitting with method: {fit_method}")
    slopes = fit_slopes_per_session(block_acc, method=fit_method)
    return block_acc, slopes


# -----------------------------
# Original paper data adapters
# -----------------------------

def refit_approximate_original_paper_curves(
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

    return block_acc, slopes


def load_original_paper_provided_learning_rates(data_dir: str, filename: str = "groupLR_forLMM.csv") -> pd.DataFrame | None:
    """
    Load provided learning rates from a CSV file.

    Returns DataFrame with columns: [participant_id, cond, b] where b is the learning rate.
    Returns None if file doesn't exist.
    """
    import os

    lr_path = os.path.join(data_dir, filename)
    if not os.path.exists(lr_path):
        return None

    df_lr = pd.read_csv(lr_path)
    df_lr.columns = df_lr.columns.str.strip().str.lower()  # normalise

    # Map phase/match to condition codes
    # phase=1, match=1 -> P-match
    # phase=2, match=1 -> T-match
    # phase=2, match=2 -> T-nonMatch

    rows = []
    for _, row in df_lr.iterrows():
        phase = int(row['phase'])
        match = int(row['match'])
        lr = float(row['lr'])
        subid = int(row['subid'])

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


def _render_strip_plot(
    ax: plt.Axes,
    groups_info: list[tuple[str, np.ndarray, str, int]],
    section_labels: dict[int, str] | None = None,
    x_label: str = "Learning Rate",
    first_gap: float | None = None,
):
    """
    Core strip plot renderer. Draws greedy-layered dot strips on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        groups_info: List of (label, values, color, section_id).
        section_labels: Optional dict mapping section_id to bold heading text.
        x_label: X-axis label.
        first_gap: Gap between strip 0 and strip 1 when same section. Default: same as section_break.
    """
    import bisect

    dot_size = 40
    dot_spacing = 0.10
    dot_radius_data = np.sqrt(dot_size) / 50
    overlap_threshold = 2 * dot_radius_data

    # Compute y positions with section gaps
    strip_spacing = 0.7
    section_break = 1.4
    if first_gap is None:
        first_gap = section_break
    y_positions = []
    for i in range(len(groups_info)):
        if i == 0:
            y_positions.append(0)
        elif i == 1 and groups_info[i][3] == groups_info[i - 1][3]:
            y_positions.append(y_positions[-1] + first_gap)
        elif groups_info[i][3] != groups_info[i - 1][3]:
            y_positions.append(y_positions[-1] + section_break)
        else:
            y_positions.append(y_positions[-1] + strip_spacing)

    for group_idx, (label, values, color, section) in enumerate(groups_info):
        y_baseline = y_positions[group_idx]

        sorted_indices = np.argsort(values)
        layers = {}
        dot_layers = {}

        for idx in sorted_indices:
            val = values[idx]
            layer = 0
            while True:
                if layer not in layers:
                    layers[layer] = []
                fits = True
                for existing_x in layers[layer]:
                    if abs(val - existing_x) < overlap_threshold:
                        fits = False
                        break
                if fits:
                    bisect.insort(layers[layer], val)
                    dot_layers[idx] = layer
                    break
                else:
                    layer += 1

        for idx, val in enumerate(values):
            layer = dot_layers[idx]
            y_pos = y_baseline + layer * dot_spacing
            ax.scatter(val, y_pos, s=dot_size, color=color, alpha=0.7,
                      edgecolors='white', linewidth=0.5, zorder=5)

        mean = np.mean(values)
        max_layer = max(dot_layers.values()) if dot_layers else 0
        line_top = y_baseline + max_layer * dot_spacing + 0.15
        ax.vlines(mean, y_baseline - 0.05, line_top,
                 color=color, linewidth=1, zorder=10)
        is_first = (group_idx == 0)
        is_last = (group_idx == len(groups_info) - 1)
        mean_label = f"mean={mean:.2f}" if is_first or is_last else f"{mean:.2f}"
        ax.text(mean, y_baseline - 0.15, mean_label,
                color=color, fontsize=9, ha="center", va="top", zorder=10)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([g[0] for g in groups_info], fontsize=9)
    ax.set_ylim(-0.5, y_positions[-1] + 0.8)

    if section_labels:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        tick_x = -0.01
        for sec_id, sec_label in section_labels.items():
            sec_indices = [i for i, g in enumerate(groups_info) if g[3] == sec_id]
            if sec_indices:
                ax.text(tick_x, y_positions[sec_indices[-1]] + 0.4, sec_label,
                        transform=trans, fontsize=11, fontweight="bold",
                        ha="right", va="bottom", color="black")

    ax.tick_params(axis="both", colors="black", length=3, width=0.5)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold", color="black", labelpad=10)
    ax.set_ylabel("")

    return y_positions


def visualize_lr_strip_plot(
    orig_p_slopes: pd.DataFrame,
    orig_t_slopes: pd.DataFrame,
    orig_tn_slopes: pd.DataFrame,
    orig_c_slopes: pd.DataFrame,
    rep_p_slopes: pd.DataFrame,
    rep_t_slopes: pd.DataFrame,
    compact: bool = False,
) -> plt.Figure:
    """Dot plot showing individual learning rates for each group.

    Args:
        compact: If True, show only P-match and T-match (no pooled, TnM, or control).
    """
    if compact:
        fig, ax = plt.subplots(figsize=(10, 5))
        groups_info = [
            ("P-match", orig_p_slopes["b"].values, "#2d8a2d", 0),
            ("T-match", orig_t_slopes["b"].values, "#1f5f8a", 0),
            ("P-match", rep_p_slopes["b"].values, "#2d8a2d", 1),
            ("T-match", rep_t_slopes["b"].values, "#1f5f8a", 1),
        ]
        filename = "strip_plot_compact.png"
    else:
        all_orig_values = np.concatenate([
            orig_p_slopes["b"].values, orig_t_slopes["b"].values,
            orig_tn_slopes["b"].values, orig_c_slopes["b"].values,
        ])
        fig, ax = plt.subplots(figsize=(10, 7))
        groups_info = [
            ("Original pooled", all_orig_values, "#333333", 0),
            ("Arrhythmic Control\n(recomputed)", orig_c_slopes["b"].values, "#666666", 0),
            ("T-nonMatch\n(recomputed)", orig_tn_slopes["b"].values, "#9370DB", 0),
            ("P-match", orig_p_slopes["b"].values, "#2d8a2d", 0),
            ("T-match", orig_t_slopes["b"].values, "#1f5f8a", 0),
            ("P-match", rep_p_slopes["b"].values, "#2d8a2d", 1),
            ("T-match", rep_t_slopes["b"].values, "#1f5f8a", 1),
        ]
        filename = "strip_plot.png"

    _render_strip_plot(ax, groups_info,
                       section_labels={0: "Original, Day 1", 1: "Replication, Day 1"},
                       x_label="Learning Rates by Group",
                       first_gap=0.7 if compact else None)

    fig.tight_layout()
    _save(fig, filename)
    return fig


def visualize_validation_strip_plot(
    provided_slopes: pd.DataFrame,
    recomputed_slopes: pd.DataFrame,
    recomputed_tn_slopes: pd.DataFrame,
) -> plt.Figure:
    """
    Strip plot comparing original (provided) vs recomputed learning rates
    for T-match, P-match, and T-nonMatch.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    groups_info = [
        ("Provided", provided_slopes[provided_slopes["cond"] == "TN"]["b"].values, "#9370DB", 0),
        ("Recomputed", recomputed_tn_slopes[recomputed_tn_slopes["cond"] == "TN"]["b"].values, "#c9b1e8", 0),
        ("Provided", provided_slopes[provided_slopes["cond"] == "P"]["b"].values, "#2d8a2d", 1),
        ("Recomputed", recomputed_slopes[recomputed_slopes["cond"] == "P"]["b"].values, "#74c476", 1),
        ("Provided", provided_slopes[provided_slopes["cond"] == "T"]["b"].values, "#1f5f8a", 2),
        ("Recomputed", recomputed_slopes[recomputed_slopes["cond"] == "T"]["b"].values, "#6baed6", 2),
    ]

    _render_strip_plot(ax, groups_info,
                       section_labels={0: "T-nonMatch", 1: "P-match", 2: "T-match"},
                       x_label="Learning Rate",
                       first_gap=0.7)

    fig.tight_layout()
    _save(fig, "validation_strip_plot.png")
    return fig


def visualize_day2_strip_plot(
    day2_slopes: pd.DataFrame,
) -> plt.Figure:
    """Strip plot of Day 2 (second session) learning rates by group."""
    fig, ax = plt.subplots(figsize=(10, 5))

    groups_info = [
        ("T-nonMatch", day2_slopes[day2_slopes["cond"] == "TN"]["b"].values, "#9370DB", 0),
        ("P-match", day2_slopes[day2_slopes["cond"] == "P"]["b"].values, "#2d8a2d", 0),
        ("T-match", day2_slopes[day2_slopes["cond"] == "T"]["b"].values, "#1f5f8a", 0),
    ]

    _render_strip_plot(ax, groups_info,
                       section_labels={0: "Original Study, Day 2"},
                       x_label="Learning Rates by Group")

    fig.tight_layout()
    _save(fig, "strip_plot_day2.png")
    return fig


def visualize_initial_accuracy_strip_plot(
    orig_b1_path: str,
    rep_block_acc: pd.DataFrame | None = None,
    orig_slopes: pd.DataFrame | None = None,
    rep_slopes: pd.DataFrame | None = None,
) -> plt.Figure:
    """Strip plot of first-block accuracy and log-fit intercepts for T-match and P-match."""
    from scipy import stats as sp_stats

    df = pd.read_csv(orig_b1_path)
    # groupID 1=P-match, 2=T-match
    p_acc = df[df["groupID"] == 1]["acc"].values
    t_acc = df[df["groupID"] == 2]["acc"].values

    t_stat, p_val = sp_stats.ttest_ind(t_acc, p_acc, equal_var=False)
    print(f"  Original block 1: T-match={np.mean(t_acc):.1f}±{np.std(t_acc,ddof=1):.1f}, "
          f"P-match={np.mean(p_acc):.1f}±{np.std(p_acc,ddof=1):.1f}, "
          f"Welch t={t_stat:.3f}, p={p_val:.4f}")

    # Build groups bottom-to-top (matplotlib draws upward)
    # Visual order top-to-bottom: Rep Block 1, Rep Fit, Orig Block 1, Orig Fit
    groups_info = []

    # Section 0 (bottom): Original, Fit Intercept
    if orig_slopes is not None:
        orig_p_a = orig_slopes[orig_slopes["cond"] == "P"]["a"].values
        orig_t_a = orig_slopes[orig_slopes["cond"] == "T"]["a"].values
        t_stat_a, p_val_a = sp_stats.ttest_ind(orig_t_a, orig_p_a, equal_var=False)
        print(f"  Original intercept: T-match={np.mean(orig_t_a):.1f}±{np.std(orig_t_a,ddof=1):.1f}, "
              f"P-match={np.mean(orig_p_a):.1f}±{np.std(orig_p_a,ddof=1):.1f}, "
              f"Welch t={t_stat_a:.3f}, p={p_val_a:.4f}")
        groups_info.extend([
            ("P-match", orig_p_a, "#2d8a2d", 0),
            ("T-match", orig_t_a, "#1f5f8a", 0),
        ])

    # Section 1: Original, Block 1
    groups_info.extend([
        ("P-match", p_acc, "#2d8a2d", 1),
        ("T-match", t_acc, "#1f5f8a", 1),
    ])

    # Section 2: Replication, Fit Intercept
    if rep_slopes is not None:
        rep_p_a = rep_slopes[rep_slopes["cond"] == "P"]["a"].values
        rep_t_a = rep_slopes[rep_slopes["cond"] == "T"]["a"].values
        t_stat_ra, p_val_ra = sp_stats.ttest_ind(rep_t_a, rep_p_a, equal_var=False)
        print(f"  Replication intercept: T-match={np.mean(rep_t_a):.1f}±{np.std(rep_t_a,ddof=1):.1f}, "
              f"P-match={np.mean(rep_p_a):.1f}±{np.std(rep_p_a,ddof=1):.1f}, "
              f"Welch t={t_stat_ra:.3f}, p={p_val_ra:.4f}")
        groups_info.extend([
            ("P-match", rep_p_a, "#2d8a2d", 2),
            ("T-match", rep_t_a, "#1f5f8a", 2),
        ])

    # Section 3 (top): Replication, Block 1
    if rep_block_acc is not None:
        rep_b1 = rep_block_acc[rep_block_acc["block"] == 1]
        rep_p = rep_b1[rep_b1["cond"] == "P"]["accuracy"].values
        rep_t = rep_b1[rep_b1["cond"] == "T"]["accuracy"].values
        t_stat_r, p_val_r = sp_stats.ttest_ind(rep_t, rep_p, equal_var=False)
        print(f"  Replication block 1: T-match={np.mean(rep_t):.1f}±{np.std(rep_t,ddof=1):.1f}, "
              f"P-match={np.mean(rep_p):.1f}±{np.std(rep_p,ddof=1):.1f}, "
              f"Welch t={t_stat_r:.3f}, p={p_val_r:.4f}")
        groups_info.extend([
            ("P-match", rep_p, "#2d8a2d", 3),
            ("T-match", rep_t, "#1f5f8a", 3),
        ])

    section_labels = {0: "Original, Fit Intercept", 1: "Original, Block 1",
                      2: "Replication, Fit Intercept", 3: "Replication, Block 1"}
    # Remove sections with no data
    used_sections = {g[3] for g in groups_info}
    section_labels = {k: v for k, v in section_labels.items() if k in used_sections}

    fig, ax = plt.subplots(figsize=(10, 6))
    _render_strip_plot(ax, groups_info,
                       section_labels=section_labels,
                       x_label="Accuracy",
                       first_gap=0.7)
    ax.xaxis.set_major_formatter(_accuracy_formatter())

    fig.tight_layout()
    _save(fig, "strip_plot_initial_accuracy.png")
    return fig


def visualize_leave_one_out(
    provided_slopes: pd.DataFrame,
) -> plt.Figure:
    """
    Leave-one-out sensitivity plot: for each participant, remove them and
    recompute Welch's two-tailed p-value for T-match vs P-match.
    Uses the original provided learning rates (groupLR_forLMM.csv).
    """
    t_lr = provided_slopes[provided_slopes["cond"] == "T"]["b"].values
    p_df = provided_slopes[provided_slopes["cond"] == "P"][["participant_id", "b"]].copy().reset_index(drop=True)

    reported_p = 0.045

    results = []  # (participant_id, group, lr, p_without)

    # Leave out each T-match participant
    for i in range(len(t_lr)):
        t_reduced = np.delete(t_lr, i)
        p_val = stats.ttest_ind(t_reduced, p_df["b"].values, equal_var=False).pvalue
        results.append((f"T_{i+1:02d}", "T-match", t_lr[i], p_val))

    # Leave out each P-match participant
    for i, row in p_df.iterrows():
        p_reduced = p_df.drop(i)["b"].values
        p_val = stats.ttest_ind(t_lr, p_reduced, equal_var=False).pvalue
        results.append((row["participant_id"], "P-match", row["b"], p_val))

    import seaborn as sns

    df = pd.DataFrame(results, columns=["participant_id", "group", "lr", "p_value"])
    df["group"] = df["group"] + " removed"

    fig, ax = plt.subplots(figsize=(8, 2.4))

    palette = {"T-match removed": "#1f5f8a", "P-match removed": "#2d8a2d"}
    sns.swarmplot(data=df, x="p_value", y="group", palette=palette, size=4.5,
                  edgecolor="white", linewidth=0.5, alpha=0.7, ax=ax, orient="h")
    ax.set_ylim(1.25, -0.25)

    # Significance threshold
    ax.axvline(x=0.05, color="red", linestyle="-", linewidth=1, alpha=0.7, label="p = 0.05")

    # Reported p-value from the original analysis.
    ax.axvline(x=reported_p, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label="Reported p = 0.045")

    ax.set_xlabel("Welch's p-value (two-tailed)", fontsize=10, labelpad=8)
    ax.set_ylabel("")
    n_flip = sum(1 for r in results if r[3] >= 0.05)
    ax.set_title(
        "Leave-one-out sensitivity: T-match vs P-match\n"
        f"{n_flip}/{len(results)} single removals make p > 0.05",
        fontsize=12, fontweight="bold", pad=16
    )

    ax.legend(fontsize=8, loc="center right", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", length=3, width=0.5)

    fig.tight_layout()
    _save(fig, "leave_one_out.png")
    return fig


def _spaghetti_grid(
    grid: list[list[tuple[str, pd.DataFrame, pd.DataFrame, str] | None]],
    filename: str,
) -> plt.Figure:
    """
    Render a spaghetti plot from a 2D grid of panels.

    Args:
        grid: rows x cols of (title, block_acc, slopes, color) or None for empty cells.
        filename: Output filename.
    """
    n_rows = len(grid)
    n_cols = max(len(row) for row in grid)

    panel_size = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )

    # Global y range across all panels
    all_acc_parts = []
    for row in grid:
        for cell in row:
            if cell is not None:
                all_acc_parts.append(cell[1]["accuracy"].values)
    all_acc = np.concatenate(all_acc_parts)
    y_min, y_max = all_acc.min(), all_acc.max()
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    x_fit = np.linspace(1, 8, 100)

    for r, row in enumerate(grid):
        for c in range(n_cols):
            ax = axes[r][c]
            cell = row[c] if c < len(row) else None

            if cell is None:
                ax.set_visible(False)
                continue

            title, ba, sl, color = cell
            pids = ba["participant_id"].unique()

            for pid in pids:
                p_data = ba[ba["participant_id"] == pid].sort_values("block")
                p_slope = sl[sl["participant_id"] == pid]
                blocks = p_data["block"].values
                acc = p_data["accuracy"].values

                ax.scatter(blocks, acc, s=12, color=color, alpha=0.25, zorder=3, linewidths=0)

                if not p_slope.empty:
                    a_val = p_slope["a"].iloc[0]
                    b_val = p_slope["b"].iloc[0]
                    y_fit = log_linear(x_fit, a_val, b_val)
                    ax.plot(x_fit, y_fit, color=color, alpha=0.2, linewidth=0.8, zorder=2)

            ax.text(
                1,
                1,
                f"{title} (n={len(pids)})",
                transform=ax.get_xaxis_transform(),
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="top",
                color="#000000",
            )
            ax.set_xlim(0.5, 8.5)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(range(1, 9))
            ax.set_box_aspect(1)
            if r == 0 and c == 0:
                ax.set_xlabel("Block", fontsize=9)
                ax.set_ylabel("Accuracy", fontsize=9, labelpad=8)
            elif r == n_rows - 1 and c == 0:
                pass
            else:
                ax.set_xticklabels([])
                if c != 0:
                    ax.set_yticklabels([])
            if c == 0:
                ax.yaxis.set_major_formatter(_accuracy_formatter())

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)
            ax.tick_params(axis="both", length=3, width=0.5)

    fig.tight_layout(rect=(0.02, 0, 1, 1))

    _save(fig, filename)
    return fig


_SPAGHETTI_COND_STYLE = {
    "T":  ("T-match",            "#1f5f8a"),
    "P":  ("P-match",            "#2d8a2d"),
    "TN": ("T-nonMatch",         "#9370DB"),
    "C":  ("Arrhythmic Control", "#666666"),
}


def _spaghetti_panel(
    block_acc: pd.DataFrame,
    slopes: pd.DataFrame,
    cond: str,
    section: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Build one (title, block_acc, slopes, color) panel for a given condition."""
    label, color = _SPAGHETTI_COND_STYLE[cond]
    return (
        f"{section} {label}",
        block_acc[block_acc["cond"] == cond],
        slopes[slopes["cond"] == cond],
        color,
    )


def visualize_spaghetti_all(
    original_data: tuple[pd.DataFrame, pd.DataFrame],
    tn_data: tuple[pd.DataFrame, pd.DataFrame],
    control_data: tuple[pd.DataFrame, pd.DataFrame],
    replication_day1_data: tuple[pd.DataFrame, pd.DataFrame],
) -> plt.Figure:
    """Spaghetti grid: original P/T/TN/C on top, replication P/T on bottom."""
    orig_ba, orig_sl = original_data
    tn_ba, tn_sl = tn_data
    ctrl_ba, ctrl_sl = control_data
    rep_ba, rep_sl = replication_day1_data

    return _spaghetti_grid([
        [
            _spaghetti_panel(orig_ba, orig_sl, "T", "Original"),
            _spaghetti_panel(orig_ba, orig_sl, "P", "Original"),
            _spaghetti_panel(tn_ba, tn_sl, "TN", "Original"),
            _spaghetti_panel(ctrl_ba, ctrl_sl, "C", "Original"),
        ],
        [
            _spaghetti_panel(rep_ba, rep_sl, "T", "Replication"),
            _spaghetti_panel(rep_ba, rep_sl, "P", "Replication"),
            None,
            None,
        ],
    ], "spaghetti_all.png")


def visualize_spaghetti_PT_comparison(
    original_data: tuple[pd.DataFrame, pd.DataFrame],
    replication_day1_data: tuple[pd.DataFrame, pd.DataFrame],
) -> plt.Figure:
    """Spaghetti grid: P-match and T-match side-by-side, original vs replication."""
    orig_ba, orig_sl = original_data
    rep_ba, rep_sl = replication_day1_data

    return _spaghetti_grid([
        [
            _spaghetti_panel(orig_ba, orig_sl, "T", "Original"),
            _spaghetti_panel(rep_ba, rep_sl, "T", "Replication"),
        ],
        [
            _spaghetti_panel(orig_ba, orig_sl, "P", "Original"),
            _spaghetti_panel(rep_ba, rep_sl, "P", "Replication"),
        ],
    ], "spaghetti_PT_comparison.png")


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

    # Get the two conditions from the data, using the same first-comparison
    # order as the manuscript figures.
    condition_order = {"T": 0, "P": 1, "TN": 2, "C": 3}
    conditions = sorted(slopes["cond"].unique(), key=lambda c: (condition_order.get(c, 99), c))
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

    panel_width = 2.25
    panel_height = 2.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

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

        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(1, 9))
        ax.set_box_aspect(1)
        ax.set_anchor("W")
        if fit_annotation:
            ax.text(0, -0.05, fit_annotation, transform=ax.transAxes,
                    fontsize=8, ha="left", va="top", color="#000000", clip_on=False)

        if is_last_row:
            ax.set_xlabel("Block", fontsize=8, color="#000000", labelpad=2)
        else:
            ax.set_xticklabels([])

        if not show_ylabel:
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
            plot_one(axes[row_idx, cols_per_group + col], pid, is_last, show_ylabel=False)

    # Drop x-tick labels and the "Block" xlabel everywhere — inline caption grounds each panel
    for ax in axes.flat:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    # Y-tick labels on the leftmost column of every row.
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if col_idx != 0:
                axes[row_idx, col_idx].set_yticklabels([])

    # Y-axis label only on the top-left chart
    for ax in axes.flat:
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("Accuracy", fontsize=8, color="#000000")
    # Percent formatter on every axis that still shows y-tick labels.
    for row_idx in range(n_rows):
        axes[row_idx, 0].yaxis.set_major_formatter(_accuracy_formatter())

    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.06, top=0.972,
                        wspace=0.00, hspace=0.16)

    # Add extra space between condition 1 and condition 2 columns
    for row_idx in range(n_rows):
        for col_idx in range(cols_per_group, n_cols):
            ax = axes[row_idx, col_idx]
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.025, pos.y0, pos.width, pos.height])

    # Column headers (centered over each group's columns)
    fig.text(0.25, 0.992, cond_labels[cond1], ha="center", va="top",
             fontsize=12, fontweight="bold", color="#000000")
    fig.text(0.75, 0.992, cond_labels[cond2], ha="center", va="top",
             fontsize=12, fontweight="bold", color="#000000")

    _save(fig, f"original learning rates {cond_labels[cond1]} vs {cond_labels[cond2]}.png")
    return fig


def visualize_original_vs_replication_aggregate(
    original_data: tuple[pd.DataFrame, pd.DataFrame],
    replication_data: tuple[pd.DataFrame, pd.DataFrame],
    min_lr_1st_day_original: float,
) -> plt.Figure:
    """
    Visualize aggregate learning curves: original paper data alongside replication.

    Three panels: (1) original all participants, (2) original filtered by
    min_lr_1st_day_original, (3) replication day 1.

    Args:
        original_data: tuple of (block_acc, slopes) from the original study
        replication_data: tuple of (block_acc, slopes) from replication study (day 1 only)
        min_lr_1st_day_original: exclude original-study participants with day-1 LR below this threshold
            for the middle "filtered" panel
    """
    block_acc, slopes = original_data

    colors = {
        "P": "#2d8a2d",  # Green for P-match
        "T": "#1f5f8a",  # Blue for T-match
    }
    labels = {
        "P": "P-match",
        "T": "T-match",
    }

    def participants_per_group_label(slopes_data: pd.DataFrame) -> str:
        counts = slopes_data.groupby("cond")["participant_id"].nunique()
        counts = counts.reindex(["T", "P"]).dropna().astype(int)
        if counts.empty:
            return "n=0 per group"
        if counts.nunique() == 1:
            return f"n={counts.iloc[0]} per group"
        return ", ".join(f"{labels[cond]} n={count}" for cond, count in counts.items())

    def plot_aggregate(
        ax,
        block_acc_data,
        slopes_data,
        title,
        y_lim=None,
        show_y_axis=True,
        show_x_axis=True,
    ):
        """Helper to plot one aggregate view"""
        endpoints = {}

        for cond in ["P", "T"]:
            cond_acc = block_acc_data[block_acc_data["cond"] == cond]
            grp_means = cond_acc.groupby("block")["accuracy"].mean().reset_index()
            blocks = grp_means["block"].values
            acc = grp_means["accuracy"].values

            color = colors[cond]

            # Plot dots
            ax.scatter(blocks, acc, c=color, s=18, zorder=3)

            # Fit and plot curve
            fit = fit_learning_rate(blocks, acc, method="ols")
            x_fit = np.linspace(1, 8, 100)
            y_fit = log_linear(x_fit, fit.a, fit.b)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, zorder=2)

            endpoints[cond] = (x_fit[-1], y_fit[-1], fit.b)

        # Direct labeling: T-match on top, P-match on bottom. Right edge aligns
        # with the rightmost edge of the final dot at block 8.
        from matplotlib.transforms import offset_copy
        dot_radius_pts = float(np.sqrt(18 / np.pi))
        label_trans = offset_copy(ax.transData, fig=fig, x=dot_radius_pts, units="points")
        for cond, (x, y, lr) in endpoints.items():
            color = colors[cond]
            if cond == "T":
                y_offset = 2.0
                va = "bottom"
            else:  # P-match
                y_offset = -2.0
                va = "top"
            ax.text(x, y + y_offset, f"{labels[cond]}  LR={_fmt_lr(lr)}",
                    color=color, fontsize=9, va=va, ha="right",
                    transform=label_trans)

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
        if show_x_axis:
            ax.set_xlabel("Block", fontsize=10, color="black")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if show_y_axis:
            ax.set_ylabel("Accuracy", fontsize=10, color="black")
            ax.yaxis.set_major_formatter(_accuracy_formatter())
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(axis="y", colors="black", length=3, width=0.5)

        # Set y-axis limits if provided
        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.set_title(title, fontsize=11)

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 5.5))

    # Compute shared y-axis range from all data
    all_means = block_acc.groupby(["cond", "block"])["accuracy"].mean()
    y_min = all_means.min()
    y_max = all_means.max()

    rep_block_acc, rep_slopes = replication_data
    rep_day1 = rep_block_acc[rep_block_acc["day_index"] == 1]
    rep_means = rep_day1.groupby(["cond", "block"])["accuracy"].mean()
    y_min = min(y_min, rep_means.min())
    y_max = max(y_max, rep_means.max())

    y_padding = (y_max - y_min) * 0.1
    y_lim = (y_min - y_padding, y_max + y_padding)

    # Left: All data from original paper
    plot_aggregate(axes[0], block_acc, slopes,
                   f"Original, Day 1, {participants_per_group_label(slopes)}",
                   y_lim=y_lim, show_y_axis=True, show_x_axis=True)

    # Middle: Filtered data from original paper
    valid_pids = slopes[slopes["b"] >= min_lr_1st_day_original]["participant_id"].unique()
    block_acc_filtered = block_acc[block_acc["participant_id"].isin(valid_pids)]
    slopes_filtered = slopes[slopes["participant_id"].isin(valid_pids)]
    n_excluded = len(slopes["participant_id"].unique()) - len(valid_pids)
    plot_aggregate(axes[1], block_acc_filtered, slopes_filtered,
                  f"Original, Day 1, filtered (LR ≥ {min_lr_1st_day_original}, n={n_excluded} excluded)",
                  y_lim=y_lim, show_y_axis=False, show_x_axis=False)

    # Right: Replication day 1
    rep_day1_acc = rep_block_acc[rep_block_acc["day_index"] == 1].copy()
    rep_day1_slopes = rep_slopes[rep_slopes["day_index"] == 1].copy()
    plot_aggregate(axes[2], rep_day1_acc, rep_day1_slopes,
                   f"Replication, Day 1, {participants_per_group_label(rep_day1_slopes)}", y_lim=y_lim,
                   show_y_axis=False, show_x_axis=False)

    fig.suptitle("Learning Rate Comparison: Original vs Replication",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 1))

    _save(fig, "original_vs_replication_aggregate.png")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze learning rate data from Glass pattern experiment"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database. Mutually exclusive with --from-export."
    )
    parser.add_argument(
        "--from-export",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to replication accuracy CSV (produced by export_replication_data.py). "
            "Slopes are recomputed using --fit-method. Mutually exclusive with --db."
        ),
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
        "--use-internal-ids",
        action="store_true",
        help=(
            "Use internal DB participant IDs in charts. Only valid with --db; "
            "exported data already uses public IDs."
        ),
    )
    parser.add_argument(
        "--original-data-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory with original paper data (Michael et al., 2023): AccPerLat.csv, groupLR_forLMM.csv, groupAccs2_lmm_b1.csv"
    )
    parser.add_argument(
        "--charts-save-dir",
        type=str,
        default="_generated_charts",
        metavar="DIR",
        help="Directory to save generated charts (default: _generated_charts)"
    )

    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.charts_save_dir

    # Mutually exclusive data sources
    if args.db is not None and args.from_export is not None:
        parser.error("--db and --from-export are mutually exclusive")
    if args.db is None and args.from_export is None:
        parser.error("Must provide either --db or --from-export")
    if args.from_export is not None and args.use_internal_ids:
        parser.error("--use-internal-ids cannot be used with --from-export")

    # Parse participant filters
    include_only = parse_participant_list(args.include_only_participants)
    exclude = parse_participant_list(args.exclude_participants)

    if include_only is not None and exclude is not None:
        parser.error("Cannot use both --include-only-participants and --exclude-participants")

    block_acc, slopes = load_replication_data(
        db_path=args.db,
        from_export_path=args.from_export,
        include_only=include_only,
        exclude=exclude,
        fit_method=args.fit_method,
        use_internal_ids=args.use_internal_ids,
    )

    print(f"\nLoading original paper data from {args.original_data_dir}...")

    # Load P-match vs T-match
    orig_block_acc, orig_slopes = refit_approximate_original_paper_curves(args.original_data_dir)
    print(f"Loaded {len(orig_slopes)} participants (P-match and T-match)")

    # Print fitted slopes
    print("\n=== Original Paper Fitted slopes (P-match vs T-match) ===")
    print(orig_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

    # Load T-match vs Control (needed for aggregate chart)
    print("\nLoading T-match vs Control comparison...")
    tc_block_acc, tc_slopes = refit_approximate_original_paper_curves(args.original_data_dir, groups={2: "T", 4: "C"})
    print(f"Loaded {len(tc_slopes)} participants (T-match and Control)")

    # Extract control data for aggregate chart
    control_block_acc = tc_block_acc[tc_block_acc["cond"] == "C"]
    control_slopes = tc_slopes[tc_slopes["cond"] == "C"]

    # Load T-nonMatch (needed for specific aggregate chart)
    print("\nLoading T-nonMatch...")
    tn_block_acc, tn_slopes = refit_approximate_original_paper_curves(args.original_data_dir, groups={3: "TN"})
    print(f"Loaded {len(tn_slopes)} participants (T-nonMatch)")

    # Visualize P-match vs T-match individual curves
    visualize_original_individual_curves(orig_block_acc, orig_slopes, max_per_group=20, cols_per_group=3)

    # Original (all + filtered) alongside replication, in 3 panels
    visualize_original_vs_replication_aggregate(
        original_data=(orig_block_acc, orig_slopes),
        replication_data=(block_acc, slopes),
        min_lr_1st_day_original=-1,
    )

    # Print fitted slopes
    print("\n=== Original Paper Fitted slopes (T-match vs Control) ===")
    print(tc_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

    # Visualize T-match vs Control
    visualize_original_individual_curves(tc_block_acc, tc_slopes, max_per_group=20, cols_per_group=3,
                                        cond_labels={"T": "T-match", "C": "Arrhythmic Control"})

    # Load P-match vs Control
    print("\nLoading P-match vs Control comparison...")
    pc_block_acc, pc_slopes = refit_approximate_original_paper_curves(args.original_data_dir, groups={1: "P", 4: "C"})
    print(f"Loaded {len(pc_slopes)} participants (P-match and Control)")

    # Print fitted slopes
    print("\n=== Original Paper Fitted slopes (P-match vs Control) ===")
    print(pc_slopes.sort_values(["cond", "participant_id"]).to_string(index=False))

    # Visualize P-match vs Control
    visualize_original_individual_curves(pc_block_acc, pc_slopes, max_per_group=20, cols_per_group=3,
                                        cond_labels={"P": "P-match", "C": "Arrhythmic Control"})

    # Load T-nonMatch data for strip plot
    print("\nLoading T-nonMatch data...")
    tn_block_acc, tn_slopes = refit_approximate_original_paper_curves(args.original_data_dir, groups={3: "TN"})
    print(f"Loaded {len(tn_slopes)} participants (T-nonMatch)")

    # Prepare data for strip plot - use PROVIDED learning rates for original data
    print("\nLoading provided learning rates from groupLR_forLMM.csv...")
    provided_slopes = load_original_paper_provided_learning_rates(args.original_data_dir)

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
    visualize_lr_strip_plot(
        p_slopes_only, t_slopes_only, tn_slopes_only, c_slopes_only,
        rep_p_slopes_only, rep_t_slopes_only, compact=True
    )

    # Day 2 strip plot (original study, second session)
    print("\nGenerating Day 2 strip plot...")
    day2_slopes = load_original_paper_provided_learning_rates(".", filename="groupLR_postLMM.csv")
    visualize_day2_strip_plot(day2_slopes)

    # Initial accuracy strip plot
    print("\nGenerating initial accuracy strip plot...")
    b1_path = os.path.join(args.original_data_dir, "groupAccs2_lmm_b1.csv")
    rep_day1_block_acc_for_b1 = block_acc[block_acc["day_index"] == 1]
    rep_day1_slopes_for_b1 = slopes[slopes["day_index"] == 1]
    visualize_initial_accuracy_strip_plot(
        b1_path, rep_day1_block_acc_for_b1,
        orig_slopes=orig_slopes, rep_slopes=rep_day1_slopes_for_b1,
    )

    # Spaghetti plots: individual learning curves overlaid per group
    print("\nGenerating spaghetti plots...")
    rep_day1_data = (
        block_acc[block_acc["day_index"] == 1],
        slopes[slopes["day_index"] == 1],
    )
    visualize_spaghetti_all(
        original_data=(orig_block_acc, orig_slopes),
        tn_data=(tn_block_acc, tn_slopes),
        control_data=(control_block_acc, control_slopes),
        replication_day1_data=rep_day1_data,
    )
    visualize_spaghetti_PT_comparison(
        original_data=(orig_block_acc, orig_slopes),
        replication_day1_data=rep_day1_data,
    )

    # Leave-one-out sensitivity plot (original provided learning rates)
    print("\nGenerating leave-one-out sensitivity plot...")
    visualize_leave_one_out(provided_slopes)

    # Validation strip plot: provided vs recomputed learning rates
    print("\nGenerating validation strip plot (provided vs recomputed)...")
    visualize_validation_strip_plot(provided_slopes, orig_slopes, tn_slopes)

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

    # Run hypothesis tests
    run_h1_within_subject(slopes)
    run_h2_between_groups_day1(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=False)
    run_h3_between_groups_day1_intercept(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=True)
    run_h3_between_groups_day2_intercept(slopes, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=True)
    run_h3_between_groups_day1_initial_accuracy(block_acc, n_perm=args.n_permutations, seed=args.permutation_seed, two_sided=True)

    visualize_learning_curves(block_acc, slopes)
    visualize_replication_aggregate_both_days(block_acc, slopes)
    plt.show()


if __name__ == "__main__":
    main()
