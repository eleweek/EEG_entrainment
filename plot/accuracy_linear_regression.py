#!/usr/bin/env python3
# plot_block_logistic.py
import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def parse_exclusions(excl_str: str) -> set[tuple[str, int]]:
    """
    Parse exclusions like:
      "ses-123:1, ses-123:3-5, ses-789:2"
    Returns a set of (session_id, block_int).
    """
    out: set[tuple[str, int]] = set()
    if not excl_str:
        return out
    tokens = [t.strip() for t in excl_str.split(",") if t.strip()]
    for tok in tokens:
        m = re.match(r"^(?P<sid>[^:]+):(?P<blk>[\d\-]+)$", tok)
        if not m:
            raise ValueError(f"Bad exclusion token: {tok!r}. Use 'session_id:block' or 'session_id:b1-b2'")
        sid = m.group("sid")
        blk = m.group("blk")
        if "-" in blk:
            a, b = blk.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = min(a, b), max(a, b)
            for k in range(lo, hi + 1):
                out.add((sid, k))
        else:
            out.add((sid, int(blk)))
    return out


def load_block_df(db_path: Path, tperblock: int, drop_incomplete: bool = True) -> pd.DataFrame:
    """
    Recover block summaries from *trial-level* rows by ordering in time and
    chunking every `tperblock` trials per session into a block.

    Returns columns:
      session_id, start_ts, block, cond, n, hits

    - `cond` is taken from the trials inside the chunk. If (unexpectedly) a chunk
      mixes conditions, you'll get two rows for that (session_id, block), one per cond.
      In a clean dataset (one condition per block), there will be exactly one row.
    - Set drop_incomplete=False to keep the last partial block of a session.
    """
    with sqlite3.connect(str(db_path)) as cn:
        q = """
        SELECT
            t.session_id,
            COALESCE(s.start_ts, 0)          AS start_ts,
            t.cond,
            COALESCE(t.correct,0)             AS correct,
            COALESCE(t.timed_out,0)           AS timed_out,
            t.ts_onset                        AS ts_onset,
            t.trial_index                     AS trial_index,
            t.rowid                           AS rid
        FROM trial t
        LEFT JOIN session s ON s.id = t.session_id
        ORDER BY start_ts,
                 t.session_id,
                 COALESCE(t.ts_onset, 0),
                 COALESCE(t.trial_index, 2147483647),
                 t.rowid;
        """
        df = pd.read_sql(q, cn)

    if df.empty:
        return pd.DataFrame(columns=["session_id","start_ts","block","cond","n","hits"])

    # sequential index within each session
    df["seq"] = df.groupby("session_id").cumcount()

    # derive block number (1-based) purely from sequence
    df["block"] = (df["seq"] // int(tperblock)) + 1

    # hits = correct AND not timed out
    df["hit"] = (df["correct"].astype(int) & (1 - df["timed_out"].astype(int))).astype(int)

    # aggregate to block level (note: if a chunk mixes conds, you'll see one row per cond)
    out = (
        df.groupby(["session_id","start_ts","block","cond"], as_index=False)
          .agg(n=("hit","size"), hits=("hit","sum"))
          .sort_values(["start_ts","session_id","block","cond"])
          .reset_index(drop=True)
    )

    # sanity check & optional drop of incomplete blocks
    if drop_incomplete:
        before = len(out)
        out = out[out["n"] == int(tperblock)].copy()
        dropped = before - len(out)
        if dropped:
            print(f"⚠️  Dropped {dropped} incomplete block rows (n != {tperblock}). "
                  f"Use drop_incomplete=False to keep them.")

    # warn if any (session, block) has >1 cond (shouldn’t happen if your task uses one cond per block)
    mixed = (
        out.groupby(["session_id","block"])["cond"]
           .nunique()
           .reset_index(name="n_cond")
    )
    bad = mixed[mixed["n_cond"] > 1]
    if not bad.empty:
        print("⚠️  Detected blocks that mix conditions (check your logs):")
        print(bad.to_string(index=False))

    # tidy types
    out["block"] = out["block"].astype(int)
    out["n"]     = out["n"].astype(int)
    out["hits"]  = out["hits"].astype(int)
    return out




def add_exposure_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds k = within-condition exposure index across the entire timeline
    (order by start_ts, session_id, block), separately for P and T.
    """
    df = df.sort_values(["start_ts", "session_id", "block"]).reset_index(drop=True)
    df["k"] = df.groupby("cond").cumcount() + 1
    return df


def fit_wls_linear(df_cond: pd.DataFrame):
    """
    Fit Weighted Least Squares:
        acc ~ const + k
    with weights = n (trials per block).
    Returns (result, design_matrix_columns)
    """
    if df_cond.empty:
        return None, None
    y = df_cond["hits"] / df_cond["n"]
    X = sm.add_constant(df_cond["k"].astype(float))
    w = df_cond["n"].astype(float)  # simple binomial-inspired weighting
    model = sm.WLS(y, X, weights=w)
    res = model.fit(cov_type="HC1")  # robust SEs
    return res, X.columns.tolist()


def predict_wls(res, k_min: int, k_max: int, num: int = 200, clip: bool = False):
    """
    Predict mean and 95% CI over a k-grid.
    """
    if res is None:
        return pd.DataFrame(columns=["k","mean","lo","hi"])
    k_grid = np.linspace(k_min, k_max, num)
    Xp = sm.add_constant(k_grid)
    pred = res.get_prediction(Xp).summary_frame(alpha=0.05)  # mean, mean_ci_lower/upper
    out = pd.DataFrame({
        "k": k_grid,
        "mean": pred["mean"].to_numpy(),
        "lo": pred["mean_ci_lower"].to_numpy(),
        "hi": pred["mean_ci_upper"].to_numpy(),
    })
    if clip:
        out[["mean","lo","hi"]] = out[["mean","lo","hi"]].clip(0.0, 1.0)
    return out




def main():
    ap = argparse.ArgumentParser(description="Plot block accuracies and linear fits (WLS) for P vs T.")
    ap.add_argument("--db", type=Path, default=Path("study.db"), help="Path to SQLite DB")
    ap.add_argument("--tperblock", type=int, default=60, help="Trials per block if trial.block is absent")
    ap.add_argument("--exclude", type=str, default="", help="Exclusions like 'ses-123:1, ses-123:3-5'")
    ap.add_argument("--title", type=str, default="Block accuracy & linear fits (P vs T)")
    ap.add_argument("--save", type=Path, default=None, help="If set, save figure to this path")
    ap.add_argument("--clip", action="store_true", help="Clip fitted curves to [0,1] in the plot")
    args = ap.parse_args()

    # Load
    df = load_block_df(args.db, args.tperblock)

    # Exclude
    excl = parse_exclusions(args.exclude)
    if excl:
        before = len(df)
        df = df[~df.apply(lambda r: (r["session_id"], int(r["block"])) in excl, axis=1)].copy()
        print(f"Excluded {before - len(df)} block rows via --exclude")

    # Keep only P/T
    df = df[df["cond"].isin(["P","T"])].copy()
    if df.empty:
        raise SystemExit("No P/T blocks found after exclusions.")

    # Exposure index and accuracy
    df = add_exposure_index(df)
    df["acc"] = df["hits"] / df["n"]

    # Fit per condition
    fits = {}
    print(df)
    for cond in ["P","T"]:
        sub = df[df["cond"] == cond].copy()
        if sub.empty:
            print(f"[WARN] No rows for cond={cond}; skipping.")
            fits[cond] = (None, pd.DataFrame())
            continue
        res, _ = fit_wls_linear(sub)
        kmin, kmax = int(sub["k"].min()), int(sub["k"].max())
        curve = predict_wls(res, kmin, kmax, num=400, clip=args.clip)
        fits[cond] = (res, curve)
        b = res.params.get("k", np.nan)
        ci = res.conf_int().loc["k"].to_numpy() if "k" in res.params.index else [np.nan, np.nan]
        print(f"{cond} slope (pp/block approx): {b*100:.2f}  "
              f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}]")

    # Plot
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10,6))
    palette = {"P":"#1f77b4", "T":"#d62728"}
    jitter   = {"P":-0.05, "T":+0.05}

    for cond in ["P","T"]:
        sub = df[df["cond"] == cond]
        if sub.empty: continue
        x = sub["k"].astype(float) + jitter[cond]
        ax.scatter(x, sub["acc"], s=50, alpha=0.85, color=palette[cond],
                   label=f"{cond} blocks", edgecolors="none")

    for cond in ["P","T"]:
        res, curve = fits[cond]
        if res is None or curve.empty: continue
        ax.plot(curve["k"], curve["mean"], lw=3, color=palette[cond], label=f"{cond} linear fit")
        ax.fill_between(curve["k"], curve["lo"], curve["hi"], color=palette[cond], alpha=0.15, linewidth=0)

    ax.set_xlabel("Within-condition exposure index (k)")
    ax.set_ylabel("Block accuracy (proportion correct)")
    ax.set_xlim(left=0.5)
    y_min = float(df["acc"].min())
    y_max = float(df["acc"].max())
    pad = 0.02
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved → {args.save}")
    else:
        plt.show()



if __name__ == "__main__":
    main()

