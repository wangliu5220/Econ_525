"""Post-run validation, diagnostics, and output generation for LLM headline scores."""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")       # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CHECKPOINT_PATH = DATA_DIR / "llm_scores_checkpoint.csv"
INPUT_PATH = DATA_DIR / "llm_input_prompts.csv"
VALIDATION_PATH = DATA_DIR / "validation_set_scored.csv"
OUT_HEADLINE = DATA_DIR / "llm_scores_headline.csv"
OUT_DAILY = DATA_DIR / "llm_scores_daily.csv"
OUT_PLOT = DATA_DIR / "score_distributions.png"

SCORE_COLS = ["tangibility", "relevance", "llm_sentiment"]


def load_data():
    """Load checkpoint scores and input headlines. Exit if missing."""
    if not CHECKPOINT_PATH.exists():
        sys.exit(f"Checkpoint not found at {CHECKPOINT_PATH}. Run the scoring script first.")
    scores_df = pd.read_csv(CHECKPOINT_PATH)
    input_df = pd.read_csv(INPUT_PATH)
    input_df["row_id"] = input_df.index
    return scores_df, input_df


def count_check(scores_df: pd.DataFrame, input_df: pd.DataFrame):
    print("\nCoverage:")
    total = len(input_df)
    scored = len(scores_df)
    gap = total - scored
    print(f"  Expected rows: {total:,}")
    print(f"  Scored rows:   {scored:,}")
    print(f"  Missing:       {gap:,}")
    if gap > 0:
        scored_ids = set(scores_df["row_id"])
        all_ids = set(input_df["row_id"])
        missing_ids = sorted(all_ids - scored_ids)
        print(f"  Missing row_ids (first 20): {missing_ids[:20]}")
    print()


def null_check(scores_df: pd.DataFrame):
    print("\nNull/NaN rates:")
    total = len(scores_df)
    for col in SCORE_COLS:
        n_null = scores_df[col].isna().sum()
        pct = 100 * n_null / total if total > 0 else 0
        flag = " *** WARNING ***" if pct > 2 else ""
        print(f"  {col:20s}: {n_null:,} nulls ({pct:.2f}%){flag}")
    print()


def distribution_check(scores_df: pd.DataFrame):
    print("\nScore distributions:")
    for col in SCORE_COLS:
        s = scores_df[col].dropna()
        print(f"\n  {col}:")
        print(f"    mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
        print(f"    25%={s.quantile(0.25):.3f}  50%={s.quantile(0.50):.3f}  "
              f"75%={s.quantile(0.75):.3f}")

        # Clustering warning
        near_zero = ((s >= -0.1) & (s <= 0.1)).sum()
        pct_zero = 100 * near_zero / len(s) if len(s) > 0 else 0
        if pct_zero > 50:
            print(f"    *** WARNING: {pct_zero:.1f}% of scores in [-0.1, 0.1] — "
                  f"possible clustering around zero ***")

        # Extremes usage
        in_tails = ((s <= -0.5) | (s >= 0.5)).sum()
        pct_tails = 100 * in_tails / len(s) if len(s) > 0 else 0
        if pct_tails < 10:
            print(f"    *** WARNING: Only {pct_tails:.1f}% of scores in tails "
                  f"(|score| >= 0.5) — underuse of range ***")

    # Save histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, SCORE_COLS):
        scores_df[col].dropna().hist(bins=50, ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("Score")
        ax.set_xlim(-1.05, 1.05)
    fig.suptitle("LLM Score Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\n  Histograms saved to {OUT_PLOT.name}")
    print()


def validation_check(scores_df: pd.DataFrame):
    print("\nValidation set comparison:")
    if not VALIDATION_PATH.exists():
        print("  Validation set not found. Skipping.\n")
        return

    val_df = pd.read_csv(VALIDATION_PATH)
    manual_cols = ["manual_tangibility", "manual_relevance", "manual_sentiment"]

    # Check if manual labels exist
    if not all(c in val_df.columns for c in manual_cols):
        print("  Manual label columns not found in validation_set.csv. Skipping.\n")
        return

    # Check if any manual labels are filled in
    if val_df[manual_cols].isna().all().all():
        print("  Manual labels are all empty. Fill them in first. Skipping.\n")
        return

    merged = val_df.merge(
        scores_df, on="row_id", how="inner", suffixes=("_manual", "_llm")
    )
    if merged.empty:
        print("  No overlapping row_ids between validation set and scores. Skipping.\n")
        return

    print(f"  Matched {len(merged)} headlines\n")

    pairs = [
        ("manual_tangibility", "tangibility"),
        ("manual_relevance", "relevance"),
        ("manual_sentiment", "llm_sentiment"),
    ]
    for manual_col, llm_col in pairs:
        mask = merged[manual_col].notna() & merged[llm_col].notna()
        if mask.sum() < 3:
            print(f"  {llm_col}: Not enough paired observations")
            continue
        m = merged.loc[mask, manual_col]
        l = merged.loc[mask, llm_col]
        corr = m.corr(l)
        mae = (m - l).abs().mean()
        print(f"  {llm_col:20s}:  Pearson r = {corr:.3f}  |  MAE = {mae:.3f}")

    # Flag large divergences
    print("\n  Large divergences (|manual - llm| > 0.5):")
    for manual_col, llm_col in pairs:
        mask = merged[manual_col].notna() & merged[llm_col].notna()
        sub = merged[mask].copy()
        sub["_diff"] = (sub[manual_col] - sub[llm_col]).abs()
        divergent = sub[sub["_diff"] > 0.5]
        if not divergent.empty:
            for _, row in divergent.iterrows():
                print(f"    row_id={int(row['row_id']):5d}  {llm_col:20s}: "
                      f"manual={row[manual_col]:+.2f}  llm={row[llm_col]:+.2f}  "
                      f"TICKER={row['TICKER']}  {row['headline'][:60]}")
    print()


def extremes_check(scores_df: pd.DataFrame, input_df: pd.DataFrame):
    print("\nSpot-check extremes (top/bottom 10):")
    merged = input_df.merge(scores_df, on="row_id", how="inner")

    for col in SCORE_COLS:
        print(f"\n  --- {col} ---")
        top10 = merged.nlargest(10, col)
        bot10 = merged.nsmallest(10, col)

        print(f"  TOP 10 (highest {col}):")
        for _, r in top10.iterrows():
            print(f"    [{r[col]:+.2f}] {r['TICKER']:6s} {r['headline'][:70]}")

        print(f"  BOTTOM 10 (lowest {col}):")
        for _, r in bot10.iterrows():
            print(f"    [{r[col]:+.2f}] {r['TICKER']:6s} {r['headline'][:70]}")
    print()


def generate_outputs(scores_df: pd.DataFrame, input_df: pd.DataFrame):
    print("\nWriting output files:")

    # Headline-level scores
    headline_df = input_df.merge(scores_df, on="row_id", how="left")
    headline_df.to_csv(OUT_HEADLINE, index=False)
    print(f"  {OUT_HEADLINE.name}: {len(headline_df):,} rows")

    # Daily aggregates
    daily_df = (
        headline_df
        .dropna(subset=SCORE_COLS, how="all")
        .groupby(["Effective_Trading_Date", "TICKER"], as_index=False)
        .agg(
            mean_tangibility=("tangibility", "mean"),
            mean_relevance=("relevance", "mean"),
            mean_llm_sentiment=("llm_sentiment", "mean"),
            headline_count=("row_id", "size"),
        )
    )
    daily_df.to_csv(OUT_DAILY, index=False)
    print(f"  {OUT_DAILY.name}: {len(daily_df):,} rows")
    print()


def main():
    scores_df, input_df = load_data()

    count_check(scores_df, input_df)
    null_check(scores_df)
    distribution_check(scores_df)
    validation_check(scores_df)
    extremes_check(scores_df, input_df)
    generate_outputs(scores_df, input_df)

    print("Validation complete.")


if __name__ == "__main__":
    main()
