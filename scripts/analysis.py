"""
Breast Cancer Wisconsin Diagnostic Dataset — Exploratory Analysis
=================================================================
Programme : AI & ML and Computational Biology | Bversity 2026
Author    : [SYED HUMAID ALI AKHTAR]
Date      : 2025-06-04

Dataset   : Wolberg, W.H., Street, W.N. and Mangasarian, O.L. (1995)
            Breast Cancer Wisconsin Diagnostic Dataset.
            UCI Machine Learning Repository.
            https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
            Accessed: 4 June 2025
 
This script loads the raw WDBC data, inspects and cleans it, ranks the
five most discriminatory features between malignant and benign tumours,
and produces two summary figures.
 
BUGFIX (this version)
----------------------
The previous version ranked "id" (the row identifier) as the #1
"discriminatory feature" and plotted it in Figure 2. This happened
because the id column was left inside `feature_cols` when computing
group means. The id number is just a record identifier — it carries no
biological meaning, and any apparent difference between the malignant
and benign group means is a coincidence of how the records were
ordered/sampled, not a real diagnostic signal. This version explicitly
excludes non-biological columns ("id") from the feature-ranking step.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
 
# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
SUFFIXES = ["_mean", "_se", "_worst"]
COLUMNS = (
    ["id", "diagnosis"] +
    [f + s for s in SUFFIXES for f in FEATURE_NAMES]
)
 
# Columns that must never be treated as biological/analytical features.
NON_FEATURE_COLUMNS = ["id", "diagnosis"]
 
COL_M = "#C0392B"  # colour for malignant
COL_B = "#2980B9"  # colour for benign
 
 
def get_project_paths(script_path: Path) -> dict:
    """
    Resolve the project's directory layout relative to this script's location.
 
    Assumes the following structure, with this script/notebook living
    directly inside the project root:
 
        <root>/                     <- this script/notebook lives here
            data/raw/wdbc.data
            data/processed/
            results/figures/
 
    Parameters
    ----------
    script_path : Path
        Path to this script (typically ``Path(__file__)``).
 
    Returns
    -------
    dict
        Dictionary with keys ``root``, ``raw_dir``, ``proc_dir``, ``fig_dir``.
        The processed-data and figures directories are created if missing.
    """
    root = script_path.resolve().parent
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"
    fig_dir = root / "results" / "figures"
 
    proc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
 
    return {"root": root, "raw_dir": raw_dir, "proc_dir": proc_dir, "fig_dir": fig_dir}
 
 
def load_data(raw_file: Path) -> pd.DataFrame:
    """
    Load the raw WDBC data file into a labelled DataFrame.
 
    The raw ``wdbc.data`` file has no header row, so column names are
    assigned from the known WDBC schema (id, diagnosis, then 10 base
    features each with _mean/_se/_worst suffixes = 30 feature columns).
 
    Parameters
    ----------
    raw_file : Path
        Path to ``wdbc.data``.
 
    Returns
    -------
    pd.DataFrame
        Raw data with 32 columns: id, diagnosis, and 30 numeric features.
    """
    df_raw = pd.read_csv(raw_file, header=None, names=COLUMNS)
    print(f"  ✓ Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    return df_raw
 
 
def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick summary of class balance and missing values.
 
    Parameters
    ----------
    df : pd.DataFrame
        The raw (or cleaned) dataset, must contain a "diagnosis" column.
    """
    print("\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts())
 
    missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing}")
    if missing == 0:
        print("  ✓ No missing values detected")
 
 
def clean_data(df: pd.DataFrame, proc_file: Path) -> pd.DataFrame:
    """
    Drop rows with missing values and persist the cleaned dataset to CSV.
 
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    proc_file : Path
        Destination CSV path for the cleaned dataset.
 
    Returns
    -------
    pd.DataFrame
        Cleaned dataset (rows with any NA dropped).
    """
    df_clean = df.dropna().copy()
    print(f"  ✓ Clean dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
 
    df_clean.to_csv(proc_file, index=False)
    print(f"  ✓ Saved to {proc_file}")
    return df_clean
 
 
def compute_top_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Rank features by standardized effect size (Cohen's d) between M and B.
 
    Only genuine biological measurement columns are considered — "id"
    (and "diagnosis" itself) are excluded, since they are not
    diagnostic features and including "id" previously caused it to be
    mis-ranked as the top discriminatory feature.
 
    Ranking method
    --------------
    Earlier versions ranked features by raw absolute mean difference.
    That is scale-dependent: features measured in larger raw units
    (e.g. ``area_worst``, in mm^2) will always dominate the ranking over
    features on a small/normalized scale (e.g. ``symmetry_mean``, a
    ratio near 0-1) regardless of how cleanly they actually separate
    the two diagnostic classes. This produces a ranking that can look
    impressive in a raw-mean bar chart, yet contradict a boxplot of the
    same "top" feature if its within-group spread is also large.
 
    Cohen's d standardizes the mean difference by the pooled standard
    deviation, so features are ranked by how *separable* the two
    groups actually are, not by which feature happens to have the
    biggest numbers. Rough interpretation: |d| ~0.2 small, ~0.5
    medium, ~0.8 large, and >1.5-2 is a very strong separation.
 
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing a "diagnosis" column plus feature columns.
    n : int, default 5
        Number of top features to return.
 
    Returns
    -------
    pd.DataFrame
        Top-``n`` features with columns: feature, malignant_mean,
        benign_mean, mean_difference, cohens_d — sorted by the absolute
        value of cohens_d, descending.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
 
    malignant = df[df["diagnosis"] == "M"][feature_cols]
    benign = df[df["diagnosis"] == "B"][feature_cols]
 
    n_m, n_b = len(malignant), len(benign)
    mean_diff = malignant.mean() - benign.mean()
 
    # Pooled standard deviation (sample, ddof=1) for Cohen's d.
    var_m, var_b = malignant.var(ddof=1), benign.var(ddof=1)
    pooled_std = np.sqrt(((n_m - 1) * var_m + (n_b - 1) * var_b) / (n_m + n_b - 2))
 
    cohens_d = mean_diff / pooled_std
 
    stats = pd.DataFrame({
        "malignant_mean": malignant.mean(),
        "benign_mean": benign.mean(),
        "mean_difference": mean_diff,
        "cohens_d": cohens_d,
    })
    stats["abs_cohens_d"] = stats["cohens_d"].abs()
    stats = stats.sort_values("abs_cohens_d", ascending=False).drop(columns="abs_cohens_d")
 
    top_n = stats.head(n).reset_index()
    top_n.columns = ["feature", "malignant_mean", "benign_mean", "mean_difference", "cohens_d"]
    return top_n
 
 
def print_top_features(top_features: pd.DataFrame) -> None:
    """
    Pretty-print the ranked feature table to stdout.
 
    Parameters
    ----------
    top_features : pd.DataFrame
        Output of :func:`compute_top_features`.
    """
    print(f"\n{'Rank':<5} {'Feature':<30} {'Malignant':>10} {'Benign':>10} {'Cohen d':>10}")
    print("-" * 68)
    for i, row in top_features.iterrows():
        print(f"{i + 1:<5} {row['feature']:<30} {row['malignant_mean']:>10.4f} "
              f"{row['benign_mean']:>10.4f} {row['cohens_d']:>10.4f}")
 
 
def plot_top_features_bar(top_features: pd.DataFrame, fig_path: Path) -> None:
    """
    Save a bar chart of standardized effect sizes (Cohen's d) for the top
    discriminatory features, with a self-contained caption.
 
    Design notes
    ------------
    This plots Cohen's d, not raw group means. Raw means are in
    different units per feature (e.g. area in mm^2 vs. symmetry as a
    0-1 ratio), so putting them on one shared axis makes large-scale
    features look "important" purely because of their units, not
    because they separate the classes well. Cohen's d is unit-free, so
    every bar is directly comparable, and the bar heights actually
    match the claim "this feature best separates malignant from
    benign" — avoiding the figure/report mismatch called out in
    feedback. A caption is written directly beneath the axes so the
    figure is interpretable without any surrounding report text.
 
    Parameters
    ----------
    top_features : pd.DataFrame
        Output of :func:`compute_top_features` (must include a
        "cohens_d" column; sign indicates whether malignant or benign
        has the higher mean).
    fig_path : Path
        Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(top_features))
 
    # Colour each bar by which class has the higher value for that feature.
    bar_colors = [COL_M if d > 0 else COL_B for d in top_features["cohens_d"]]
    bars = ax.bar(x, top_features["cohens_d"], color=bar_colors, edgecolor="white", width=0.6)
 
    for bar, d in zip(bars, top_features["cohens_d"]):
        va = "bottom" if d >= 0 else "top"
        offset = 0.03 * np.sign(d) if d != 0 else 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                f"{d:.2f}", ha="center", va=va, fontsize=9, fontweight="bold")
 
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_features["feature"], rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Cohen's d (standardized effect size)", fontsize=12)
    ax.set_title("Figure 1. Which Features Best Separate Malignant from Benign Tumours",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
 
    # Legend explaining bar colour/direction.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL_M, label="Higher in malignant tumours"),
        plt.Rectangle((0, 0), 1, 1, color=COL_B, label="Higher in benign tumours"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="best")
 
    # ══════════════════ ADDED FOR CAPTION FEEDBACK (Figure 1) ══════════════════
    # Auto-generated caption: built from the same numbers being plotted above,
    # so the text can never drift out of sync with the bars. Printed directly
    # onto the figure (not left to the surrounding report) so it stands alone.
    top_row = top_features.iloc[0]
    direction = "higher" if top_row["cohens_d"] > 0 else "lower"
    caption = (
        f"Caption: Bars show Cohen's d, the difference between malignant and benign group means "
        f"divided by the pooled standard deviation, so every feature is on the same unit-free scale "
        f"(|d| ~0.2 small, ~0.5 medium, ~0.8 large, >1.5 very large separation). "
        f"'{top_row['feature']}' shows the strongest separation (d = {top_row['cohens_d']:.2f}), "
        f"with malignant tumours {direction} on average than benign tumours for this feature. "
        f"All {len(top_features)} features shown have malignant values "
        f"{'higher' if (top_features['cohens_d'] > 0).mean() >= 0.5 else 'mixed relative to'} benign values overall."
    )
    fig.text(0.02, -0.06, caption, ha="left", va="top", fontsize=9, wrap=True,
              transform=ax.transAxes, style="italic")
    # ═══════════════════════════════════════════════════════════════════════════
 
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 1 saved")
 
 
def plot_top_feature_boxplot(df: pd.DataFrame, top_feature_row: pd.Series, fig_path: Path) -> None:
    """
    Save a boxplot of the #1 ranked feature's distribution across
    diagnostic classes, with a self-contained, data-derived caption.
 
    The caption reports the actual sample sizes, medians, and degree of
    IQR overlap for this specific feature, so a reader can verify the
    claim of separation directly against what the boxes and whiskers
    show — this keeps the figure's story consistent with the reported
    effect size in Figure 1 rather than a generic/templated caption.
 
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing "diagnosis" and the feature column.
    top_feature_row : pd.Series
        A row from :func:`compute_top_features` output (must have
        "feature" and "cohens_d"); typically ``top_features.iloc[0]``.
    fig_path : Path
        Destination PNG path.
    """
    top_feature = top_feature_row["feature"]
    cohens_d = top_feature_row["cohens_d"]
 
    df_plot = df[["diagnosis", top_feature]].copy()
    df_plot["Class"] = df_plot["diagnosis"].map({"M": "Malignant", "B": "Benign"})
 
    malignant_vals = df_plot.loc[df_plot["Class"] == "Malignant", top_feature]
    benign_vals = df_plot.loc[df_plot["Class"] == "Benign", top_feature]
 
    n_m, n_b = len(malignant_vals), len(benign_vals)
    med_m, med_b = malignant_vals.median(), benign_vals.median()
    q1_m, q3_m = malignant_vals.quantile([0.25, 0.75])
    q1_b, q3_b = benign_vals.quantile([0.25, 0.75])
 
    # Whether the two groups' interquartile ranges overlap at all.
    iqr_overlap = not (q3_m < q1_b or q3_b < q1_m)
    higher_group = "malignant" if med_m > med_b else "benign"
 
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.boxplot(
        data=df_plot, x="Class", y=top_feature, hue="Class",
        palette={"Malignant": COL_M, "Benign": COL_B},
        width=0.5, linewidth=1.2, fliersize=4, legend=False, ax=ax,
    )
    ax.set_xlabel(f"Diagnostic Class (n={n_m} malignant, n={n_b} benign)", fontsize=12)
    ax.set_ylabel(f"{top_feature}", fontsize=12)
    ax.set_title(f"Figure 2. Distribution of '{top_feature}'\n(most separable feature per Figure 1, "
                 f"d = {cohens_d:.2f})",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
 
    # ══════════════════ ADDED FOR CAPTION FEEDBACK (Figure 2) ══════════════════
    # Auto-generated caption: derived from this exact feature's own medians/IQR
    # (computed just above), and explicitly cross-references Figure 1's effect
    # size — this is what keeps the two figures' "stories" consistent with
    # each other instead of contradicting one another.
    overlap_phrase = (
        "the two groups' middle 50% of values (boxes) still overlap"
        if iqr_overlap else
        "the two groups' middle 50% of values (boxes) do not overlap"
    )
    caption = (
        f"Caption: Each box spans the interquartile range (25th-75th percentile); the line inside "
        f"is the median. Median {top_feature} is {med_m:.2f} for malignant tumours (n={n_m}) vs. "
        f"{med_b:.2f} for benign tumours (n={n_b}) — {higher_group} tumours are higher. "
        f"Despite this difference, {overlap_phrase}, meaning this feature alone cannot perfectly "
        f"classify every case, consistent with the moderate/large-but-not-total effect size "
        f"(d = {cohens_d:.2f}) reported in Figure 1."
    )
    fig.text(0.02, -0.09, caption, ha="left", va="top", fontsize=9, wrap=True,
              transform=ax.transAxes, style="italic")
    # ═══════════════════════════════════════════════════════════════════════════
 
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure 2 saved — feature: {top_feature}")
 
 
def main() -> None:
    """
    Run the full exploratory-analysis pipeline end-to-end:
    load → inspect → clean → rank features → plot figures → summary.
    """
    paths = get_project_paths(Path.cwd())
    raw_file = paths["raw_dir"] / "wdbc.data"
    proc_file = paths["proc_dir"] / "breast_cancer_clean.csv"
 
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df_raw = load_data(raw_file)
 
    print("\n" + "=" * 60)
    print("STEP 2: Inspection")
    print("=" * 60)
    inspect_data(df_raw)
 
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning")
    print("=" * 60)
    df = clean_data(df_raw, proc_file)
 
    print("\n" + "=" * 60)
    print("STEP 4: Group comparison — top 5 discriminatory features")
    print("=" * 60)
    top5 = compute_top_features(df, n=5)
    print_top_features(top5)
    top5.to_csv(paths["proc_dir"] / "top5_features_summary.csv", index=False)
 
    print("\n" + "=" * 60)
    print("STEP 5: Generating figures")
    print("=" * 60)
    top_feature_row = top5.iloc[0]
    plot_top_features_bar(top5, paths["fig_dir"] / "figure1_top5_features_bar.png")
    plot_top_feature_boxplot(df, top_feature_row, paths["fig_dir"] / "figure2_top_feature_boxplot.png")
 
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Malignant samples : {(df['diagnosis'] == 'M').sum()}")
    print(f"  Benign samples    : {(df['diagnosis'] == 'B').sum()}")
    print(f"  Top feature       : {top_feature_row['feature']} (Cohen's d = {top_feature_row['cohens_d']:.2f})")
    print(f"  Figures saved to  : {paths['fig_dir']}")
 
 
if __name__ == "__main__":
    main()
