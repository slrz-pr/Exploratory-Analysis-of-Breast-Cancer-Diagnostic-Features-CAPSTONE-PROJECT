"""
Breast Cancer Wisconsin Diagnostic Dataset — Exploratory Analysis
=================================================================
Programme : AI & ML and Computational Biology | Bversity 2026
Author    : [Your Name]
Date      : 2025-06-04

Dataset   : Wolberg, W.H., Street, W.N. and Mangasarian, O.L. (1995)
            Breast Cancer Wisconsin Diagnostic Dataset.
            UCI Machine Learning Repository.
            https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
            Accessed: 4 June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path("Exploratory-Analysis-of-Breast-Cancer-Diagnostic-Features-CAPSTONE-PROJECT").resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "results" / "figures"

PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

raw_file = RAW_DIR / "wdbc.data"
df_raw = FEATURE_NAMES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]
SUFFIXES = ["_mean", "_se", "_worst"]
COLUMNS = (
    ["id", "diagnosis"] +
    [f + s for s in SUFFIXES for f in FEATURE_NAMES]
)

df_raw = pd.read_csv(raw_file, header=None, names=COLUMNS)
print(f"  ✓ Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Inspection")
print("=" * 60)

print(f"\nDiagnosis distribution:")
print(df_raw["diagnosis"].value_counts())
missing = df_raw.isnull().sum().sum()
print(f"\nTotal missing values: {missing}")
if missing == 0:
    print("  ✓ No missing values detected")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Cleaning")
print("=" * 60)

df = df_raw.dropna().copy()
print(f"  ✓ Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")

proc_file = PROC_DIR / "breast_cancer_clean.csv"
df.to_csv(proc_file, index=False)
print(f"  ✓ Saved to {proc_file}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GROUP COMPARISON & RANKING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Group comparison — top 5 discriminatory features")
print("=" * 60)

feature_cols = [c for c in df.columns if c != "diagnosis"]
malignant = df[df["diagnosis"] == "M"][feature_cols]
benign    = df[df["diagnosis"] == "B"][feature_cols]

means = pd.DataFrame({
    "malignant_mean": malignant.mean(),
    "benign_mean"   : benign.mean(),
})
means["absolute_difference"] = (means["malignant_mean"] - means["benign_mean"]).abs()
means = means.sort_values("absolute_difference", ascending=False)

top5 = means.head(5).reset_index()
top5.columns = ["feature", "malignant_mean", "benign_mean", "absolute_difference"]

print(f"\n{'Rank':<5} {'Feature':<30} {'Malignant':>10} {'Benign':>10} {'Abs Diff':>10}")
print("-" * 68)
for i, row in top5.iterrows():
    print(f"{i+1:<5} {row['feature']:<30} {row['malignant_mean']:>10.4f} "
          f"{row['benign_mean']:>10.4f} {row['absolute_difference']:>10.4f}")

top5.to_csv(PROC_DIR / "top5_features_summary.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Generating figures")
print("=" * 60)

COL_M = "#C0392B"
COL_B = "#2980B9"

# Figure 1 — Bar chart
fig1, ax1 = plt.subplots(figsize=(11, 6))
x = np.arange(len(top5))
w = 0.35
bars_m = ax1.bar(x - w/2, top5["malignant_mean"], w, label="Malignant", color=COL_M, edgecolor="white")
bars_b = ax1.bar(x + w/2, top5["benign_mean"],    w, label="Benign",    color=COL_B, edgecolor="white")

for bar in bars_m:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in bars_b:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(top5["feature"], rotation=20, ha="right", fontsize=10)
ax1.set_xlabel("Feature", fontsize=12)
ax1.set_ylabel("Mean Value", fontsize=12)
ax1.set_title("Top 5 Discriminatory Features: Malignant vs Benign Group Means",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.spines[["top","right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "figure1_top5_features_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  ✓ Figure 1 saved")

# Figure 2 — Boxplot
top_feature = top5.iloc[0]["feature"]
df_plot = df[["diagnosis", top_feature]].copy()
df_plot["Class"] = df_plot["diagnosis"].map({"M": "Malignant", "B": "Benign"})

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_plot, x="Class", y=top_feature,
            palette={"Malignant": COL_M, "Benign": COL_B},
            width=0.5, linewidth=1.2, fliersize=4, ax=ax2)
ax2.set_xlabel("Diagnostic Class", fontsize=12)
ax2.set_ylabel(f"{top_feature}", fontsize=12)
ax2.set_title(f"Distribution of '{top_feature}'\nacross Malignant and Benign Diagnostic Classes",
              fontsize=13, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "figure2_top_feature_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  ✓ Figure 2 saved — feature: {top_feature}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"  Malignant samples : {(df['diagnosis']=='M').sum()}")
print(f"  Benign samples    : {(df['diagnosis']=='B').sum()}")
print(f"  Top feature       : {top_feature}")
print(f"  Figures saved to  : {FIG_DIR}")
