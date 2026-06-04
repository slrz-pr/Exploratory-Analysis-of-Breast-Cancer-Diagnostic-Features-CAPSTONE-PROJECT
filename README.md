# Breast Cancer Expression Analysis

**Exploratory Analysis of Breast Cancer Diagnostic Features**
Programme: AI & ML and Computational Biology | Bversity 2026

---

## Project Overview

This project performs an exploratory data analysis (EDA) on the Breast Cancer Wisconsin Diagnostic Dataset to identify which cellular measurements differ most between malignant (M) and benign (B) tumour samples. The analysis compares group means across 30 numeric features and produces two visualisations to support scientific reporting.

---

## Dataset Citation

**Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
**Accession ID:** 17
**Portal:** UCI Machine Learning Repository
**URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
**Accessed:** 4 June 2025
**Samples:** 569 patients — M (Malignant) / B (Benign)
**Features:** 30 numeric cellular measurements

> Wolberg, W.H., Street, W.N. and Mangasarian, O.L. (1995) *Breast Cancer Wisconsin Diagnostic Dataset*. UCI Machine Learning Repository. Available at: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (Accessed: 4 June 2025).

---

## Repository Structure

```
breast-cancer-expression-analysis/
├── data/
│   ├── raw/          ← Original downloaded file (wdbc.data, wdbc.names)
│   └── processed/    ← Cleaned CSV with headers (breast_cancer_clean.csv)
├── scripts/
│   └── analysis.py   ← Full analysis script (EDA, group comparison, figures)
├── results/
│   └── figures/
│       ├── figure1_top5_features_bar.png
│       └── figure2_top_feature_boxplot.png
├── report/
│   └── project_report.pdf
└── README.md
```

---

## How to Reproduce This Analysis

### Requirements

- Python 3.8+
- pip packages: `pandas`, `matplotlib`, `seaborn`, `numpy`

### Installation

```bash
pip install pandas matplotlib seaborn numpy
```

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/breast-cancer-expression-analysis.git
   cd breast-cancer-expression-analysis
   ```

2. Download the dataset from the UCI ML Repository and place the raw files in `data/raw/`.

3. Run the analysis script:
   ```bash
   python scripts/analysis.py
   ```

4. Output figures will be saved to `results/figures/` and the processed dataset to `data/processed/`.

---

## Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Programming language |
| pandas | 2.x | Data loading and manipulation |
| numpy | 1.x | Numerical computation |
| matplotlib | 3.x | Plotting |
| seaborn | 0.x | Statistical visualisation |

---

## Commit History

Minimum 4 meaningful commits:
1. `init: set up repository structure and README`
2. `data: add raw dataset and initial inspection`
3. `analysis: clean data, compute group means, rank top 5 features`
4. `viz: add Figure 1 (bar chart) and Figure 2 (boxplot)`
5. `report: add final PDF report`

---

## Declaration

This project was completed independently as part of the Bversity 2026 AI & ML and Computational Biology programme.
