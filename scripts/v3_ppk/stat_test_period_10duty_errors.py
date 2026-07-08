import os
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kruskal, ks_2samp


# ============================================================
# Settings
# ============================================================

INPUT_CSV = (
    "data/processed/v3_ppk/period_10duty_boxplots/"
    "period_10duty_firmware_binary_distribution_data.csv"
)

OUT_DIR = "data/processed/v3_ppk/period_10duty_boxplots/statistical_tests"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05

# Use the same error type as the boxplot you want to statistically support.
# For signed prediction error boxplot:
ERROR_COL = "abs_prediction_error_mA"

# For absolute prediction error boxplot, use this instead:
# ERROR_COL = "abs_prediction_error_mA"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_cols = ["condition", "condition_name", "run", ERROR_COL]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df[np.isfinite(df[ERROR_COL])].copy()

conditions = list(df["condition"].drop_duplicates())

print("Conditions:")
print(conditions)

groups = []

summary_rows = []

for condition in conditions:
    values = df.loc[df["condition"] == condition, ERROR_COL].dropna().to_numpy()
    groups.append(values)

    summary_rows.append({
        "condition": condition,
        "n_samples": len(values),
        "mean_error_mA": np.mean(values),
        "median_error_mA": np.median(values),
        "std_error_mA": np.std(values, ddof=1),
        "q1_error_mA": np.quantile(values, 0.25),
        "q3_error_mA": np.quantile(values, 0.75),
        "iqr_error_mA": np.quantile(values, 0.75) - np.quantile(values, 0.25),
        "min_error_mA": np.min(values),
        "max_error_mA": np.max(values),
    })

summary_df = pd.DataFrame(summary_rows)

summary_csv = os.path.join(
    OUT_DIR,
    f"period_10duty_{ERROR_COL}_distribution_summary.csv",
)
summary_df.to_csv(summary_csv, index=False)


# ============================================================
# Kruskal-Wallis test
# ============================================================

kw_stat, kw_p = kruskal(*groups)

kruskal_df = pd.DataFrame([
    {
        "test": "Kruskal-Wallis",
        "error_column": ERROR_COL,
        "groups": ", ".join(conditions),
        "statistic": kw_stat,
        "p_value": kw_p,
        "alpha": ALPHA,
        "significant": kw_p < ALPHA,
        "interpretation": (
            "Significant difference found among groups"
            if kw_p < ALPHA
            else "No statistically significant difference found among groups"
        ),
    }
])

kruskal_csv = os.path.join(
    OUT_DIR,
    f"kruskal_wallis_period_10duty_{ERROR_COL}.csv",
)
kruskal_df.to_csv(kruskal_csv, index=False)


# ============================================================
# Pairwise Kolmogorov-Smirnov tests
# ============================================================

ks_rows = []
pairs = list(itertools.combinations(conditions, 2))
n_comparisons = len(pairs)

for condition_a, condition_b in pairs:
    values_a = df.loc[df["condition"] == condition_a, ERROR_COL].dropna().to_numpy()
    values_b = df.loc[df["condition"] == condition_b, ERROR_COL].dropna().to_numpy()

    stat, p_raw = ks_2samp(values_a, values_b)

    p_bonferroni = min(p_raw * n_comparisons, 1.0)

    ks_rows.append({
        "test": "Kolmogorov-Smirnov",
        "error_column": ERROR_COL,
        "group_1": condition_a,
        "group_2": condition_b,
        "statistic": stat,
        "p_value_raw": p_raw,
        "p_value_bonferroni": p_bonferroni,
        "alpha": ALPHA,
        "significant_raw": p_raw < ALPHA,
        "significant_bonferroni": p_bonferroni < ALPHA,
        "interpretation": (
            "Significant pairwise distribution difference found"
            if p_bonferroni < ALPHA
            else "No statistically significant pairwise distribution difference found"
        ),
    })

ks_df = pd.DataFrame(ks_rows)

ks_csv = os.path.join(
    OUT_DIR,
    f"ks_pairwise_period_10duty_{ERROR_COL}.csv",
)
ks_df.to_csv(ks_csv, index=False)


# ============================================================
# Print results
# ============================================================

print("\n=== Distribution summary ===")
print(summary_df)

print("\n=== Kruskal-Wallis ===")
print(kruskal_df)

print("\n=== Pairwise KS tests ===")
print(ks_df)

print("\nSaved:")
print(summary_csv)
print(kruskal_csv)
print(ks_csv)