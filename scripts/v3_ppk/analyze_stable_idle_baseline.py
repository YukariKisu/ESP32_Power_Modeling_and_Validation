import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

INPUT_DIR = "data/raw/v3_ppk/idle_baseline"
OUT_DIR = "data/processed/v3_ppk/idle_baseline"

os.makedirs(OUT_DIR, exist_ok=True)

# PPK2 raw data: timestamp ms, current uA
DOWNSAMPLE_STEP = 10

# Use stable middle window
EVAL_WINDOW_S = (5.0, 55.0)

# Remove obvious corrupted samples / power-on spikes
CURRENT_MIN_MA = 30.0
CURRENT_MAX_MA = 100.0

SHOW_PLOTS = False


# ============================================================
# Utility
# ============================================================

def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path)]


def detect_columns(csv_path):
    cols = pd.read_csv(csv_path, nrows=0).columns

    timestamp_col = None
    current_col = None

    for col in cols:
        name = col.lower()

        if timestamp_col is None and ("timestamp" in name or "time" in name):
            timestamp_col = col

        if current_col is None and "current" in name:
            current_col = col

    if timestamp_col is None or current_col is None:
        print("Columns found:")
        print(list(cols))
        raise ValueError(f"Could not detect timestamp/current columns in {csv_path}")

    return timestamp_col, current_col


def load_ppk2_csv(csv_path):
    timestamp_col, current_col = detect_columns(csv_path)

    df = pd.read_csv(csv_path, usecols=[timestamp_col, current_col])

    time_raw = df[timestamp_col].astype(float)
    current_raw = df[current_col].astype(float)

    # PPK2 timestamp: ms -> s
    time_s = (time_raw - time_raw.iloc[0]) / 1000.0

    # PPK2 current: uA -> mA
    current_mA = current_raw / 1000.0

    return (
        time_s.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True),
        current_mA.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True),
    )


# ============================================================
# Main
# ============================================================

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.csv")),
    key=natural_key,
)

if not files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

summary_rows = []
all_distribution_rows = []

for csv_path in files:
    run_name = os.path.basename(csv_path).replace(".csv", "")
    print(f"Processing: {run_name}")

    time_s, current_mA = load_ppk2_csv(csv_path)

    mask = (
        (time_s >= EVAL_WINDOW_S[0])
        & (time_s <= EVAL_WINDOW_S[1])
        & np.isfinite(current_mA)
        & (current_mA >= CURRENT_MIN_MA)
        & (current_mA <= CURRENT_MAX_MA)
    )

    current_eval = current_mA[mask].to_numpy()
    time_eval = time_s[mask].to_numpy()

    if len(current_eval) == 0:
        print(f"# SKIP {run_name}: no valid samples")
        continue

    summary_rows.append({
        "run": run_name,
        "n_samples": len(current_eval),
        "eval_start_s": EVAL_WINDOW_S[0],
        "eval_end_s": EVAL_WINDOW_S[1],

        "mean_current_mA": np.mean(current_eval),
        "median_current_mA": np.median(current_eval),
        "std_current_mA": np.std(current_eval, ddof=1),
        "min_current_mA": np.min(current_eval),
        "max_current_mA": np.max(current_eval),

        "p5_current_mA": np.percentile(current_eval, 5),
        "p95_current_mA": np.percentile(current_eval, 95),

        "q1_current_mA": np.percentile(current_eval, 25),
        "q3_current_mA": np.percentile(current_eval, 75),
        "iqr_current_mA": np.percentile(current_eval, 75) - np.percentile(current_eval, 25),
    })

    run_dist = pd.DataFrame({
        "run": run_name,
        "time_s": time_eval,
        "current_mA": current_eval,
    })

    all_distribution_rows.append(run_dist)


summary_df = pd.DataFrame(summary_rows)

out_summary = os.path.join(
    OUT_DIR,
    "stable_idle_baseline_summary_by_run.csv",
)
summary_df.to_csv(out_summary, index=False)

combined_summary = pd.DataFrame([{
    "n_runs": len(summary_df),

    "mean_of_run_mean_current_mA": summary_df["mean_current_mA"].mean(),
    "std_of_run_mean_current_mA": summary_df["mean_current_mA"].std(ddof=1),

    "mean_of_run_median_current_mA": summary_df["median_current_mA"].mean(),
    "std_of_run_median_current_mA": summary_df["median_current_mA"].std(ddof=1),

    "mean_of_run_std_current_mA": summary_df["std_current_mA"].mean(),

    "mean_p5_current_mA": summary_df["p5_current_mA"].mean(),
    "mean_p95_current_mA": summary_df["p95_current_mA"].mean(),

    "mean_q1_current_mA": summary_df["q1_current_mA"].mean(),
    "mean_q3_current_mA": summary_df["q3_current_mA"].mean(),
    "mean_iqr_current_mA": summary_df["iqr_current_mA"].mean(),
}])

out_combined = os.path.join(
    OUT_DIR,
    "stable_idle_baseline_summary_combined.csv",
)
combined_summary.to_csv(out_combined, index=False)

distribution_df = pd.concat(all_distribution_rows, ignore_index=True)

out_distribution = os.path.join(
    OUT_DIR,
    "stable_idle_baseline_distribution_data.csv",
)
distribution_df.to_csv(out_distribution, index=False)


# ============================================================
# Plots
# ============================================================

# 1. Current distribution boxplot by run
plt.figure(figsize=(10, 6))

data_by_run = [
    distribution_df.loc[distribution_df["run"] == run, "current_mA"].to_numpy()
    for run in summary_df["run"]
]

plt.boxplot(
    data_by_run,
    labels=summary_df["run"],
    showfliers=False,
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Current [mA]")
plt.title("Stable idle baseline current distribution by run")
plt.grid(True, axis="y")
plt.tight_layout()

out_boxplot = os.path.join(
    OUT_DIR,
    "stable_idle_baseline_current_boxplot_by_run.png",
)
plt.savefig(out_boxplot, dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# 2. Mean current by run
plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["run"],
    summary_df["mean_current_mA"],
    marker="o",
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean current [mA]")
plt.title("Stable idle baseline mean current by run")
plt.grid(True)
plt.tight_layout()

out_mean_plot = os.path.join(
    OUT_DIR,
    "stable_idle_baseline_mean_current_by_run.png",
)
plt.savefig(out_mean_plot, dpi=200)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


print("\nSaved:")
print(out_summary)
print(out_combined)
print(out_distribution)
print(out_boxplot)
print(out_mean_plot)

print("\nCombined stable idle baseline summary:")
print(combined_summary)