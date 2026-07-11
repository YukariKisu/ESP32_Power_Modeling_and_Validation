import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RAW_ROOT = Path("data/raw/v3_ppk")
OUT_ROOT = Path("data/processed/v3_ppk/core_only_range")

MAX_ROOT = RAW_ROOT / "cpu_maximum"
MIN_ROOT = RAW_ROOT / "cpu_minimum"

# Existing arithmetic-loop baseline.
# Change this pattern if the actual location or filenames differ.
ARITHMETIC_PATTERN = (
    "data/raw/v3_ppk/cpu_100/ppk_cpu_100_run*.csv"
)

# Experiment timing relative to the detected active rising edge.
INITIAL_IDLE_START_S = -9.0
INITIAL_IDLE_END_S = -1.0

ACTIVE_START_S = 1.0
ACTIVE_END_S = 19.0

FINAL_IDLE_START_S = 21.0
FINAL_IDLE_END_S = 29.0

# Expected waveform:
# -10 to 0 s: initial idle
#   0 to 20 s: active
#  20 to 30 s: final idle

DOWNSAMPLE_STEP = 10

# Approximate common time grid after alignment.
ALIGNED_START_S = -10.0
ALIGNED_END_S = 30.0
ALIGNED_INTERVAL_S = 0.0001

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# General utilities
# ============================================================

def natural_key(value):
    """Sort strings containing numbers in natural numerical order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def normalize_workload_name(folder_name):
    """Convert folder names into readable workload names."""
    name_map = {
        "cpu_100_float": "Floating point",
        "cpu_100_seqRAM": "Sequential RAM read/write",
        "cpu_100_largeRAMcopy": "Large-buffer RAM copy",
        "cpu_100_mixRAMarith": "Mixed RAM + arithmetic",
        "cpu_100_arithmetic": "Arithmetic loop",
    }

    return name_map.get(
        folder_name,
        folder_name.replace("_", " "),
    )


def detect_columns(csv_path):
    """Detect timestamp and current columns from a PPK2 CSV."""
    columns = pd.read_csv(csv_path, nrows=0).columns

    timestamp_col = None
    current_col = None

    for column in columns:
        name = column.lower()

        if timestamp_col is None and (
            "timestamp" in name or "time" in name
        ):
            timestamp_col = column

        if current_col is None and "current" in name:
            current_col = column

    if timestamp_col is None or current_col is None:
        raise ValueError(
            f"Could not detect timestamp/current columns in {csv_path}. "
            f"Columns: {list(columns)}"
        )

    return timestamp_col, current_col


def load_ppk2_csv(csv_path):
    """Load and convert one PPK2 CSV to seconds and milliamps."""
    timestamp_col, current_col = detect_columns(csv_path)

    df = pd.read_csv(
        csv_path,
        usecols=[timestamp_col, current_col],
    )

    time_raw = pd.to_numeric(
        df[timestamp_col],
        errors="coerce",
    )

    current_raw = pd.to_numeric(
        df[current_col],
        errors="coerce",
    )

    valid_mask = time_raw.notna() & current_raw.notna()

    time_raw = time_raw[valid_mask].reset_index(drop=True)
    current_raw = current_raw[valid_mask].reset_index(drop=True)

    if len(time_raw) == 0:
        raise ValueError(f"No valid samples found in {csv_path}")

    # PPK2 timestamp: milliseconds -> seconds
    time_s = (time_raw - time_raw.iloc[0]) / 1000.0

    # PPK2 current: microamps -> milliamps
    current_mA = current_raw / 1000.0

    # 100 kS/s -> approximately 10 kS/s
    time_s = time_s.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)
    current_mA = current_mA.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)

    return time_s, current_mA


# ============================================================
# Active-edge detection
# ============================================================

def detect_active_edges(time_s, current_mA):
    """
    Detect active rising and falling edges.

    The smoothing and time windows assume:
    approximately 10 s idle, 20 s active, 10 s idle.
    """
    smooth = current_mA.rolling(
        window=200,
        center=True,
        min_periods=1,
    ).mean()

    max_level = smooth.quantile(0.99)
    power_threshold = max_level * 0.1

    power_candidates = np.where(smooth > power_threshold)[0]

    if len(power_candidates) == 0:
        raise ValueError("Could not detect power-on")

    power_on_time = float(time_s.iloc[power_candidates[0]])

    idle_mask = (
        (time_s > power_on_time + 3.0)
        & (time_s < power_on_time + 9.0)
    )

    active_mask = (
        (time_s > power_on_time + 13.0)
        & (time_s < power_on_time + 25.0)
    )

    idle_level = float(smooth[idle_mask].median())
    active_level = float(smooth[active_mask].median())

    if np.isnan(idle_level) or np.isnan(active_level):
        raise ValueError(
            "Could not estimate idle or active current level"
        )

    threshold = (idle_level + active_level) / 2.0

    rising_candidates = np.where(
        (time_s > power_on_time + 5.0)
        & (smooth > threshold)
    )[0]

    if len(rising_candidates) == 0:
        raise ValueError("Could not detect active rising edge")

    rising_idx = int(rising_candidates[0])
    rising_time = float(time_s.iloc[rising_idx])

    falling_candidates = np.where(
        (time_s > rising_time + 10.0)
        & (smooth < threshold)
    )[0]

    if len(falling_candidates) == 0:
        falling_time = np.nan
    else:
        falling_idx = int(falling_candidates[0])
        falling_time = float(time_s.iloc[falling_idx])

    return {
        "power_on_time_s": power_on_time,
        "rising_time_s": rising_time,
        "falling_time_s": falling_time,
        "detected_idle_level_mA": idle_level,
        "detected_active_level_mA": active_level,
        "threshold_mA": threshold,
    }


# ============================================================
# Per-run characterization
# ============================================================

def calculate_window_statistics(
    aligned_time,
    current_mA,
    start_s,
    end_s,
):
    """Calculate descriptive statistics for one time window."""
    mask = (
        (aligned_time >= start_s)
        & (aligned_time <= end_s)
    )

    values = current_mA[mask]

    if len(values) == 0:
        raise ValueError(
            f"No samples in window {start_s} to {end_s} s"
        )

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "sample_count": int(len(values)),
    }


def analyze_single_run(csv_path, workload, category):
    """Analyze one PPK2 CSV and return one summary row."""
    time_s, current_mA = load_ppk2_csv(csv_path)

    edge_info = detect_active_edges(time_s, current_mA)

    aligned_time = time_s - edge_info["rising_time_s"]

    initial_idle = calculate_window_statistics(
        aligned_time,
        current_mA,
        INITIAL_IDLE_START_S,
        INITIAL_IDLE_END_S,
    )

    active = calculate_window_statistics(
        aligned_time,
        current_mA,
        ACTIVE_START_S,
        ACTIVE_END_S,
    )

    final_idle = calculate_window_statistics(
        aligned_time,
        current_mA,
        FINAL_IDLE_START_S,
        FINAL_IDLE_END_S,
    )

    idle_baseline_mean = (
        initial_idle["mean"] + final_idle["mean"]
    ) / 2.0

    idle_baseline_median = (
        initial_idle["median"] + final_idle["median"]
    ) / 2.0

    delta_mean_mA = active["mean"] - idle_baseline_mean
    delta_median_mA = active["median"] - idle_baseline_median

    falling_time = edge_info["falling_time_s"]

    if np.isnan(falling_time):
        active_duration_s = np.nan
    else:
        active_duration_s = (
            falling_time - edge_info["rising_time_s"]
        )

    return {
        "category": category,
        "workload": workload,
        "run": Path(csv_path).stem,
        "file": str(csv_path),

        "power_on_time_s": edge_info["power_on_time_s"],
        "active_start_time_s": edge_info["rising_time_s"],
        "active_end_time_s": falling_time,
        "active_duration_s": active_duration_s,

        "initial_idle_mean_mA": initial_idle["mean"],
        "initial_idle_median_mA": initial_idle["median"],
        "initial_idle_std_mA": initial_idle["std"],

        "active_mean_mA": active["mean"],
        "active_median_mA": active["median"],
        "active_std_mA": active["std"],
        "active_min_mA": active["min"],
        "active_max_mA": active["max"],

        "final_idle_mean_mA": final_idle["mean"],
        "final_idle_median_mA": final_idle["median"],
        "final_idle_std_mA": final_idle["std"],

        "idle_baseline_mean_mA": idle_baseline_mean,
        "idle_baseline_median_mA": idle_baseline_median,

        "delta_mean_mA": delta_mean_mA,
        "delta_median_mA": delta_median_mA,

        "threshold_mA": edge_info["threshold_mA"],
    }


# ============================================================
# Dataset discovery
# ============================================================

def discover_category_datasets(root_dir, category):
    """
    Discover workload directories below cpu_maximum or cpu_minimum.
    """
    datasets = []

    if not root_dir.exists():
        print(f"Skipping missing directory: {root_dir}")
        return datasets

    workload_dirs = sorted(
        [path for path in root_dir.iterdir() if path.is_dir()],
        key=natural_key,
    )

    for workload_dir in workload_dirs:
        csv_files = sorted(
            glob.glob(str(workload_dir / "*.csv")),
            key=natural_key,
        )

        if not csv_files:
            print(f"No CSV files found in: {workload_dir}")
            continue

        datasets.append({
            "category": category,
            "workload": normalize_workload_name(
                workload_dir.name
            ),
            "files": csv_files,
        })

    return datasets


def discover_datasets():
    """Discover arithmetic baseline, Max datasets and Min datasets."""
    datasets = []

    arithmetic_files = sorted(
        glob.glob(ARITHMETIC_PATTERN),
        key=natural_key,
    )

    if arithmetic_files:
        datasets.append({
            "category": "maximum",
            "workload": "Arithmetic loop",
            "files": arithmetic_files,
        })
    else:
        print(
            "Warning: arithmetic baseline files were not found: "
            f"{ARITHMETIC_PATTERN}"
        )

    datasets.extend(
        discover_category_datasets(
            MAX_ROOT,
            "maximum",
        )
    )

    datasets.extend(
        discover_category_datasets(
            MIN_ROOT,
            "minimum",
        )
    )

    return datasets


# ============================================================
# Aggregation
# ============================================================

def aggregate_workloads(run_df):
    """Aggregate all runs for each workload."""
    workload_df = (
        run_df
        .groupby(
            ["category", "workload"],
            as_index=False,
        )
        .agg(
            run_count=("run", "count"),

            idle_mean_mA=(
                "idle_baseline_mean_mA",
                "mean",
            ),
            idle_between_run_std_mA=(
                "idle_baseline_mean_mA",
                "std",
            ),

            active_mean_mA=(
                "active_mean_mA",
                "mean",
            ),
            active_between_run_std_mA=(
                "active_mean_mA",
                "std",
            ),

            delta_mean_mA=(
                "delta_mean_mA",
                "mean",
            ),
            delta_between_run_std_mA=(
                "delta_mean_mA",
                "std",
            ),

            active_sample_std_mean_mA=(
                "active_std_mA",
                "mean",
            ),
        )
    )

    workload_df["active_rank"] = (
        workload_df
        .groupby("category")["active_mean_mA"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    workload_df["delta_rank"] = (
        workload_df
        .groupby("category")["delta_mean_mA"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return workload_df.sort_values(
        ["category", "active_rank", "workload"]
    ).reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================

def plot_category_comparison(workload_df, category):
    """Create active-current and delta-current bar plots."""
    category_df = workload_df[
        workload_df["category"] == category
    ].copy()

    if category_df.empty:
        return

    category_df = category_df.sort_values(
        "active_mean_mA",
        ascending=False,
    )

    labels = category_df["workload"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))

    plt.bar(
        x,
        category_df["active_mean_mA"],
        yerr=category_df["active_between_run_std_mA"].fillna(0),
        capsize=4,
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Active mean current [mA]")
    plt.title(
        f"Core-only {category}: active mean current"
    )
    plt.grid(axis="y")
    plt.tight_layout()

    out_path = (
        OUT_ROOT
        / f"core_only_{category}_active_mean_comparison.png"
    )

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")

    category_df = category_df.sort_values(
        "delta_mean_mA",
        ascending=False,
    )

    labels = category_df["workload"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))

    plt.bar(
        x,
        category_df["delta_mean_mA"],
        yerr=category_df["delta_between_run_std_mA"].fillna(0),
        capsize=4,
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Active − idle current [mA]")
    plt.title(
        f"Core-only {category}: workload current increase"
    )
    plt.grid(axis="y")
    plt.tight_layout()

    out_path = (
        OUT_ROOT
        / f"core_only_{category}_delta_comparison.png"
    )

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    datasets = discover_datasets()

    if not datasets:
        raise FileNotFoundError(
            "No core-only CSV datasets were found."
        )

    run_rows = []

    for dataset in datasets:
        category = dataset["category"]
        workload = dataset["workload"]
        files = dataset["files"]

        print()
        print("=" * 70)
        print(f"Category: {category}")
        print(f"Workload: {workload}")
        print(f"Run count: {len(files)}")
        print("=" * 70)

        for csv_path in files:
            print(f"Processing: {csv_path}")

            try:
                row = analyze_single_run(
                    csv_path,
                    workload,
                    category,
                )
                run_rows.append(row)

            except Exception as exc:
                print(f"ERROR: {csv_path}")
                print(f"Reason: {exc}")

    if not run_rows:
        raise RuntimeError(
            "All CSV files failed during analysis."
        )

    run_df = pd.DataFrame(run_rows)

    workload_df = aggregate_workloads(run_df)

    run_summary_path = (
        OUT_ROOT / "core_only_run_summary.csv"
    )

    workload_summary_path = (
        OUT_ROOT / "core_only_workload_summary.csv"
    )

    run_df.to_csv(
        run_summary_path,
        index=False,
    )

    workload_df.to_csv(
        workload_summary_path,
        index=False,
    )

    print()
    print("Saved:")
    print(run_summary_path)
    print(workload_summary_path)

    plot_category_comparison(
        workload_df,
        "maximum",
    )

    plot_category_comparison(
        workload_df,
        "minimum",
    )

    maximum_df = workload_df[
        workload_df["category"] == "maximum"
    ].sort_values(
        "active_mean_mA",
        ascending=False,
    )

    minimum_df = workload_df[
        workload_df["category"] == "minimum"
    ].sort_values(
        "active_mean_mA",
        ascending=True,
    )

    print()
    print("Core-only workload summary:")
    print(
        workload_df[
            [
                "category",
                "workload",
                "run_count",
                "idle_mean_mA",
                "active_mean_mA",
                "delta_mean_mA",
                "active_rank",
                "delta_rank",
            ]
        ].to_string(index=False)
    )

    if not maximum_df.empty:
        max_row = maximum_df.iloc[0]

        print()
        print("Detected Core-only Max candidate:")
        print(f"Workload: {max_row['workload']}")
        print(
            f"Active mean: "
            f"{max_row['active_mean_mA']:.3f} mA"
        )
        print(
            f"Delta: "
            f"{max_row['delta_mean_mA']:.3f} mA"
        )

    if not minimum_df.empty:
        min_row = minimum_df.iloc[0]

        print()
        print("Detected Core-only Min candidate:")
        print(f"Workload/state: {min_row['workload']}")
        print(
            f"Active/state mean: "
            f"{min_row['active_mean_mA']:.3f} mA"
        )

    if not maximum_df.empty and not minimum_df.empty:
        core_range_mA = (
            maximum_df.iloc[0]["active_mean_mA"]
            - minimum_df.iloc[0]["active_mean_mA"]
        )

        print()
        print(
            f"Observed Core-only operating range: "
            f"{core_range_mA:.3f} mA"
        )


if __name__ == "__main__":
    main()