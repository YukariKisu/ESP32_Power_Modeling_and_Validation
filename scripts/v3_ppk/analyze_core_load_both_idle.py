import os
import glob
import numpy as np
import pandas as pd


# ==============================
# Settings
# ==============================

RAW_DIR = "data/raw/v3_ppk/core_load_comparison/both_idle"
OUT_DIR = "data/processed/v3_ppk/core_load_comparison"

OUT_RUN_CSV = os.path.join(OUT_DIR, "core_load_both_idle_summary.csv")
OUT_CONDITION_CSV = os.path.join(OUT_DIR, "core_load_both_idle_condition_summary.csv")

# Use stable middle part of 60 s recording
IDLE_WINDOW = (20.0, 40.0)


# ==============================
# Helper functions
# ==============================

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of columns: {candidates}")


def load_ppk_csv(csv_path):
    df = pd.read_csv(csv_path)

    time_col = find_column(df, [
        "Timestamp(ms)",
        "time_s",
        "Time(s)",
        "time",
        "timestamp_s",
    ])

    current_col = find_column(df, [
        "Current(uA)",
        "current_mA",
        "Current(mA)",
        "current",
        "Current",
    ])

    time_s = df[time_col].to_numpy(dtype=float)
    current_raw = df[current_col].to_numpy(dtype=float)

    # Convert time to seconds
    if "ms" in time_col.lower():
        time_s = time_s / 1000.0

    # Make recording start 0 s
    time_s = time_s - time_s[0]

    # Convert current to mA
    if "ua" in current_col.lower():
        current_mA = current_raw / 1000.0
    else:
        current_mA = current_raw

    return time_s, current_mA


def calc_window_stats(time_s, current_mA, window):
    start_s, end_s = window
    mask = (time_s >= start_s) & (time_s <= end_s)

    sample_count = int(np.sum(mask))

    if sample_count == 0:
        return np.nan, np.nan, 0

    mean_mA = np.nanmean(current_mA[mask])
    std_mA = np.nanstd(current_mA[mask])

    return mean_mA, std_mA, sample_count


# ==============================
# Main
# ==============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    print(f"Found {len(csv_files)} both_idle files")

    results = []

    for csv_path in csv_files:
        run_name = os.path.splitext(os.path.basename(csv_path))[0]

        print(f"Processing: {run_name}")

        time_s, current_mA = load_ppk_csv(csv_path)

        idle_mean_mA, idle_std_mA, sample_count = calc_window_stats(
            time_s,
            current_mA,
            IDLE_WINDOW,
        )

        results.append({
            "condition": "both_idle",
            "run": run_name,
            "csv_path": csv_path,
            "time_min_s": np.nanmin(time_s),
            "time_max_s": np.nanmax(time_s),
            "duration_s": np.nanmax(time_s) - np.nanmin(time_s),
            "idle_window_s": f"{IDLE_WINDOW[0]} to {IDLE_WINDOW[1]}",
            "sample_count": sample_count,
            "both_idle_mean_mA": idle_mean_mA,
            "both_idle_std_mA": idle_std_mA,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_RUN_CSV, index=False)

    print("\nSaved:")
    print(OUT_RUN_CSV)

    if len(result_df) == 0:
        print("\nNo CSV files found. Check RAW_DIR:")
        print(RAW_DIR)
        return

    condition_summary = pd.DataFrame([{
        "condition": "both_idle",
        "n_runs": len(result_df),
        "mean_both_idle_mA": result_df["both_idle_mean_mA"].mean(),
        "std_both_idle_mean_mA": result_df["both_idle_mean_mA"].std(),
        "mean_within_run_std_mA": result_df["both_idle_std_mA"].mean(),
        "min_both_idle_mA": result_df["both_idle_mean_mA"].min(),
        "max_both_idle_mA": result_df["both_idle_mean_mA"].max(),
    }])

    condition_summary.to_csv(OUT_CONDITION_CSV, index=False)

    print("\nCondition-level summary:")
    print(condition_summary)

    print("\nSaved:")
    print(OUT_CONDITION_CSV)


if __name__ == "__main__":
    main()