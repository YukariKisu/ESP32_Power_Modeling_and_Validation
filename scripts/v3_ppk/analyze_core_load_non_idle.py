import os
import glob
import numpy as np
import pandas as pd


# ==============================
# Settings
# ==============================

RAW_BASE_DIR = "data/raw/v3_ppk/core_load_comparison"

CONDITIONS = [
    "core0_work_core1_idle",
    "core0_idle_core1_work",
    "core0_work_core1_work",
]

OUT_DIR = "data/processed/v3_ppk/core_load_comparison"
OUT_CSV = os.path.join(OUT_DIR, "core_load_non_idle_summary.csv")

SMOOTH_WINDOW_SAMPLES = 200

# Relative windows after current-based alignment
INITIAL_IDLE_WINDOW = (-5.0, -1.0)
BUSY_WINDOW = (5.0, 15.0)
FINAL_IDLE_WINDOW = (25.0, 29.0)


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

    # Align recording start to 0 first
    time_s = time_s - time_s[0]

    # Convert current to mA
    if "ua" in current_col.lower():
        current_mA = current_raw / 1000.0
    else:
        current_mA = current_raw

    return time_s, current_mA


def detect_active_start(time_s, current_mA):
    """
    Detect idle -> active transition using current-based threshold.
    This ignores the manual PPK start / Enable Power timing.
    """

    current_smooth = pd.Series(current_mA).rolling(
        window=SMOOTH_WINDOW_SAMPLES,
        center=True,
        min_periods=1
    ).mean().to_numpy()

    # Use current distribution to estimate idle and busy levels
    idle_level = np.nanpercentile(current_smooth, 10)
    busy_level = np.nanpercentile(current_smooth, 90)

    threshold = (idle_level + busy_level) / 2.0

    above = current_smooth >= threshold

    crossing_idx = np.where((~above[:-1]) & (above[1:]))[0] + 1

    if len(crossing_idx) == 0:
        raise ValueError(
            "Could not detect active start. "
            f"time range = {np.nanmin(time_s):.3f} to {np.nanmax(time_s):.3f} s, "
            f"idle_level = {idle_level:.3f} mA, "
            f"busy_level = {busy_level:.3f} mA, "
            f"threshold = {threshold:.3f} mA"
        )

    # Ignore very early boot/power-on transition.
    # The workload active start should be after the initial idle phase.
    valid_crossing_idx = crossing_idx[time_s[crossing_idx] >= 5.0]

    if len(valid_crossing_idx) == 0:
        raise ValueError(
            "Could not detect workload active start after 5 s. "
            f"time range = {np.nanmin(time_s):.3f} to {np.nanmax(time_s):.3f} s, "
            f"idle_level = {idle_level:.3f} mA, "
            f"busy_level = {busy_level:.3f} mA, "
            f"threshold = {threshold:.3f} mA"
        )

    active_start_idx = valid_crossing_idx[0]
    active_start_time_s = time_s[active_start_idx]

    return active_start_time_s, idle_level, busy_level, threshold


def calc_window_stats(relative_time_s, current_mA, window):
    start_s, end_s = window
    mask = (relative_time_s >= start_s) & (relative_time_s <= end_s)

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

    results = []

    for condition in CONDITIONS:
        pattern = os.path.join(RAW_BASE_DIR, condition, "*.csv")
        csv_files = sorted(glob.glob(pattern))

        print(f"\nCondition: {condition}")
        print(f"Found {len(csv_files)} files")

        for csv_path in csv_files:
            run_name = os.path.splitext(os.path.basename(csv_path))[0]

            print(f"Processing: {run_name}")

            time_s, current_mA = load_ppk_csv(csv_path)

            active_start_time_s, detected_idle_level_mA, detected_busy_level_mA, threshold_mA = (
                detect_active_start(time_s, current_mA)
            )

            relative_time_s = time_s - active_start_time_s

            initial_idle_mean_mA, initial_idle_std_mA, initial_idle_samples = calc_window_stats(
                relative_time_s,
                current_mA,
                INITIAL_IDLE_WINDOW,
            )

            busy_mean_mA, busy_std_mA, busy_samples = calc_window_stats(
                relative_time_s,
                current_mA,
                BUSY_WINDOW,
            )

            final_idle_mean_mA, final_idle_std_mA, final_idle_samples = calc_window_stats(
                relative_time_s,
                current_mA,
                FINAL_IDLE_WINDOW,
            )

            idle_mean_mA = np.nanmean([
                initial_idle_mean_mA,
                final_idle_mean_mA,
            ])

            busy_increment_from_initial_mA = busy_mean_mA - initial_idle_mean_mA
            busy_increment_from_final_mA = busy_mean_mA - final_idle_mean_mA
            busy_increment_from_mean_idle_mA = busy_mean_mA - idle_mean_mA

            idle_drift_mA = final_idle_mean_mA - initial_idle_mean_mA
            idle_drift_percent = 100.0 * idle_drift_mA / initial_idle_mean_mA

            results.append({
                "condition": condition,
                "run": run_name,
                "csv_path": csv_path,

                "active_start_time_s": active_start_time_s,
                "relative_time_min_s": np.nanmin(relative_time_s),
                "relative_time_max_s": np.nanmax(relative_time_s),

                "initial_idle_window_s": f"{INITIAL_IDLE_WINDOW[0]} to {INITIAL_IDLE_WINDOW[1]}",
                "busy_window_s": f"{BUSY_WINDOW[0]} to {BUSY_WINDOW[1]}",
                "final_idle_window_s": f"{FINAL_IDLE_WINDOW[0]} to {FINAL_IDLE_WINDOW[1]}",

                "initial_idle_samples": initial_idle_samples,
                "busy_samples": busy_samples,
                "final_idle_samples": final_idle_samples,

                "initial_idle_mean_mA": initial_idle_mean_mA,
                "initial_idle_std_mA": initial_idle_std_mA,

                "busy_mean_mA": busy_mean_mA,
                "busy_std_mA": busy_std_mA,

                "final_idle_mean_mA": final_idle_mean_mA,
                "final_idle_std_mA": final_idle_std_mA,

                "idle_mean_mA": idle_mean_mA,

                "busy_increment_from_initial_mA": busy_increment_from_initial_mA,
                "busy_increment_from_final_mA": busy_increment_from_final_mA,
                "busy_increment_from_mean_idle_mA": busy_increment_from_mean_idle_mA,

                "idle_drift_mA": idle_drift_mA,
                "idle_drift_percent": idle_drift_percent,

                "detected_idle_level_mA": detected_idle_level_mA,
                "detected_busy_level_mA": detected_busy_level_mA,
                "threshold_mA": threshold_mA,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_CSV, index=False)

    print("\nSaved:")
    print(OUT_CSV)

    print("\nCondition-level summary:")
    summary = result_df.groupby("condition").agg(
        n_runs=("run", "count"),
        mean_initial_idle_mA=("initial_idle_mean_mA", "mean"),
        mean_busy_mA=("busy_mean_mA", "mean"),
        mean_final_idle_mA=("final_idle_mean_mA", "mean"),
        mean_idle_mA=("idle_mean_mA", "mean"),
        mean_busy_increment_mA=("busy_increment_from_mean_idle_mA", "mean"),
        std_busy_increment_mA=("busy_increment_from_mean_idle_mA", "std"),
        mean_idle_drift_mA=("idle_drift_mA", "mean"),
        max_abs_idle_drift_mA=("idle_drift_mA", lambda x: np.nanmax(np.abs(x))),
    ).reset_index()

    print(summary)

    summary_csv = os.path.join(OUT_DIR, "core_load_non_idle_condition_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(summary_csv)


if __name__ == "__main__":
    main()