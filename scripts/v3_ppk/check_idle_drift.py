import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Change this depending on the dataset
DATA_PATTERN = "data/raw/v3_ppk/cpu_100/ppk_cpu_100_run*.csv"
OUT_DIR = "data/processed/v3_ppk/cpu_100"

os.makedirs(OUT_DIR, exist_ok=True)


def natural_key(path):
    """
    Sort files as run1, run2, ..., run10 instead of run1, run10, run2.
    """
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of these columns: {candidates}")


def detect_active_start_end(time_s, current_mA, search_start_s=8.0):
    """
    Detect active start and active end using current-based threshold.

    active_start:
        first upward crossing after search_start_s

    active_end:
        first downward crossing after active_start + 5 s
    """

    smooth_window = 200
    current_smooth = pd.Series(current_mA).rolling(
        window=smooth_window,
        center=True,
        min_periods=1
    ).mean().to_numpy()

    # Exclude power-off / boot / very low-current region
    valid_mask = (time_s >= search_start_s) & (current_smooth > 20.0)

    i_valid = current_smooth[valid_mask]

    if len(i_valid) == 0:
        raise ValueError("No valid current data after search_start_s.")

    # Estimate idle and active levels from powered workload region
    idle_level = np.nanpercentile(i_valid, 20)
    active_level = np.nanpercentile(i_valid, 90)

    threshold = (idle_level + active_level) / 2.0

    above = current_smooth >= threshold

    # Upward crossing: idle -> active
    upward_crossing_idx = np.where((~above[:-1]) & (above[1:]))[0] + 1

    upward_crossing_idx = upward_crossing_idx[
        time_s[upward_crossing_idx] >= search_start_s
    ]

    if len(upward_crossing_idx) == 0:
        raise ValueError(
            "Could not detect workload active start. "
            f"time range = {np.nanmin(time_s):.3f} to {np.nanmax(time_s):.3f} s, "
            f"idle_level = {idle_level:.3f} mA, "
            f"active_level = {active_level:.3f} mA, "
            f"threshold = {threshold:.3f} mA"
        )

    active_start_idx = upward_crossing_idx[0]
    active_start_time = time_s[active_start_idx]

    # Downward crossing: active -> final idle
    downward_crossing_idx = np.where((above[:-1]) & (~above[1:]))[0] + 1

    # Ignore tiny glitches. Active duration is expected to be about 20 s.
    # Search for falling edge after active_start + 5 s.
    downward_crossing_idx = downward_crossing_idx[
        time_s[downward_crossing_idx] >= active_start_time + 5.0
    ]

    if len(downward_crossing_idx) == 0:
        active_end_time = np.nan
    else:
        active_end_idx = downward_crossing_idx[0]
        active_end_time = time_s[active_end_idx]

    return active_start_time, active_end_time, idle_level, active_level, threshold, current_smooth


results = []

csv_files = sorted(glob.glob(DATA_PATTERN), key=natural_key)

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found: {DATA_PATTERN}")


for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    time_col = find_column(df, [
        "time_s",
        "Time(s)",
        "time",
        "timestamp_s",
        "Timestamp(ms)",
    ])

    current_col = find_column(df, [
        "current_mA",
        "Current(mA)",
        "current",
        "Current",
        "Current(uA)",
    ])

    time_s = df[time_col].to_numpy(dtype=float)
    current_raw = df[current_col].to_numpy(dtype=float)

    # Convert Timestamp(ms) -> seconds
    if "ms" in time_col.lower():
        time_s = time_s / 1000.0

    # Make time start from 0 s
    time_s = time_s - time_s[0]

    # Convert Current(uA) -> mA
    if "ua" in current_col.lower():
        current_mA = current_raw / 1000.0
    else:
        current_mA = current_raw

    run_name = os.path.splitext(os.path.basename(csv_path))[0]

    active_start_time, active_end_time, idle_level, active_level, threshold, current_smooth = (
        detect_active_start_end(time_s, current_mA)
    )

    relative_time = time_s - active_start_time

    # Initial idle:
    # stable region before active start
    initial_idle_mask = (relative_time >= -5.0) & (relative_time <= -1.0)

    # Final idle:
    # stable region after detected active end
    # active_end_time is absolute time, so convert it to relative time.
    if np.isnan(active_end_time):
        active_end_relative_time = np.nan
        final_idle_mask = np.zeros_like(relative_time, dtype=bool)
    else:
        active_end_relative_time = active_end_time - active_start_time
        final_idle_mask = (
            (relative_time >= active_end_relative_time + 5.0) &
            (relative_time <= active_end_relative_time + 9.0)
        )

    initial_count = int(np.sum(initial_idle_mask))
    final_count = int(np.sum(final_idle_mask))

    if initial_count > 0:
        initial_idle_mean = np.nanmean(current_mA[initial_idle_mask])
        initial_idle_std = np.nanstd(current_mA[initial_idle_mask])
    else:
        initial_idle_mean = np.nan
        initial_idle_std = np.nan

    if final_count > 0:
        final_idle_mean = np.nanmean(current_mA[final_idle_mask])
        final_idle_std = np.nanstd(current_mA[final_idle_mask])
    else:
        final_idle_mean = np.nan
        final_idle_std = np.nan

    if np.isnan(initial_idle_mean) or np.isnan(final_idle_mean):
        drift_mA = np.nan
        drift_percent = np.nan
    else:
        drift_mA = final_idle_mean - initial_idle_mean
        drift_percent = 100.0 * drift_mA / initial_idle_mean

    print()
    print(run_name)
    print(f"  time range: {np.nanmin(time_s):.3f} to {np.nanmax(time_s):.3f} s")
    print(f"  active_start_time: {active_start_time:.6f} s")
    print(f"  active_end_time: {active_end_time:.6f} s")
    print(f"  active_duration: {active_end_time - active_start_time:.6f} s")
    print(f"  relative_time range: {np.nanmin(relative_time):.3f} to {np.nanmax(relative_time):.3f} s")
    print(f"  initial samples: {initial_count}")
    print(f"  final samples: {final_count}")
    print(f"  drift: {drift_mA:.4f} mA")

    results.append({
        "run": run_name,
        "active_start_time_s": active_start_time,
        "active_end_time_s": active_end_time,
        "active_duration_s": active_end_time - active_start_time if not np.isnan(active_end_time) else np.nan,
        "active_end_relative_time_s": active_end_relative_time,
        "relative_time_min_s": np.nanmin(relative_time),
        "relative_time_max_s": np.nanmax(relative_time),
        "initial_idle_samples": initial_count,
        "final_idle_samples": final_count,
        "initial_idle_mean_mA": initial_idle_mean,
        "initial_idle_std_mA": initial_idle_std,
        "final_idle_mean_mA": final_idle_mean,
        "final_idle_std_mA": final_idle_std,
        "drift_mA": drift_mA,
        "drift_percent": drift_percent,
        "detected_idle_level_mA": idle_level,
        "detected_active_level_mA": active_level,
        "threshold_mA": threshold,
    })

    # Debug plot for each run
    plt.figure(figsize=(12, 4))
    plt.plot(relative_time, current_mA, linewidth=0.5, label="current")
    plt.plot(relative_time, current_smooth, linewidth=1.0, label="smoothed current")

    plt.axvline(0, color="black", linestyle="--", linewidth=1, label="active start")

    if not np.isnan(active_end_relative_time):
        plt.axvline(
            active_end_relative_time,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="active end"
        )

    plt.axvspan(-5.0, -1.0, alpha=0.2, label="initial idle window")

    if not np.isnan(active_end_relative_time):
        plt.axvspan(
            active_end_relative_time + 5.0,
            active_end_relative_time + 9.0,
            alpha=0.2,
            label="final idle window"
        )

    plt.xlim(-12, 32)
    plt.xlabel("Time from active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(run_name)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    debug_plot_path = os.path.join(OUT_DIR, f"{run_name}_idle_drift_debug.png")
    plt.savefig(debug_plot_path, dpi=200)
    plt.close()


summary_df = pd.DataFrame(results)

summary_path = os.path.join(OUT_DIR, "ppk_cpu_100_idle_drift_summary.csv")
summary_df.to_csv(summary_path, index=False)

print()
print(summary_df)
print()
print("Average drift:")
print(f"mean drift = {summary_df['drift_mA'].mean(skipna=True):.4f} mA")
print(f"std drift  = {summary_df['drift_mA'].std(skipna=True):.4f} mA")
print(f"mean drift percent = {summary_df['drift_percent'].mean(skipna=True):.4f} %")


# Plot initial vs final idle for each run
plt.figure(figsize=(10, 5))

x = np.arange(len(summary_df))

plt.scatter(x, summary_df["initial_idle_mean_mA"], label="initial idle mean")
plt.scatter(x, summary_df["final_idle_mean_mA"], label="final idle mean")

for idx, row in summary_df.iterrows():
    if not np.isnan(row["initial_idle_mean_mA"]) and not np.isnan(row["final_idle_mean_mA"]):
        plt.plot(
            [idx, idx],
            [row["initial_idle_mean_mA"], row["final_idle_mean_mA"]],
            linewidth=1,
            alpha=0.7
        )

plt.xticks(x, summary_df["run"], rotation=45, ha="right")
plt.ylabel("Current [mA]")
plt.title("Initial idle vs final idle current for each run")
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "ppk_cpu_100_idle_drift_initial_vs_final.png")
plt.savefig(plot_path, dpi=200)
plt.show()

print()
print("Saved:")
print(summary_path)
print(plot_path)
print(f"Debug plots saved in: {OUT_DIR}")