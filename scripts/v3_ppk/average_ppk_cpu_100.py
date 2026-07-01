import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATTERN = "data/raw/v3_ppk/cpu_100/ppk_cpu_100_run*.csv"
OUT_DIR = "data/processed/v3_ppk/cpu_100"

os.makedirs(OUT_DIR, exist_ok=True)


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path)]


FILES = sorted(glob.glob(DATA_PATTERN), key=natural_key)

if not FILES:
    raise FileNotFoundError(f"No CSV files found: {DATA_PATTERN}")

print("Files:")
for f in FILES:
    print(f)


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

    # Downsample: 100 kS/s -> about 10 kS/s
    # 0.1 ms interval
    step = 10

    return (
        time_s.iloc[::step].reset_index(drop=True),
        current_mA.iloc[::step].reset_index(drop=True),
    )


def detect_active_edges(time_s, current_mA):
    # Smooth current for edge detection
    smooth = current_mA.rolling(window=200, center=True, min_periods=1).mean()

    # Detect power-on roughly
    max_level = smooth.quantile(0.99)
    power_threshold = max_level * 0.1

    power_candidates = np.where(smooth > power_threshold)[0]
    if len(power_candidates) == 0:
        raise ValueError("Could not detect power-on")

    power_on_time = time_s.iloc[power_candidates[0]]

    # Estimate idle and active levels relative to power-on
    idle_mask = (time_s > power_on_time + 3) & (time_s < power_on_time + 9)
    active_mask = (time_s > power_on_time + 13) & (time_s < power_on_time + 25)

    idle_level = smooth[idle_mask].median()
    active_level = smooth[active_mask].median()

    if np.isnan(idle_level) or np.isnan(active_level):
        raise ValueError("Could not estimate idle or active level")

    threshold = (idle_level + active_level) / 2.0

    # Rising edge: idle -> active
    rising_candidates = np.where(
        (time_s > power_on_time + 5) & (smooth > threshold)
    )[0]

    if len(rising_candidates) == 0:
        raise ValueError("Could not detect active rising edge")

    rising_idx = rising_candidates[0]
    rising_time = time_s.iloc[rising_idx]

    # Falling edge: active -> final idle
    # This is useful for summary, but not required for alignment.
    falling_candidates = np.where(
        (time_s > rising_time + 10) & (smooth < threshold)
    )[0]

    if len(falling_candidates) == 0:
        print("Warning: Could not detect active falling edge.")
        print(f"power_on_time = {power_on_time:.3f}")
        print(f"rising_time = {rising_time:.3f}")
        print(f"idle_level = {idle_level:.3f}")
        print(f"active_level = {active_level:.3f}")
        print(f"threshold = {threshold:.3f}")

        falling_time = np.nan
    else:
        falling_idx = falling_candidates[0]
        falling_time = time_s.iloc[falling_idx]

    return power_on_time, rising_time, falling_time, idle_level, active_level, threshold


# Common relative time axis
# active start = 0 s
aligned_grid = np.arange(-10.0, 30.0, 0.0001)

aligned_currents = []
summary_rows = []

plt.figure(figsize=(12, 6))

for csv_path in FILES:
    label = os.path.basename(csv_path).replace(".csv", "")

    print(f"\nProcessing: {label}")

    time_s, current_mA = load_ppk2_csv(csv_path)

    power_on_time, rising_time, falling_time, idle_level, active_level, threshold = detect_active_edges(
        time_s, current_mA
    )

    aligned_time = time_s - rising_time

    # Interpolate each run onto the same relative time grid
    interpolated_current = np.interp(
        aligned_grid,
        aligned_time,
        current_mA,
        left=np.nan,
        right=np.nan,
    )

    aligned_currents.append(interpolated_current)

    if np.isnan(falling_time):
        active_duration = np.nan
    else:
        active_duration = falling_time - rising_time

    summary_rows.append({
        "run": label,
        "power_on_time_s": power_on_time,
        "active_start_time_s": rising_time,
        "active_end_time_s": falling_time,
        "active_duration_s": active_duration,
        "idle_level_mA": idle_level,
        "active_level_mA": active_level,
        "delta_mA": active_level - idle_level,
        "threshold_mA": threshold,
    })

    # Plot each aligned run lightly
    plt.plot(
        aligned_grid,
        interpolated_current,
        alpha=0.25,
        linewidth=0.8,
        label=label,
    )


aligned_array = np.array(aligned_currents)

# Mean and standard deviation at each relative time point
mean_current = np.nanmean(aligned_array, axis=0)
std_current = np.nanstd(aligned_array, axis=0, ddof=1)

# Plot mean waveform
plt.plot(
    aligned_grid,
    mean_current,
    linewidth=2.5,
    label="mean waveform",
)

# Plot standard deviation band
plt.fill_between(
    aligned_grid,
    mean_current - std_current,
    mean_current + std_current,
    alpha=0.25,
    label="mean ± 1 std",
)

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Time from active start [s]")
plt.ylabel("Current [mA]")
plt.title("PPK2 CPU-only 100% busy: aligned mean waveform with standard deviation")
plt.xlim(-5, 25)
plt.ylim(40, 90)
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

out_plot = os.path.join(OUT_DIR, "ppk_cpu_100_aligned_mean_std.png")
plt.savefig(out_plot, dpi=200)
plt.show()


summary_df = pd.DataFrame(summary_rows)
out_summary = os.path.join(OUT_DIR, "ppk_cpu_100_run_summary.csv")
summary_df.to_csv(out_summary, index=False)

mean_df = pd.DataFrame({
    "time_from_active_start_s": aligned_grid,
    "mean_current_mA": mean_current,
    "std_current_mA": std_current,
})

out_mean = os.path.join(OUT_DIR, "ppk_cpu_100_aligned_mean_waveform.csv")
mean_df.to_csv(out_mean, index=False)

print("\nSaved:")
print(out_plot)
print(out_summary)
print(out_mean)

print("\nRun summary:")
print(summary_df)