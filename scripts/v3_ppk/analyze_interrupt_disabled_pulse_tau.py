import os
import glob
import numpy as np
import pandas as pd


# ==============================
# Settings
# ==============================

RAW_DIR = "data/raw/v3_ppk/interrupt_disable_pulse/cpu100_10ms"
OUT_DIR = "data/processed/v3_ppk/interrupt_disable_pulse"

OUT_RUN_CSV = os.path.join(OUT_DIR, "interrupt_disabled_tau_summary.csv")
OUT_CONDITION_CSV = os.path.join(OUT_DIR, "interrupt_disabled_tau_condition_summary.csv")

# PPK data is high-resolution, so keep smoothing small.
# If sampling is 10 us, 21 samples is about 0.21 ms.
SMOOTH_WINDOW_SAMPLES = 21

# Ignore early boot/current rise region
MIN_SEARCH_TIME_S = 5.0

# Windows relative to detected pulse start
PRE_IDLE_WINDOW = (-5.0, -1.0)
POST_IDLE_WINDOW = (5.0, 9.0)

# Active pulse is 10 ms. Use middle/late part to estimate busy level.
BUSY_LEVEL_WINDOW = (0.005, 0.009)

# For tau estimation
F_LOW = 0.10
F_HIGH_RISE = 1.0 - np.exp(-1.0)   # 0.632120558...
F_HIGH_FALL = np.exp(-1.0)         # 0.367879441...

DENOM_10_TO_632 = -np.log(1.0 - F_HIGH_RISE) - (-np.log(1.0 - F_LOW))
DENOM_90_TO_368 = -np.log(F_HIGH_FALL) - (-np.log(0.90))


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


def smooth_current(current_mA):
    return pd.Series(current_mA).rolling(
        window=SMOOTH_WINDOW_SAMPLES,
        center=True,
        min_periods=1
    ).mean().to_numpy()


def calc_window_stats(relative_time_s, current_mA, window):
    start_s, end_s = window
    mask = (relative_time_s >= start_s) & (relative_time_s <= end_s)

    sample_count = int(np.sum(mask))

    if sample_count == 0:
        return np.nan, np.nan, 0

    return (
        np.nanmean(current_mA[mask]),
        np.nanstd(current_mA[mask]),
        sample_count,
    )


def interpolate_crossing_time(time_s, y, target, start_idx, end_idx, direction):
    """
    Find interpolated crossing time.

    direction:
      "rising"  : y crosses upward through target
      "falling" : y crosses downward through target
    """

    if end_idx <= start_idx:
        return np.nan

    for i in range(start_idx, end_idx):
        y0 = y[i]
        y1 = y[i + 1]

        if not np.isfinite(y0) or not np.isfinite(y1):
            continue

        if direction == "rising":
            crossed = (y0 < target) and (y1 >= target)
        elif direction == "falling":
            crossed = (y0 > target) and (y1 <= target)
        else:
            raise ValueError("direction must be 'rising' or 'falling'")

        if crossed:
            t0 = time_s[i]
            t1 = time_s[i + 1]

            if y1 == y0:
                return t1

            alpha = (target - y0) / (y1 - y0)
            return t0 + alpha * (t1 - t0)

    return np.nan


def detect_pulse_region(time_s, current_smooth):
    """
    Detect short active pulse using current threshold.
    Since the active pulse is only 10 ms, percentiles are not suitable.
    Use median baseline and peak current instead.
    """

    search_mask = time_s >= MIN_SEARCH_TIME_S

    if not np.any(search_mask):
        raise ValueError("No data after MIN_SEARCH_TIME_S")

    baseline_est_mA = np.nanmedian(current_smooth[search_mask])
    peak_est_mA = np.nanmax(current_smooth[search_mask])

    amplitude_est_mA = peak_est_mA - baseline_est_mA

    if amplitude_est_mA <= 1.0:
        raise ValueError(
            f"Pulse amplitude too small. "
            f"baseline={baseline_est_mA:.3f} mA, peak={peak_est_mA:.3f} mA"
        )

    threshold_mA = baseline_est_mA + 0.5 * amplitude_est_mA

    above = (current_smooth >= threshold_mA) & search_mask

    crossing_up = np.where((~above[:-1]) & (above[1:]))[0] + 1
    crossing_down = np.where((above[:-1]) & (~above[1:]))[0] + 1

    if len(crossing_up) == 0:
        raise ValueError(
            f"Could not detect pulse start. "
            f"baseline={baseline_est_mA:.3f} mA, peak={peak_est_mA:.3f} mA, "
            f"threshold={threshold_mA:.3f} mA"
        )

    pulse_start_idx = crossing_up[0]

    valid_down = crossing_down[crossing_down > pulse_start_idx]

    if len(valid_down) == 0:
        pulse_end_idx = np.nan
        pulse_end_time_s = np.nan
    else:
        pulse_end_idx = valid_down[0]
        pulse_end_time_s = time_s[pulse_end_idx]

    pulse_start_time_s = time_s[pulse_start_idx]

    return (
        pulse_start_idx,
        pulse_start_time_s,
        pulse_end_idx,
        pulse_end_time_s,
        baseline_est_mA,
        peak_est_mA,
        threshold_mA,
    )


def estimate_tau_rise(time_s, current_smooth, pulse_start_idx, idle_level_mA, busy_level_mA):
    """
    Estimate rise tau from 10% and 63.2% crossings.

    Note:
    pulse_start_idx is detected using a 50% threshold, so the 10% crossing
    occurs before pulse_start_idx. Therefore, search starts slightly before
    pulse_start_idx.
    """

    amplitude = busy_level_mA - idle_level_mA

    if amplitude <= 0:
        return np.nan, np.nan, np.nan

    y = (current_smooth - idle_level_mA) / amplitude

    # Search from 2 ms before detected 50% crossing
    search_start_time = time_s[pulse_start_idx] - 0.002
    search_end_time = time_s[pulse_start_idx] + 0.005

    search_start_idx = max(0, np.searchsorted(time_s, search_start_time))
    search_end_idx = min(np.searchsorted(time_s, search_end_time), len(time_s) - 1)

    t10 = interpolate_crossing_time(
        time_s,
        y,
        F_LOW,
        search_start_idx,
        search_end_idx,
        direction="rising",
    )

    t632 = interpolate_crossing_time(
        time_s,
        y,
        F_HIGH_RISE,
        search_start_idx,
        search_end_idx,
        direction="rising",
    )

    if not np.isfinite(t10) or not np.isfinite(t632):
        return np.nan, t10, t632

    tau_s = (t632 - t10) / DENOM_10_TO_632
    tau_ms = tau_s * 1000.0

    return tau_ms, t10, t632


def estimate_tau_fall(time_s, current_smooth, pulse_end_idx, busy_level_mA, post_idle_level_mA):
    """
    Estimate fall tau from 90% and 36.8% crossings.

    First-order fall:
      y(t) = exp(-t/tau)

    t90  = -tau ln(0.9)
    t368 = tau

    tau = (t368 - t90) / (1 - [-ln(0.9)])
    """

    if not isinstance(pulse_end_idx, (int, np.integer)):
        return np.nan, np.nan, np.nan

    amplitude = busy_level_mA - post_idle_level_mA

    if amplitude <= 0:
        return np.nan, np.nan, np.nan

    y = (current_smooth - post_idle_level_mA) / amplitude

    # Search only near pulse end.
    search_end_time = time_s[pulse_end_idx] + 0.005
    search_end_idx = np.searchsorted(time_s, search_end_time)

    t90 = interpolate_crossing_time(
        time_s,
        y,
        0.90,
        pulse_end_idx,
        min(search_end_idx, len(time_s) - 1),
        direction="falling",
    )

    t368 = interpolate_crossing_time(
        time_s,
        y,
        F_HIGH_FALL,
        pulse_end_idx,
        min(search_end_idx, len(time_s) - 1),
        direction="falling",
    )

    if not np.isfinite(t90) or not np.isfinite(t368):
        return np.nan, t90, t368

    tau_s = (t368 - t90) / DENOM_90_TO_368
    tau_ms = tau_s * 1000.0

    return tau_ms, t90, t368


# ==============================
# Main
# ==============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    print(f"Found {len(csv_files)} files")
    print(f"RAW_DIR: {RAW_DIR}")

    if len(csv_files) == 0:
        print("No CSV files found. Check RAW_DIR.")
        return

    results = []

    for csv_path in csv_files:
        run_name = os.path.splitext(os.path.basename(csv_path))[0]

        print(f"Processing: {run_name}")

        time_s, current_mA = load_ppk_csv(csv_path)
        current_smooth = smooth_current(current_mA)

        (
            pulse_start_idx,
            pulse_start_time_s,
            pulse_end_idx,
            pulse_end_time_s,
            baseline_est_mA,
            peak_est_mA,
            threshold_mA,
        ) = detect_pulse_region(time_s, current_smooth)

        relative_time_s = time_s - pulse_start_time_s

        pre_idle_mean_mA, pre_idle_std_mA, pre_idle_samples = calc_window_stats(
            relative_time_s,
            current_mA,
            PRE_IDLE_WINDOW,
        )

        busy_level_mean_mA, busy_level_std_mA, busy_level_samples = calc_window_stats(
            relative_time_s,
            current_mA,
            BUSY_LEVEL_WINDOW,
        )

        post_idle_mean_mA, post_idle_std_mA, post_idle_samples = calc_window_stats(
            relative_time_s,
            current_mA,
            POST_IDLE_WINDOW,
        )

        idle_shift_mA = post_idle_mean_mA - pre_idle_mean_mA
        idle_shift_percent = 100.0 * idle_shift_mA / pre_idle_mean_mA

        tau_rise_ms, t10_s, t632_s = estimate_tau_rise(
            time_s,
            current_smooth,
            pulse_start_idx,
            pre_idle_mean_mA,
            busy_level_mean_mA,
        )

        tau_fall_ms, t90_s, t368_s = estimate_tau_fall(
            time_s,
            current_smooth,
            pulse_end_idx,
            busy_level_mean_mA,
            post_idle_mean_mA,
        )

        pulse_duration_ms = (
            (pulse_end_time_s - pulse_start_time_s) * 1000.0
            if np.isfinite(pulse_end_time_s)
            else np.nan
        )

        results.append({
            "run": run_name,
            "csv_path": csv_path,

            "time_min_s": np.nanmin(time_s),
            "time_max_s": np.nanmax(time_s),
            "duration_s": np.nanmax(time_s) - np.nanmin(time_s),

            "pulse_start_time_s": pulse_start_time_s,
            "pulse_end_time_s": pulse_end_time_s,
            "pulse_duration_ms": pulse_duration_ms,

            "relative_time_min_s": np.nanmin(relative_time_s),
            "relative_time_max_s": np.nanmax(relative_time_s),

            "pre_idle_window_s": f"{PRE_IDLE_WINDOW[0]} to {PRE_IDLE_WINDOW[1]}",
            "busy_level_window_s": f"{BUSY_LEVEL_WINDOW[0]} to {BUSY_LEVEL_WINDOW[1]}",
            "post_idle_window_s": f"{POST_IDLE_WINDOW[0]} to {POST_IDLE_WINDOW[1]}",

            "pre_idle_samples": pre_idle_samples,
            "busy_level_samples": busy_level_samples,
            "post_idle_samples": post_idle_samples,

            "pre_idle_mean_mA": pre_idle_mean_mA,
            "pre_idle_std_mA": pre_idle_std_mA,

            "busy_level_mean_mA": busy_level_mean_mA,
            "busy_level_std_mA": busy_level_std_mA,

            "post_idle_mean_mA": post_idle_mean_mA,
            "post_idle_std_mA": post_idle_std_mA,

            "idle_shift_mA": idle_shift_mA,
            "idle_shift_percent": idle_shift_percent,

            "baseline_est_mA": baseline_est_mA,
            "peak_est_mA": peak_est_mA,
            "threshold_mA": threshold_mA,

            "tau_rise_ms": tau_rise_ms,
            "rise_t10_s": t10_s,
            "rise_t632_s": t632_s,

            "tau_fall_ms": tau_fall_ms,
            "fall_t90_s": t90_s,
            "fall_t368_s": t368_s,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_RUN_CSV, index=False)

    print("\nSaved:")
    print(OUT_RUN_CSV)

    condition_summary = pd.DataFrame([{
        "n_runs": len(result_df),

        "mean_tau_rise_ms": result_df["tau_rise_ms"].mean(),
        "std_tau_rise_ms": result_df["tau_rise_ms"].std(),
        "min_tau_rise_ms": result_df["tau_rise_ms"].min(),
        "max_tau_rise_ms": result_df["tau_rise_ms"].max(),

        "mean_tau_fall_ms": result_df["tau_fall_ms"].mean(),
        "std_tau_fall_ms": result_df["tau_fall_ms"].std(),

        "mean_pre_idle_mA": result_df["pre_idle_mean_mA"].mean(),
        "mean_busy_level_mA": result_df["busy_level_mean_mA"].mean(),
        "mean_post_idle_mA": result_df["post_idle_mean_mA"].mean(),

        "mean_idle_shift_mA": result_df["idle_shift_mA"].mean(),
        "max_abs_idle_shift_mA": np.nanmax(np.abs(result_df["idle_shift_mA"])),

        "mean_pulse_duration_ms": result_df["pulse_duration_ms"].mean(),
        "std_pulse_duration_ms": result_df["pulse_duration_ms"].std(),
    }])

    condition_summary.to_csv(OUT_CONDITION_CSV, index=False)

    print("\nCondition-level summary:")
    print(condition_summary)

    print("\nSaved:")
    print(OUT_CONDITION_CSV)


if __name__ == "__main__":
    main()