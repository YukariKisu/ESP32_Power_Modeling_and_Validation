import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

INTENSITIES = [75, 50, 25]

BASE_RAW_DIR = "data/raw/v3_ppk"
BASE_OUT_DIR = "data/processed/v3_ppk"

# CPU100-based first-order model parameters
I_IDLE_MODEL_MA = 47.2668
I_ACTIVE_MODEL_MA = 67.4556
GAIN_MODEL_MA = I_ACTIVE_MODEL_MA - I_IDLE_MODEL_MA
TAU_S = 0.00049

# PPK2 raw data: 100 kS/s
# Processed waveform: 10 kS/s = 0.1 ms interval
DOWNSAMPLE_STEP = 10
GRID_DT_S = 0.0001

# Common aligned time axis
ALIGNED_GRID = np.arange(-10.0, 30.0, GRID_DT_S)

# Evaluation window
FULL_ACTIVE_WINDOW_S = (0.0, 10.0)
TRANSIENT_WINDOW_S = (0.0, 0.005)
STEADY_WINDOW_S = (0.010, 10.0)

# For estimating binary u(t) from measured current
# Small smoothing to suppress noise but keep switching
U_DETECT_SMOOTH_WINDOW_S = 0.0005  # 0.5 ms
U_DETECT_SMOOTH_WINDOW_SAMPLES = int(U_DETECT_SMOOTH_WINDOW_S / GRID_DT_S)

if U_DETECT_SMOOTH_WINDOW_SAMPLES < 1:
    U_DETECT_SMOOTH_WINDOW_SAMPLES = 1

if U_DETECT_SMOOTH_WINDOW_SAMPLES % 2 == 0:
    U_DETECT_SMOOTH_WINDOW_SAMPLES += 1

# Threshold between idle-like and active-like current
U_THRESHOLD_MA = (I_IDLE_MODEL_MA + I_ACTIVE_MODEL_MA) / 2.0

SHOW_PLOTS = False


# ============================================================
# Utility functions
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


def smooth_waveform(values, window_samples):
    values = np.asarray(values, dtype=float)
    original_nan_mask = ~np.isfinite(values)

    smoothed = (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
        .copy()
    )

    smoothed[original_nan_mask] = np.nan
    return smoothed


def detect_active_edges(time_s, current_mA):
    """
    Current-based time alignment using CPU100-derived high-current threshold.

    This is more robust for duty-cycle workloads such as CPU25,
    where the active-window median may still be close to idle.
    """

    # Smooth current for robust edge detection
    smooth = current_mA.rolling(window=20, center=True, min_periods=1).mean()
    # window=20 at 10 kS/s -> about 2 ms smoothing

    # Detect power-on roughly
    max_level = smooth.quantile(0.99)
    power_threshold = max_level * 0.1

    power_candidates = np.where(smooth > power_threshold)[0]
    if len(power_candidates) == 0:
        raise ValueError("Could not detect power-on")

    power_on_time = time_s.iloc[power_candidates[0]]

    # Estimate idle level before workload starts
    idle_mask = (time_s > power_on_time + 3) & (time_s < power_on_time + 9)
    idle_level = smooth[idle_mask].median()

    if np.isnan(idle_level):
        raise ValueError("Could not estimate idle level")

    # Use CPU100-derived high-current threshold
    high_current_threshold = (I_IDLE_MODEL_MA + I_ACTIVE_MODEL_MA) / 2.0

    # Detect first CPU100-like busy pulse after initial idle
    rising_candidates = np.where(
        (time_s > power_on_time + 5)
        & (smooth > high_current_threshold)
    )[0]

    if len(rising_candidates) == 0:
        raise ValueError(
            "Could not detect high-current busy pulse. "
            "The workload may not reach CPU100-like active current."
        )

    rising_idx = rising_candidates[0]
    rising_time = time_s.iloc[rising_idx]

    # Estimate active high level using only samples above the high-current threshold
    active_mask = (
        (time_s > rising_time)
        & (time_s < rising_time + 20)
        & (smooth > high_current_threshold)
    )

    active_level = smooth[active_mask].median()

    if np.isnan(active_level):
        active_level = np.nan

    # Detect end of active phase:
    # after about 10 s, look for sustained low current.
    falling_candidates = np.where(
        (time_s > rising_time + 10)
        & (smooth < high_current_threshold)
    )[0]

    if len(falling_candidates) == 0:
        falling_time = np.nan
    else:
        falling_idx = falling_candidates[0]
        falling_time = time_s.iloc[falling_idx]

    return (
        power_on_time,
        rising_time,
        falling_time,
        idle_level,
        active_level,
        high_current_threshold,
    )



def estimate_binary_input_from_current(time_grid, measured_current_mA):
    """
    Estimate u(t) from measured current.

    u(t) = 1 if current looks like CPU100 active state
    u(t) = 0 if current looks like idle/wait state

    For t < 0, force u(t) = 0.
    """

    smoothed_for_u = smooth_waveform(
        measured_current_mA,
        U_DETECT_SMOOTH_WINDOW_SAMPLES,
    )

    u_t = np.zeros_like(time_grid, dtype=float)

    valid_mask = np.isfinite(smoothed_for_u)
    active_time_mask = time_grid >= 0

    high_mask = (
        valid_mask
        & active_time_mask
        & (smoothed_for_u >= U_THRESHOLD_MA)
    )

    u_t[high_mask] = 1.0

    return u_t, smoothed_for_u


def simulate_first_order_model_binary_input(time_grid, u_t):
    """
    Simulate:
        dI/dt = (I_idle + gain*u(t) - I) / tau

    Uses exact discrete update for zero-order-hold input.
    """

    I_pred = np.full_like(time_grid, np.nan, dtype=float)

    first_valid_idx = 0
    I_pred[first_valid_idx] = I_IDLE_MODEL_MA

    for k in range(first_valid_idx + 1, len(time_grid)):
        dt = time_grid[k] - time_grid[k - 1]

        if dt <= 0 or not np.isfinite(dt):
            I_pred[k] = I_pred[k - 1]
            continue

        target = I_IDLE_MODEL_MA + GAIN_MODEL_MA * u_t[k - 1]

        alpha = np.exp(-dt / TAU_S)

        I_pred[k] = target + (I_pred[k - 1] - target) * alpha

    return I_pred


def calculate_mae(time_s, measured_mA, predicted_mA, window):
    start_s, end_s = window

    mask = (
        (time_s >= start_s)
        & (time_s <= end_s)
        & np.isfinite(measured_mA)
        & np.isfinite(predicted_mA)
    )

    if np.sum(mask) == 0:
        return np.nan

    return np.mean(np.abs(measured_mA[mask] - predicted_mA[mask]))


def calculate_mean_error(time_s, measured_mA, predicted_mA, window):
    start_s, end_s = window

    mask = (
        (time_s >= start_s)
        & (time_s <= end_s)
        & np.isfinite(measured_mA)
        & np.isfinite(predicted_mA)
    )

    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan

    measured_mean = np.mean(measured_mA[mask])
    predicted_mean = np.mean(predicted_mA[mask])
    signed_error = measured_mean - predicted_mean

    return measured_mean, predicted_mean, signed_error


def save_plot_for_intensity(
    intensity,
    aligned_currents,
    predicted_currents,
    u_inputs,
    mean_measured,
    mean_predicted,
    mean_u,
    out_dir,
):
    """
    Save plot: measured vs predicted.
    """

    # Full plot
    plt.figure(figsize=(12, 6))

    for i, current in enumerate(aligned_currents):
        plt.plot(
            ALIGNED_GRID,
            current,
            alpha=0.15,
            linewidth=0.6,
            label="individual measured runs" if i == 0 else None,
        )

    for i, pred in enumerate(predicted_currents):
        plt.plot(
            ALIGNED_GRID,
            pred,
            alpha=0.15,
            linewidth=0.6,
            linestyle="--",
            label="individual predicted runs" if i == 0 else None,
        )

    plt.plot(
        ALIGNED_GRID,
        mean_measured,
        linewidth=2.2,
        label="mean measured current",
    )

    plt.plot(
        ALIGNED_GRID,
        mean_predicted,
        linewidth=2.2,
        linestyle="--",
        label="mean predicted current",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"CPU {intensity}%: binary input ODE validation "
        "(u(t) estimated from PPK current)"
    )
    plt.xlim(-2, 12)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_plot = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_binary_input_measured_vs_predicted.png",
    )

    plt.savefig(out_plot, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Zoom plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        ALIGNED_GRID,
        mean_measured,
        linewidth=2.2,
        label="mean measured current",
    )

    plt.plot(
        ALIGNED_GRID,
        mean_predicted,
        linewidth=2.2,
        linestyle="--",
        label="mean predicted current",
    )

    # scale mean u(t) for visualization
    u_visual = I_IDLE_MODEL_MA + GAIN_MODEL_MA * mean_u

    plt.plot(
        ALIGNED_GRID,
        u_visual,
        linewidth=1.5,
        linestyle=":",
        label="mean estimated input level",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"CPU {intensity}%: transient/switching zoom "
        "(binary u(t) model)"
    )
    plt.xlim(-0.02, 0.20)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_zoom_plot = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_binary_input_zoom.png",
    )

    plt.savefig(out_zoom_plot, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return out_plot, out_zoom_plot


# ============================================================
# Main
# ============================================================

all_summary_rows = []

for intensity in INTENSITIES:
    expected_duty = intensity / 100.0

    data_pattern = f"{BASE_RAW_DIR}/cpu_{intensity}/ppk_cpu_{intensity}_run*.csv"
    out_dir = f"{BASE_OUT_DIR}/cpu_{intensity}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(data_pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No CSV files found: {data_pattern}")

    print("\n" + "=" * 70)
    print(f"Processing CPU_{intensity}")
    print("=" * 70)

    aligned_currents = []
    predicted_currents = []
    u_inputs = []

    run_rows = []

    for csv_path in files:
        label = os.path.basename(csv_path).replace(".csv", "")

        print(f"Processing: {label}")

        time_s, current_mA = load_ppk2_csv(csv_path)

        (
            power_on_time,
            rising_time,
            falling_time,
            idle_level,
            active_level,
            alignment_threshold,
        ) = detect_active_edges(time_s, current_mA)

        aligned_time = time_s - rising_time

        measured_interp = np.interp(
            ALIGNED_GRID,
            aligned_time,
            current_mA,
            left=np.nan,
            right=np.nan,
        )

        u_t, smoothed_for_u = estimate_binary_input_from_current(
            ALIGNED_GRID,
            measured_interp,
        )

        predicted_current = simulate_first_order_model_binary_input(
            ALIGNED_GRID,
            u_t,
        )

        raw_mae_full = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            FULL_ACTIVE_WINDOW_S,
        )

        raw_mae_transient = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            TRANSIENT_WINDOW_S,
        )

        raw_mae_steady = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            STEADY_WINDOW_S,
        )

        measured_mean, predicted_mean, mean_signed_error = calculate_mean_error(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            STEADY_WINDOW_S,
        )

        active_mask = (
            (ALIGNED_GRID >= STEADY_WINDOW_S[0])
            & (ALIGNED_GRID <= STEADY_WINDOW_S[1])
            & np.isfinite(measured_interp)
        )

        estimated_duty = np.mean(u_t[active_mask]) if np.sum(active_mask) > 0 else np.nan

        aligned_currents.append(measured_interp)
        predicted_currents.append(predicted_current)
        u_inputs.append(u_t)

        run_rows.append({
            "run": label,
            "cpu_intensity_percent": intensity,
            "expected_duty": expected_duty,
            "estimated_duty_from_current": estimated_duty,

            "power_on_time_s": power_on_time,
            "active_start_time_s": rising_time,
            "active_end_time_s": falling_time,

            "alignment_idle_level_mA": idle_level,
            "alignment_active_level_mA": active_level,
            "alignment_threshold_mA": alignment_threshold,

            "u_threshold_mA": U_THRESHOLD_MA,

            "raw_mae_full_active_0_to_10s_mA": raw_mae_full,
            "raw_mae_transient_0_to_5ms_mA": raw_mae_transient,
            "raw_mae_steady_10ms_to_10s_mA": raw_mae_steady,

            "measured_steady_mean_current_mA": measured_mean,
            "predicted_steady_mean_current_mA": predicted_mean,
            "steady_mean_signed_error_mA": mean_signed_error,
            "steady_mean_abs_error_mA": abs(mean_signed_error),
        })

        # Save one debug waveform per run
        debug_df = pd.DataFrame({
            "time_from_active_start_s": ALIGNED_GRID,
            "measured_current_mA": measured_interp,
            "smoothed_current_for_u_detection_mA": smoothed_for_u,
            "estimated_u_t": u_t,
            "predicted_current_mA": predicted_current,
        })

        out_debug = os.path.join(
            out_dir,
            f"{label}_binary_input_debug_waveform.csv",
        )
        debug_df.to_csv(out_debug, index=False)

    run_df = pd.DataFrame(run_rows)

    out_run_summary = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_binary_input_validation_by_run.csv",
    )
    run_df.to_csv(out_run_summary, index=False)

    aligned_array = np.array(aligned_currents)
    predicted_array = np.array(predicted_currents)
    u_array = np.array(u_inputs)

    mean_measured = np.nanmean(aligned_array, axis=0)
    mean_predicted = np.nanmean(predicted_array, axis=0)
    mean_u = np.nanmean(u_array, axis=0)

    mean_df = pd.DataFrame({
        "time_from_active_start_s": ALIGNED_GRID,
        "mean_measured_current_mA": mean_measured,
        "mean_predicted_current_mA": mean_predicted,
        "mean_estimated_u_t": mean_u,
        "mean_estimated_input_level_mA": I_IDLE_MODEL_MA + GAIN_MODEL_MA * mean_u,
    })

    out_mean = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_binary_input_mean_waveform.csv",
    )
    mean_df.to_csv(out_mean, index=False)

    out_plot, out_zoom_plot = save_plot_for_intensity(
        intensity=intensity,
        aligned_currents=aligned_currents,
        predicted_currents=predicted_currents,
        u_inputs=u_inputs,
        mean_measured=mean_measured,
        mean_predicted=mean_predicted,
        mean_u=mean_u,
        out_dir=out_dir,
    )

    summary_row = {
        "cpu_intensity_percent": intensity,
        "expected_duty": expected_duty,
        "n_runs": len(files),

        "mean_estimated_duty_from_current": run_df["estimated_duty_from_current"].mean(),
        "std_estimated_duty_from_current": run_df["estimated_duty_from_current"].std(ddof=1),

        "mean_raw_mae_full_active_0_to_10s_mA": run_df["raw_mae_full_active_0_to_10s_mA"].mean(),
        "std_raw_mae_full_active_0_to_10s_mA": run_df["raw_mae_full_active_0_to_10s_mA"].std(ddof=1),

        "mean_raw_mae_transient_0_to_5ms_mA": run_df["raw_mae_transient_0_to_5ms_mA"].mean(),
        "std_raw_mae_transient_0_to_5ms_mA": run_df["raw_mae_transient_0_to_5ms_mA"].std(ddof=1),

        "mean_raw_mae_steady_10ms_to_10s_mA": run_df["raw_mae_steady_10ms_to_10s_mA"].mean(),
        "std_raw_mae_steady_10ms_to_10s_mA": run_df["raw_mae_steady_10ms_to_10s_mA"].std(ddof=1),

        "mean_measured_steady_mean_current_mA": run_df["measured_steady_mean_current_mA"].mean(),
        "mean_predicted_steady_mean_current_mA": run_df["predicted_steady_mean_current_mA"].mean(),
        "mean_steady_mean_signed_error_mA": run_df["steady_mean_signed_error_mA"].mean(),
        "mean_steady_mean_abs_error_mA": run_df["steady_mean_abs_error_mA"].mean(),
    }

    all_summary_rows.append(summary_row)

    print("\nSaved:")
    print(out_run_summary)
    print(out_mean)
    print(out_plot)
    print(out_zoom_plot)

    print("\nIntensity summary:")
    print(pd.DataFrame([summary_row]))


# ============================================================
# Save combined summary
# ============================================================

summary_df = pd.DataFrame(all_summary_rows)

out_summary = os.path.join(
    BASE_OUT_DIR,
    "cpu_binary_input_validation_summary_cpu75_cpu50_cpu25.csv",
)

summary_df.to_csv(out_summary, index=False)

print("\n" + "=" * 70)
print("Combined binary-input validation summary")
print("=" * 70)
print(summary_df)

print("\nSaved:")
print(out_summary)

compact_cols = [
    "cpu_intensity_percent",
    "expected_duty",
    "mean_estimated_duty_from_current",
    "mean_raw_mae_full_active_0_to_10s_mA",
    "mean_raw_mae_steady_10ms_to_10s_mA",
    "mean_measured_steady_mean_current_mA",
    "mean_predicted_steady_mean_current_mA",
    "mean_steady_mean_signed_error_mA",
    "mean_steady_mean_abs_error_mA",
]

print("\n" + "=" * 70)
print("Compact summary")
print("=" * 70)
print(summary_df[compact_cols])