import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

CONDITIONS = [
    {
        "name": "period_100ms_10duty",
        "input_dir": "data/raw/v3_ppk/per_100ms_10duty",
        "output_dir": "data/processed/v3_ppk/per_100ms_10duty",
        "cycle_us": 100_000,
        "duty_percent": 10,
    },
    {
        "name": "period_500ms_10duty",
        "input_dir": "data/raw/v3_ppk/per_500ms_10duty",
        "output_dir": "data/processed/v3_ppk/per_500ms_10duty",
        "cycle_us": 500_000,
        "duty_percent": 10,
    },
    {
        "name": "period_1s_10duty",
        "input_dir": "data/raw/v3_ppk/per_1s_10duty",
        "output_dir": "data/processed/v3_ppk/per_1s_10duty",
        "cycle_us": 1_000_000,
        "duty_percent": 10,
    },
]

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

# Evaluation windows
FULL_EVAL_WINDOW_S = (0.0, 10.0)
TRANSIENT_WINDOW_S = (0.0, 0.005)
PERIODIC_STEADY_WINDOW_S = (0.010, 10.0)

# Smoothing for waveform MAE
SMOOTH_WINDOW_SAMPLES = 20   # 20 samples at 10 kS/s = about 2 ms

# The active rising edge is detected by current crossing a threshold,
# not by GPIO marker. For a first-order step response, midpoint crossing
# happens about tau * ln(2) after the actual input step.
USE_EDGE_PHASE_CORRECTION = True
EDGE_PHASE_CORRECTION_S = TAU_S * np.log(2.0)

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

    # Downsample: 100 kS/s -> 10 kS/s
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
    Detect the first CPU100-like busy pulse using a CPU100-derived threshold.
    This is used only for alignment.
    """

    smooth = current_mA.rolling(window=20, center=True, min_periods=1).mean()

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

    # CPU100-derived high-current threshold
    high_current_threshold = (I_IDLE_MODEL_MA + I_ACTIVE_MODEL_MA) / 2.0

    # Detect first busy pulse after initial idle
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

    active_mask = (
        (time_s > rising_time)
        & (time_s < rising_time + 20)
        & (smooth > high_current_threshold)
    )

    active_level = smooth[active_mask].median()

    # End of active phase is not critical for this validation.
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


def create_firmware_binary_input(time_grid, duty_percent, cycle_us):
    """
    Create firmware-defined binary input u(t).

    u(t) = 1 during busy section
    u(t) = 0 during wait section
    """

    cycle_s = cycle_us / 1_000_000.0
    busy_s = cycle_s * duty_percent / 100.0

    if USE_EDGE_PHASE_CORRECTION:
        phase_corrected_time = time_grid + EDGE_PHASE_CORRECTION_S
    else:
        phase_corrected_time = time_grid.copy()

    u_t = np.zeros_like(time_grid, dtype=float)

    active_mask = phase_corrected_time >= 0.0
    phase_in_cycle = np.mod(phase_corrected_time[active_mask], cycle_s)

    u_t[active_mask] = (phase_in_cycle < busy_s).astype(float)

    return u_t


def simulate_first_order_model_binary_input(time_grid, u_t):
    """
    Simulate:
        dI/dt = (I_idle + gain*u(t) - I) / tau
    """

    I_pred = np.full_like(time_grid, np.nan, dtype=float)
    I_pred[0] = I_IDLE_MODEL_MA

    for k in range(1, len(time_grid)):
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


def save_plot_for_condition(
    condition_name,
    cycle_us,
    duty_percent,
    aligned_currents,
    predicted_currents,
    firmware_inputs,
    mean_measured,
    mean_predicted,
    mean_firmware_u,
    out_dir,
):
    cycle_s = cycle_us / 1_000_000.0
    busy_s = cycle_s * duty_percent / 100.0

    # ------------------------------------------------------------
    # Full plot
    # ------------------------------------------------------------

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
    plt.xlabel("Time from first active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"{condition_name}: firmware-defined binary input ODE validation"
    )
    plt.xlim(-2, 12)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_plot = os.path.join(
        out_dir,
        f"{condition_name}_firmware_binary_input_measured_vs_predicted.png",
    )

    plt.savefig(out_plot, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # Zoom plot
    # ------------------------------------------------------------

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

    input_level = I_IDLE_MODEL_MA + GAIN_MODEL_MA * mean_firmware_u

    plt.plot(
        ALIGNED_GRID,
        input_level,
        linewidth=1.5,
        linestyle=":",
        label="firmware input target level",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from first active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"{condition_name}: switching zoom "
        f"({busy_s * 1000:.1f} ms busy / {(cycle_s - busy_s) * 1000:.1f} ms idle)"
    )

    zoom_end_s = min(max(cycle_s * 2.2, 0.25), 2.5)
    plt.xlim(-0.02, zoom_end_s)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_zoom_plot = os.path.join(
        out_dir,
        f"{condition_name}_firmware_binary_input_zoom.png",
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

for condition in CONDITIONS:
    condition_name = condition["name"]
    input_dir = condition["input_dir"]
    out_dir = condition["output_dir"]
    cycle_us = condition["cycle_us"]
    duty_percent = condition["duty_percent"]

    expected_duty = duty_percent / 100.0
    period_ms = cycle_us / 1000.0
    busy_ms = period_ms * duty_percent / 100.0
    idle_ms = period_ms - busy_ms

    data_pattern = f"{input_dir}/*.csv"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(data_pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No CSV files found: {data_pattern}")

    print("\n" + "=" * 70)
    print(f"Processing {condition_name}")
    print("=" * 70)

    aligned_currents = []
    predicted_currents = []
    firmware_inputs = []

    run_rows = []

    firmware_u_t = create_firmware_binary_input(
        ALIGNED_GRID,
        duty_percent,
        cycle_us,
    )

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

        predicted_current = simulate_first_order_model_binary_input(
            ALIGNED_GRID,
            firmware_u_t,
        )

        smoothed_measured = smooth_waveform(
            measured_interp,
            SMOOTH_WINDOW_SAMPLES,
        )

        smoothed_predicted = smooth_waveform(
            predicted_current,
            SMOOTH_WINDOW_SAMPLES,
        )

        raw_mae_full = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            FULL_EVAL_WINDOW_S,
        )

        raw_mae_transient = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            TRANSIENT_WINDOW_S,
        )

        raw_mae_periodic_steady = calculate_mae(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            PERIODIC_STEADY_WINDOW_S,
        )

        smoothed_mae_full = calculate_mae(
            ALIGNED_GRID,
            smoothed_measured,
            smoothed_predicted,
            FULL_EVAL_WINDOW_S,
        )

        smoothed_mae_periodic_steady = calculate_mae(
            ALIGNED_GRID,
            smoothed_measured,
            smoothed_predicted,
            PERIODIC_STEADY_WINDOW_S,
        )

        measured_mean, predicted_mean, mean_signed_error = calculate_mean_error(
            ALIGNED_GRID,
            measured_interp,
            predicted_current,
            PERIODIC_STEADY_WINDOW_S,
        )

        eval_mask = (
            (ALIGNED_GRID >= PERIODIC_STEADY_WINDOW_S[0])
            & (ALIGNED_GRID <= PERIODIC_STEADY_WINDOW_S[1])
            & np.isfinite(measured_interp)
        )

        effective_firmware_duty = (
            np.mean(firmware_u_t[eval_mask])
            if np.sum(eval_mask) > 0
            else np.nan
        )

        aligned_currents.append(measured_interp)
        predicted_currents.append(predicted_current)
        firmware_inputs.append(firmware_u_t)

        run_rows.append({
            "run": label,
            "condition": condition_name,
            "duty_percent": duty_percent,
            "expected_duty": expected_duty,
            "cycle_us": cycle_us,
            "period_ms": period_ms,
            "busy_ms": busy_ms,
            "idle_ms": idle_ms,
            "effective_firmware_duty_in_eval_window": effective_firmware_duty,

            "power_on_time_s": power_on_time,
            "active_start_time_s": rising_time,
            "active_end_time_s": falling_time,

            "alignment_idle_level_mA": idle_level,
            "alignment_active_level_mA": active_level,
            "alignment_threshold_mA": alignment_threshold,

            "use_edge_phase_correction": USE_EDGE_PHASE_CORRECTION,
            "edge_phase_correction_s": EDGE_PHASE_CORRECTION_S,

            "raw_mae_full_0_to_10s_mA": raw_mae_full,
            "raw_mae_transient_0_to_5ms_mA": raw_mae_transient,
            "raw_mae_periodic_steady_10ms_to_10s_mA": raw_mae_periodic_steady,

            "smoothed_mae_full_0_to_10s_mA": smoothed_mae_full,
            "smoothed_mae_periodic_steady_10ms_to_10s_mA": smoothed_mae_periodic_steady,

            "measured_periodic_steady_mean_current_mA": measured_mean,
            "predicted_periodic_steady_mean_current_mA": predicted_mean,
            "periodic_steady_mean_signed_error_mA": mean_signed_error,
            "periodic_steady_mean_abs_error_mA": abs(mean_signed_error),
        })

        debug_df = pd.DataFrame({
            "time_from_first_active_start_s": ALIGNED_GRID,
            "measured_current_mA": measured_interp,
            "smoothed_measured_current_mA": smoothed_measured,
            "firmware_u_t": firmware_u_t,
            "firmware_input_target_current_mA": I_IDLE_MODEL_MA + GAIN_MODEL_MA * firmware_u_t,
            "predicted_current_mA": predicted_current,
            "smoothed_predicted_current_mA": smoothed_predicted,
            "prediction_error_mA": measured_interp - predicted_current,
            "smoothed_prediction_error_mA": smoothed_measured - smoothed_predicted,
        })

        out_debug = os.path.join(
            out_dir,
            f"{label}_firmware_binary_input_debug_waveform.csv",
        )
        debug_df.to_csv(out_debug, index=False)

    run_df = pd.DataFrame(run_rows)

    out_run_summary = os.path.join(
        out_dir,
        f"{condition_name}_firmware_binary_input_validation_by_run.csv",
    )
    run_df.to_csv(out_run_summary, index=False)

    aligned_array = np.array(aligned_currents)
    predicted_array = np.array(predicted_currents)
    firmware_u_array = np.array(firmware_inputs)

    mean_measured = np.nanmean(aligned_array, axis=0)
    mean_predicted = np.nanmean(predicted_array, axis=0)
    mean_firmware_u = np.nanmean(firmware_u_array, axis=0)

    mean_df = pd.DataFrame({
        "time_from_first_active_start_s": ALIGNED_GRID,
        "mean_measured_current_mA": mean_measured,
        "mean_predicted_current_mA": mean_predicted,
        "mean_firmware_u_t": mean_firmware_u,
        "mean_firmware_input_target_current_mA": I_IDLE_MODEL_MA + GAIN_MODEL_MA * mean_firmware_u,
        "mean_prediction_error_mA": mean_measured - mean_predicted,
    })

    out_mean = os.path.join(
        out_dir,
        f"{condition_name}_firmware_binary_input_mean_waveform.csv",
    )
    mean_df.to_csv(out_mean, index=False)

    out_plot, out_zoom_plot = save_plot_for_condition(
        condition_name=condition_name,
        cycle_us=cycle_us,
        duty_percent=duty_percent,
        aligned_currents=aligned_currents,
        predicted_currents=predicted_currents,
        firmware_inputs=firmware_inputs,
        mean_measured=mean_measured,
        mean_predicted=mean_predicted,
        mean_firmware_u=mean_firmware_u,
        out_dir=out_dir,
    )

    summary_row = {
        "condition": condition_name,
        "duty_percent": duty_percent,
        "expected_duty": expected_duty,
        "cycle_us": cycle_us,
        "period_ms": period_ms,
        "busy_ms": busy_ms,
        "idle_ms": idle_ms,
        "n_runs": len(files),

        "effective_firmware_duty_in_eval_window_mean": run_df["effective_firmware_duty_in_eval_window"].mean(),
        "effective_firmware_duty_in_eval_window_std": run_df["effective_firmware_duty_in_eval_window"].std(ddof=1),

        "mean_raw_mae_full_0_to_10s_mA": run_df["raw_mae_full_0_to_10s_mA"].mean(),
        "std_raw_mae_full_0_to_10s_mA": run_df["raw_mae_full_0_to_10s_mA"].std(ddof=1),

        "mean_raw_mae_periodic_steady_10ms_to_10s_mA": run_df["raw_mae_periodic_steady_10ms_to_10s_mA"].mean(),
        "std_raw_mae_periodic_steady_10ms_to_10s_mA": run_df["raw_mae_periodic_steady_10ms_to_10s_mA"].std(ddof=1),

        "mean_smoothed_mae_full_0_to_10s_mA": run_df["smoothed_mae_full_0_to_10s_mA"].mean(),
        "std_smoothed_mae_full_0_to_10s_mA": run_df["smoothed_mae_full_0_to_10s_mA"].std(ddof=1),

        "mean_smoothed_mae_periodic_steady_10ms_to_10s_mA": run_df["smoothed_mae_periodic_steady_10ms_to_10s_mA"].mean(),
        "std_smoothed_mae_periodic_steady_10ms_to_10s_mA": run_df["smoothed_mae_periodic_steady_10ms_to_10s_mA"].std(ddof=1),

        "mean_measured_periodic_steady_mean_current_mA": run_df["measured_periodic_steady_mean_current_mA"].mean(),
        "mean_predicted_periodic_steady_mean_current_mA": run_df["predicted_periodic_steady_mean_current_mA"].mean(),
        "mean_periodic_steady_mean_signed_error_mA": run_df["periodic_steady_mean_signed_error_mA"].mean(),
        "mean_periodic_steady_mean_abs_error_mA": run_df["periodic_steady_mean_abs_error_mA"].mean(),
    }

    all_summary_rows.append(summary_row)

    print("\nSaved:")
    print(out_run_summary)
    print(out_mean)
    print(out_plot)
    print(out_zoom_plot)

    print("\nCondition summary:")
    print(pd.DataFrame([summary_row]))


# ============================================================
# Save combined summary
# ============================================================

summary_df = pd.DataFrame(all_summary_rows)

out_summary = os.path.join(
    BASE_OUT_DIR,
    "period_10duty_firmware_binary_input_validation_summary.csv",
)

summary_df.to_csv(out_summary, index=False)

print("\n" + "=" * 70)
print("Combined period 10% duty firmware-defined binary-input validation summary")
print("=" * 70)
print(summary_df)

print("\nSaved:")
print(out_summary)

compact_cols = [
    "condition",
    "duty_percent",
    "period_ms",
    "busy_ms",
    "idle_ms",
    "effective_firmware_duty_in_eval_window_mean",
    "mean_raw_mae_full_0_to_10s_mA",
    "mean_raw_mae_periodic_steady_10ms_to_10s_mA",
    "mean_smoothed_mae_full_0_to_10s_mA",
    "mean_smoothed_mae_periodic_steady_10ms_to_10s_mA",
    "mean_measured_periodic_steady_mean_current_mA",
    "mean_predicted_periodic_steady_mean_current_mA",
    "mean_periodic_steady_mean_signed_error_mA",
    "mean_periodic_steady_mean_abs_error_mA",
]

print("\n" + "=" * 70)
print("Compact summary")
print("=" * 70)
print(summary_df[compact_cols])