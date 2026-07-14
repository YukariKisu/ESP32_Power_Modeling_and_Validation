import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Settings
# ============================================================

BASE_RAW_DIR = Path("data/raw/v3_ppk")
BASE_OUT_DIR = Path(
    "data/processed/v3_ppk/core_only_workload_validation"
)

BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Arithmetic-trained first-order ODE model
# ------------------------------------------------------------

I_IDLE_MODEL_MA = 47.2668
GAIN_MODEL_MA = 20.1888
TAU_S = 0.00049
# Current-based alignment detects the 50% crossing.
# Shift the model pulse left so its 50% crossing is at t = 0.
MODEL_T50_SHIFT_S = TAU_S * np.log(2.0)


# ------------------------------------------------------------
# Workload datasets
# ------------------------------------------------------------

WORKLOADS = {
    "arithmetic": {
        "label": "Arithmetic loop",
        "pattern": (
            "data/raw/v3_ppk/cpu_100/"
            "ppk_cpu_100_run*.csv"
        ),
    },
    "float": {
        "label": "Floating point",
        "pattern": (
            "data/raw/v3_ppk/cpu_maximum/"
            "cpu_100_float/*.csv"
        ),
    },
    "seq_ram": {
        "label": "Sequential RAM read/write",
        "pattern": (
            "data/raw/v3_ppk/cpu_maximum/"
            "cpu_100_seqRAM/*.csv"
        ),
    },
    "large_ram_copy": {
        "label": "Large-buffer RAM copy",
        "pattern": (
            "data/raw/v3_ppk/cpu_maximum/"
            "cpu_100_largeRAMcopy/*.csv"
        ),
    },
    "mixed_ram_arithmetic": {
        "label": "Mixed RAM + arithmetic",
        "pattern": (
            "data/raw/v3_ppk/cpu_maximum/"
            "cpu_100_mixRAMarith/*.csv"
        ),
    },
    "memory_integer_float_bit": {
        "label": "Memory + integer + floating point + bit operations",
        "pattern": (
            "data/raw/v3_ppk/cpu_maximum/"
            "cpu_100_4comb/*.csv"
        ),
    },
}


# ------------------------------------------------------------
# Sampling and alignment
# ------------------------------------------------------------

# Raw PPK2 data: approximately 100 kS/s
# Downsampled waveform: approximately 10 kS/s
DOWNSAMPLE_STEP = 10

GRID_DT_S = 0.0001

# Relative to detected active start:
# -10 to 0 s  : initial idle
#   0 to 20 s : active
#  20 to 30 s : final idle
ALIGNED_GRID = np.arange(
    -10.0,
    30.0,
    GRID_DT_S,
)


# ------------------------------------------------------------
# Evaluation windows
# ------------------------------------------------------------

TRANSIENT_WINDOW_S = (0.0, 0.005)
STEADY_ACTIVE_WINDOW_S = (0.010, 19.0)
FULL_ACTIVE_WINDOW_S = (0.0, 20.0)

INITIAL_IDLE_WINDOW_S = (-9.0, -1.0)
FINAL_IDLE_WINDOW_S = (21.0, 29.0)

# Whole aligned experiment, excluding extreme edges
FULL_EXPERIMENT_WINDOW_S = (-9.0, 29.0)


# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------

SMOOTH_WINDOW_S = 0.010

SMOOTH_WINDOW_SAMPLES = int(
    SMOOTH_WINDOW_S / GRID_DT_S
)

if SMOOTH_WINDOW_SAMPLES % 2 == 0:
    SMOOTH_WINDOW_SAMPLES += 1


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

SHOW_PLOTS = False


# ------------------------------------------------------------
# Error convention
# ------------------------------------------------------------

# error = prediction - measurement
#
# positive mean error:
#     model overestimates current
#
# negative mean error:
#     model underestimates current


# ============================================================
# Utility functions
# ============================================================

def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def detect_columns(csv_path):
    columns = pd.read_csv(
        csv_path,
        nrows=0,
    ).columns

    timestamp_col = None
    current_col = None

    for column in columns:
        name = column.lower()

        if timestamp_col is None and (
            "timestamp" in name
            or "time" in name
        ):
            timestamp_col = column

        if current_col is None and "current" in name:
            current_col = column

    if timestamp_col is None or current_col is None:
        raise ValueError(
            f"Could not detect timestamp/current columns "
            f"in {csv_path}. "
            f"Columns: {list(columns)}"
        )

    return timestamp_col, current_col


def load_ppk2_csv(csv_path):
    timestamp_col, current_col = detect_columns(
        csv_path
    )

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

    valid_mask = (
        time_raw.notna()
        & current_raw.notna()
    )

    time_raw = (
        time_raw[valid_mask]
        .reset_index(drop=True)
    )

    current_raw = (
        current_raw[valid_mask]
        .reset_index(drop=True)
    )

    if len(time_raw) == 0:
        raise ValueError(
            f"No valid samples found in {csv_path}"
        )

    # PPK2 timestamp: ms -> s
    time_s = (
        time_raw - time_raw.iloc[0]
    ) / 1000.0

    # PPK2 current: uA -> mA
    current_mA = current_raw / 1000.0

    time_s = (
        time_s.iloc[::DOWNSAMPLE_STEP]
        .reset_index(drop=True)
    )

    current_mA = (
        current_mA.iloc[::DOWNSAMPLE_STEP]
        .reset_index(drop=True)
    )

    return time_s, current_mA


# ============================================================
# Current-based time alignment
# ============================================================

def detect_active_edges(time_s, current_mA):
    """
    Detect the active rising and falling edges from current.

    The expected experiment timing is:

        approximately 10 s idle
        approximately 20 s active
        approximately 10 s idle
    """

    smooth = current_mA.rolling(
        window=200,
        center=True,
        min_periods=1,
    ).mean()

    max_level = smooth.quantile(0.99)
    power_threshold = max_level * 0.1

    power_candidates = np.where(
        smooth > power_threshold
    )[0]

    if len(power_candidates) == 0:
        raise ValueError(
            "Could not detect power-on"
        )

    power_on_time = float(
        time_s.iloc[power_candidates[0]]
    )

    idle_mask = (
        (time_s > power_on_time + 3.0)
        & (time_s < power_on_time + 9.0)
    )

    active_mask = (
        (time_s > power_on_time + 13.0)
        & (time_s < power_on_time + 25.0)
    )

    idle_level = float(
        smooth[idle_mask].median()
    )

    active_level = float(
        smooth[active_mask].median()
    )

    if (
        np.isnan(idle_level)
        or np.isnan(active_level)
    ):
        raise ValueError(
            "Could not estimate idle or active level"
        )

    delta = active_level - idle_level

    if delta <= 0:
        raise ValueError(
            "Active level is not higher than idle level. "
            f"idle={idle_level:.3f} mA, "
            f"active={active_level:.3f} mA"
        )

    threshold = (
        idle_level + active_level
    ) / 2.0

    rising_candidates = np.where(
        (time_s > power_on_time + 5.0)
        & (smooth > threshold)
    )[0]

    if len(rising_candidates) == 0:
        raise ValueError(
            "Could not detect active rising edge"
        )

    rising_idx = int(
        rising_candidates[0]
    )

    rising_time = float(
        time_s.iloc[rising_idx]
    )

    falling_candidates = np.where(
        (time_s > rising_time + 10.0)
        & (smooth < threshold)
    )[0]

    if len(falling_candidates) == 0:
        falling_time = np.nan
    else:
        falling_idx = int(
            falling_candidates[0]
        )

        falling_time = float(
            time_s.iloc[falling_idx]
        )

    return {
        "power_on_time_s": power_on_time,
        "rising_time_s": rising_time,
        "falling_time_s": falling_time,
        "idle_level_mA": idle_level,
        "active_level_mA": active_level,
        "threshold_mA": threshold,
    }


# ============================================================
# ODE model
# ============================================================

def model_prediction_current_mA(
    time_from_active_start_s,
    idle_current_mA,
):
    """
    First-order ODE prediction for a 20-second active pulse.

    Measurement alignment:
        t = 0 corresponds to the measured 50% rising-edge crossing.

    Therefore, the model input pulse is shifted left by tau * ln(2),
    so the model's 50% crossing also occurs at t = 0.
    """

    time_values = np.asarray(
        time_from_active_start_s,
        dtype=float,
    )

    predicted = np.full_like(
        time_values,
        fill_value=idle_current_mA,
        dtype=float,
    )

    active_target_mA = (
        idle_current_mA + GAIN_MODEL_MA
    )

    # First-order response reaches 50% at tau * ln(2)
    t50_shift_s = MODEL_T50_SHIFT_S

    # Shift the whole 20-second pulse to the left.
    model_active_start_s = -t50_shift_s
    model_active_end_s = 20.0 - t50_shift_s

    # Rising response
    rising_mask = (
        (time_values >= model_active_start_s)
        & (time_values < model_active_end_s)
    )

    rising_elapsed_s = (
        time_values[rising_mask]
        - model_active_start_s
    )

    predicted[rising_mask] = (
        active_target_mA
        - GAIN_MODEL_MA
        * np.exp(
            -rising_elapsed_s / TAU_S
        )
    )

    # Current immediately before the active pulse ends
    active_duration_s = 20.0

    active_end_current_mA = (
        active_target_mA
        - GAIN_MODEL_MA
        * np.exp(
            -active_duration_s / TAU_S
        )
    )

    # Falling response
    falling_mask = (
        time_values >= model_active_end_s
    )

    falling_elapsed_s = (
        time_values[falling_mask]
        - model_active_end_s
    )

    predicted[falling_mask] = (
        idle_current_mA
        + (
            active_end_current_mA
            - idle_current_mA
        )
        * np.exp(
            -falling_elapsed_s / TAU_S
        )
    )

    return predicted


# ============================================================
# Measurement processing
# ============================================================

def smooth_waveform(
    values,
    window_samples,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    original_nan_mask = ~np.isfinite(
        values
    )

    smoothed = (
        pd.Series(values)
        .rolling(
            window=window_samples,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
        .copy()
    )

    smoothed[original_nan_mask] = np.nan

    return smoothed


def calculate_window_mean(
    time_s,
    values,
    window,
):
    start_s, end_s = window

    mask = (
        (time_s >= start_s)
        & (time_s <= end_s)
        & np.isfinite(values)
    )

    if np.sum(mask) == 0:
        return np.nan

    return float(
        np.mean(values[mask])
    )


def calculate_idle_baseline(
    time_s,
    measured_mA,
):
    initial_idle_mean = calculate_window_mean(
        time_s,
        measured_mA,
        INITIAL_IDLE_WINDOW_S,
    )

    final_idle_mean = calculate_window_mean(
        time_s,
        measured_mA,
        FINAL_IDLE_WINDOW_S,
    )

    if (
        np.isnan(initial_idle_mean)
        and np.isnan(final_idle_mean)
    ):
        return np.nan, np.nan, np.nan

    if np.isnan(final_idle_mean):
        idle_baseline = initial_idle_mean
    elif np.isnan(initial_idle_mean):
        idle_baseline = final_idle_mean
    else:
        idle_baseline = (
            initial_idle_mean
            + final_idle_mean
        ) / 2.0

    return (
        initial_idle_mean,
        final_idle_mean,
        idle_baseline,
    )


# ============================================================
# Error metrics
# ============================================================

def calculate_error_metrics(
    time_s,
    measured_mA,
    predicted_mA,
    window,
):
    start_s, end_s = window

    mask = (
        (time_s >= start_s)
        & (time_s <= end_s)
        & np.isfinite(measured_mA)
        & np.isfinite(predicted_mA)
    )

    if np.sum(mask) == 0:
        return {
            "sample_count": 0,
            "mae_mA": np.nan,
            "mean_error_mA": np.nan,
            "error_std_mA": np.nan,
            "rmse_mA": np.nan,
            "error_min_mA": np.nan,
            "error_max_mA": np.nan,
        }

    error = (
        predicted_mA[mask]
        - measured_mA[mask]
    )

    return {
        "sample_count": int(len(error)),
        "mae_mA": float(
            np.mean(np.abs(error))
        ),
        "mean_error_mA": float(
            np.mean(error)
        ),
        "error_std_mA": float(
            np.std(error, ddof=1)
        ),
        "rmse_mA": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "error_min_mA": float(
            np.min(error)
        ),
        "error_max_mA": float(
            np.max(error)
        ),
    }


def add_metrics_to_row(
    row,
    prefix,
    metrics,
):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


# ============================================================
# Plotting
# ============================================================

def save_validation_plot(
    workload_label,
    workload_key,
    aligned_currents,
    mean_current,
    smoothed_mean_current,
    std_current,
    fixed_prediction,
    adjusted_prediction,
    out_dir,
):
    plt.figure(figsize=(12, 6))

    for index, current in enumerate(
        aligned_currents
    ):
        plt.plot(
            ALIGNED_GRID,
            current,
            alpha=0.12,
            linewidth=0.5,
            label=(
                "Individual aligned runs"
                if index == 0
                else None
            ),
        )

    plt.plot(
        ALIGNED_GRID,
        mean_current,
        linewidth=2.2,
        label="Measured mean waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        smoothed_mean_current,
        linewidth=1.8,
        linestyle="-.",
        label="Smoothed measured mean",
    )

    plt.plot(
        ALIGNED_GRID,
        adjusted_prediction,
        linewidth=2.2,
        linestyle="--",
        label="Baseline-adjusted prediction",
    )

    plt.plot(
        ALIGNED_GRID,
        fixed_prediction,
        linewidth=1.5,
        linestyle=":",
        label="Fixed absolute model prediction",
    )

    plt.fill_between(
        ALIGNED_GRID,
        mean_current - std_current,
        mean_current + std_current,
        alpha=0.20,
        label="Measured mean ± 1 std",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        20.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Time from active start [s]"
    )

    plt.ylabel(
        "Current [mA]"
    )

    plt.title(
        f"{workload_label}: "
        "aligned measured waveform vs ODE prediction"
    )

    plt.xlim(-5.0, 25.0)
    plt.grid(True)
    plt.legend(
        fontsize=8,
        loc="upper right",
    )
    plt.tight_layout()

    out_plot = (
        out_dir
        / f"{workload_key}_measured_vs_predicted.png"
    )

    plt.savefig(
        out_plot,
        dpi=200,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return out_plot


def save_transient_zoom_plot(
    workload_label,
    workload_key,
    mean_current,
    smoothed_mean_current,
    std_current,
    adjusted_prediction,
    out_dir,
):
    plt.figure(figsize=(12, 6))

    plt.plot(
        ALIGNED_GRID,
        mean_current,
        linewidth=2.2,
        label="Measured mean waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        smoothed_mean_current,
        linewidth=1.8,
        linestyle="-.",
        label="Smoothed measured mean",
    )

    plt.plot(
        ALIGNED_GRID,
        adjusted_prediction,
        linewidth=2.2,
        linestyle="--",
        label="Baseline-adjusted prediction",
    )

    plt.fill_between(
        ALIGNED_GRID,
        mean_current - std_current,
        mean_current + std_current,
        alpha=0.20,
        label="Measured mean ± 1 std",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Time from active start [s]"
    )

    plt.ylabel(
        "Current [mA]"
    )

    plt.title(
        f"{workload_label}: rising transient zoom"
    )

    plt.xlim(-0.005, 0.030)
    plt.grid(True)
    plt.legend(
        fontsize=8,
        loc="upper right",
    )
    plt.tight_layout()

    out_plot = (
        out_dir
        / f"{workload_key}_transient_zoom.png"
    )

    plt.savefig(
        out_plot,
        dpi=200,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return out_plot


def save_error_histogram(
    workload_label,
    workload_key,
    measured_current,
    predicted_current,
    out_dir,
):
    start_s, end_s = (
        STEADY_ACTIVE_WINDOW_S
    )

    mask = (
        (ALIGNED_GRID >= start_s)
        & (ALIGNED_GRID <= end_s)
        & np.isfinite(measured_current)
        & np.isfinite(predicted_current)
    )

    error = (
        predicted_current[mask]
        - measured_current[mask]
    )

    if len(error) == 0:
        return None

    mean_error = np.mean(error)
    error_std = np.std(
        error,
        ddof=1,
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        error,
        bins=80,
        alpha=0.85,
    )

    plt.axvline(
        mean_error,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Mean error = "
            f"{mean_error:.3f} mA"
        ),
    )

    plt.xlabel(
        "Prediction − measurement [mA]"
    )

    plt.ylabel(
        "Sample count"
    )

    plt.title(
        f"{workload_label}: "
        "steady-active error distribution"
    )

    plt.grid(axis="y")

    plt.legend(
        title=(
            f"Error std = "
            f"{error_std:.3f} mA"
        )
    )

    plt.tight_layout()

    out_plot = (
        out_dir
        / f"{workload_key}_steady_active_error_histogram.png"
    )

    plt.savefig(
        out_plot,
        dpi=200,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return out_plot


# ============================================================
# Main analysis
# ============================================================

def main():
    all_workload_summary_rows = []

    for workload_key, config in WORKLOADS.items():
        workload_label = config["label"]
        data_pattern = config["pattern"]

        files = sorted(
            glob.glob(data_pattern),
            key=natural_key,
        )

        if not files:
            print()
            print(
                f"Warning: no CSV files found for "
                f"{workload_label}"
            )
            print(
                f"Pattern: {data_pattern}"
            )
            continue

        out_dir = (
            BASE_OUT_DIR / workload_key
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 72)
        print(
            f"Processing: {workload_label}"
        )
        print(
            f"Run count: {len(files)}"
        )
        print("=" * 72)

        aligned_currents = []
        run_summary_rows = []
        run_metric_rows = []

        # --------------------------------------------------------
        # Process individual runs
        # --------------------------------------------------------

        for csv_path in files:
            run_label = (
                os.path.basename(csv_path)
                .replace(".csv", "")
            )

            print(
                f"Processing: {run_label}"
            )

            time_s, current_mA = load_ppk2_csv(
                csv_path
            )

            edge_info = detect_active_edges(
                time_s,
                current_mA,
            )

            aligned_time = (
                time_s
                - edge_info["rising_time_s"]
            )

            interpolated_current = np.interp(
                ALIGNED_GRID,
                aligned_time,
                current_mA,
                left=np.nan,
                right=np.nan,
            )

            aligned_currents.append(
                interpolated_current
            )

            (
                initial_idle_mean,
                final_idle_mean,
                idle_baseline_mean,
            ) = calculate_idle_baseline(
                ALIGNED_GRID,
                interpolated_current,
            )

            fixed_prediction = (
                model_prediction_current_mA(
                    ALIGNED_GRID,
                    I_IDLE_MODEL_MA,
                )
            )

            adjusted_prediction = (
                model_prediction_current_mA(
                    ALIGNED_GRID,
                    idle_baseline_mean,
                )
            )

            smoothed_current = smooth_waveform(
                interpolated_current,
                SMOOTH_WINDOW_SAMPLES,
            )

            falling_time = edge_info[
                "falling_time_s"
            ]

            if np.isnan(falling_time):
                active_duration_s = np.nan
            else:
                active_duration_s = (
                    falling_time
                    - edge_info["rising_time_s"]
                )

            active_mean_mA = calculate_window_mean(
                ALIGNED_GRID,
                interpolated_current,
                STEADY_ACTIVE_WINDOW_S,
            )

            measured_delta_mA = (
                active_mean_mA
                - idle_baseline_mean
            )

            # ----------------------------------------------------
            # Per-run summary
            # ----------------------------------------------------

            run_summary_rows.append({
                "workload": workload_label,
                "run": run_label,
                "file": csv_path,

                "power_on_time_s": (
                    edge_info["power_on_time_s"]
                ),

                "active_start_time_s": (
                    edge_info["rising_time_s"]
                ),

                "active_end_time_s": falling_time,
                "active_duration_s": active_duration_s,

                "detected_idle_level_mA": (
                    edge_info["idle_level_mA"]
                ),

                "detected_active_level_mA": (
                    edge_info["active_level_mA"]
                ),

                "initial_idle_mean_mA": (
                    initial_idle_mean
                ),

                "final_idle_mean_mA": (
                    final_idle_mean
                ),

                "idle_baseline_mean_mA": (
                    idle_baseline_mean
                ),

                "steady_active_mean_mA": (
                    active_mean_mA
                ),

                "measured_delta_mA": (
                    measured_delta_mA
                ),

                "model_gain_mA": GAIN_MODEL_MA,
                "model_tau_s": TAU_S,
            })

            metric_row = {
                "workload": workload_label,
                "run": run_label,
                "idle_baseline_mean_mA": (
                    idle_baseline_mean
                ),
                "steady_active_mean_mA": (
                    active_mean_mA
                ),
                "measured_delta_mA": (
                    measured_delta_mA
                ),
                "model_gain_mA": GAIN_MODEL_MA,
            }

            # ----------------------------------------------------
            # Baseline-adjusted metrics
            # ----------------------------------------------------

            for window_name, window in {
                "adjusted_full_experiment":
                    FULL_EXPERIMENT_WINDOW_S,

                "adjusted_full_active":
                    FULL_ACTIVE_WINDOW_S,

                "adjusted_transient":
                    TRANSIENT_WINDOW_S,

                "adjusted_steady_active":
                    STEADY_ACTIVE_WINDOW_S,
            }.items():
                metrics = calculate_error_metrics(
                    ALIGNED_GRID,
                    interpolated_current,
                    adjusted_prediction,
                    window,
                )

                add_metrics_to_row(
                    metric_row,
                    window_name,
                    metrics,
                )

            # ----------------------------------------------------
            # Smoothed baseline-adjusted metrics
            # ----------------------------------------------------

            for window_name, window in {
                "smoothed_adjusted_full_active":
                    FULL_ACTIVE_WINDOW_S,

                "smoothed_adjusted_transient":
                    TRANSIENT_WINDOW_S,

                "smoothed_adjusted_steady_active":
                    STEADY_ACTIVE_WINDOW_S,
            }.items():
                metrics = calculate_error_metrics(
                    ALIGNED_GRID,
                    smoothed_current,
                    adjusted_prediction,
                    window,
                )

                add_metrics_to_row(
                    metric_row,
                    window_name,
                    metrics,
                )

            # ----------------------------------------------------
            # Fixed absolute-model metrics
            # ----------------------------------------------------

            for window_name, window in {
                "fixed_full_experiment":
                    FULL_EXPERIMENT_WINDOW_S,

                "fixed_full_active":
                    FULL_ACTIVE_WINDOW_S,

                "fixed_steady_active":
                    STEADY_ACTIVE_WINDOW_S,
            }.items():
                metrics = calculate_error_metrics(
                    ALIGNED_GRID,
                    interpolated_current,
                    fixed_prediction,
                    window,
                )

                add_metrics_to_row(
                    metric_row,
                    window_name,
                    metrics,
                )

            run_metric_rows.append(
                metric_row
            )

        # --------------------------------------------------------
        # Mean waveform
        # --------------------------------------------------------

        aligned_array = np.asarray(
            aligned_currents,
            dtype=float,
        )

        mean_current = np.nanmean(
            aligned_array,
            axis=0,
        )

        std_current = np.nanstd(
            aligned_array,
            axis=0,
            ddof=1,
        )

        smoothed_mean_current = smooth_waveform(
            mean_current,
            SMOOTH_WINDOW_SAMPLES,
        )

        (
            mean_initial_idle,
            mean_final_idle,
            mean_idle_baseline,
        ) = calculate_idle_baseline(
            ALIGNED_GRID,
            mean_current,
        )

        fixed_prediction = (
            model_prediction_current_mA(
                ALIGNED_GRID,
                I_IDLE_MODEL_MA,
            )
        )

        adjusted_prediction = (
            model_prediction_current_mA(
                ALIGNED_GRID,
                mean_idle_baseline,
            )
        )

        mean_active_current = calculate_window_mean(
            ALIGNED_GRID,
            mean_current,
            STEADY_ACTIVE_WINDOW_S,
        )

        measured_mean_delta = (
            mean_active_current
            - mean_idle_baseline
        )

        # --------------------------------------------------------
        # Mean-waveform metrics
        # --------------------------------------------------------

        workload_summary = {
            "workload_key": workload_key,
            "workload": workload_label,
            "run_count": len(files),

            "model_idle_current_mA": (
                I_IDLE_MODEL_MA
            ),

            "model_gain_mA": GAIN_MODEL_MA,
            "model_tau_s": TAU_S,

            "measured_mean_initial_idle_mA": (
                mean_initial_idle
            ),

            "measured_mean_final_idle_mA": (
                mean_final_idle
            ),

            "measured_mean_idle_baseline_mA": (
                mean_idle_baseline
            ),

            "measured_mean_steady_active_mA": (
                mean_active_current
            ),

            "measured_mean_delta_mA": (
                measured_mean_delta
            ),

            "delta_error_model_minus_measured_mA": (
                GAIN_MODEL_MA
                - measured_mean_delta
            ),
        }

        for window_name, window in {
            "mean_adjusted_full_experiment":
                FULL_EXPERIMENT_WINDOW_S,

            "mean_adjusted_full_active":
                FULL_ACTIVE_WINDOW_S,

            "mean_adjusted_transient":
                TRANSIENT_WINDOW_S,

            "mean_adjusted_steady_active":
                STEADY_ACTIVE_WINDOW_S,
        }.items():
            metrics = calculate_error_metrics(
                ALIGNED_GRID,
                mean_current,
                adjusted_prediction,
                window,
            )

            add_metrics_to_row(
                workload_summary,
                window_name,
                metrics,
            )

        for window_name, window in {
            "smoothed_mean_adjusted_full_active":
                FULL_ACTIVE_WINDOW_S,

            "smoothed_mean_adjusted_transient":
                TRANSIENT_WINDOW_S,

            "smoothed_mean_adjusted_steady_active":
                STEADY_ACTIVE_WINDOW_S,
        }.items():
            metrics = calculate_error_metrics(
                ALIGNED_GRID,
                smoothed_mean_current,
                adjusted_prediction,
                window,
            )

            add_metrics_to_row(
                workload_summary,
                window_name,
                metrics,
            )

        for window_name, window in {
            "mean_fixed_full_experiment":
                FULL_EXPERIMENT_WINDOW_S,

            "mean_fixed_full_active":
                FULL_ACTIVE_WINDOW_S,

            "mean_fixed_steady_active":
                STEADY_ACTIVE_WINDOW_S,
        }.items():
            metrics = calculate_error_metrics(
                ALIGNED_GRID,
                mean_current,
                fixed_prediction,
                window,
            )

            add_metrics_to_row(
                workload_summary,
                window_name,
                metrics,
            )

        # --------------------------------------------------------
        # Run-level aggregate values
        # --------------------------------------------------------

        run_metric_df = pd.DataFrame(
            run_metric_rows
        )

        for column in [
            "adjusted_full_active_mae_mA",
            "adjusted_full_active_mean_error_mA",
            "adjusted_full_active_error_std_mA",

            "adjusted_steady_active_mae_mA",
            "adjusted_steady_active_mean_error_mA",
            "adjusted_steady_active_error_std_mA",

            "smoothed_adjusted_steady_active_mae_mA",
            "smoothed_adjusted_steady_active_mean_error_mA",
            "smoothed_adjusted_steady_active_error_std_mA",

            "fixed_steady_active_mae_mA",
            "fixed_steady_active_mean_error_mA",
            "fixed_steady_active_error_std_mA",
        ]:
            workload_summary[
                f"run_mean_{column}"
            ] = run_metric_df[column].mean()

            workload_summary[
                f"run_std_{column}"
            ] = run_metric_df[column].std(
                ddof=1
            )

        all_workload_summary_rows.append(
            workload_summary
        )

        # --------------------------------------------------------
        # Save plots
        # --------------------------------------------------------

        out_waveform_plot = save_validation_plot(
            workload_label=workload_label,
            workload_key=workload_key,
            aligned_currents=aligned_currents,
            mean_current=mean_current,
            smoothed_mean_current=smoothed_mean_current,
            std_current=std_current,
            fixed_prediction=fixed_prediction,
            adjusted_prediction=adjusted_prediction,
            out_dir=out_dir,
        )

        out_transient_plot = (
            save_transient_zoom_plot(
                workload_label=workload_label,
                workload_key=workload_key,
                mean_current=mean_current,
                smoothed_mean_current=(
                    smoothed_mean_current
                ),
                std_current=std_current,
                adjusted_prediction=(
                    adjusted_prediction
                ),
                out_dir=out_dir,
            )
        )

        out_histogram = save_error_histogram(
            workload_label=workload_label,
            workload_key=workload_key,
            measured_current=mean_current,
            predicted_current=adjusted_prediction,
            out_dir=out_dir,
        )

        # --------------------------------------------------------
        # Save CSV files
        # --------------------------------------------------------

        run_summary_df = pd.DataFrame(
            run_summary_rows
        )

        out_run_summary = (
            out_dir
            / f"{workload_key}_run_summary.csv"
        )

        run_summary_df.to_csv(
            out_run_summary,
            index=False,
        )

        out_run_metrics = (
            out_dir
            / f"{workload_key}_validation_metrics_by_run.csv"
        )

        run_metric_df.to_csv(
            out_run_metrics,
            index=False,
        )

        mean_waveform_df = pd.DataFrame({
            "time_from_active_start_s":
                ALIGNED_GRID,

            "mean_measured_current_mA":
                mean_current,

            "smoothed_mean_measured_current_mA":
                smoothed_mean_current,

            "measured_std_current_mA":
                std_current,

            "baseline_adjusted_prediction_mA":
                adjusted_prediction,

            "fixed_model_prediction_mA":
                fixed_prediction,

            "baseline_adjusted_error_mA":
                adjusted_prediction
                - mean_current,

            "fixed_model_error_mA":
                fixed_prediction
                - mean_current,
        })

        out_mean_waveform = (
            out_dir
            / f"{workload_key}_aligned_mean_with_prediction.csv"
        )

        mean_waveform_df.to_csv(
            out_mean_waveform,
            index=False,
        )

        print()
        print("Saved:")
        print(out_waveform_plot)
        print(out_transient_plot)

        if out_histogram is not None:
            print(out_histogram)

        print(out_run_summary)
        print(out_run_metrics)
        print(out_mean_waveform)

        print()
        print("Mean-waveform result:")
        print(
            f"Idle baseline: "
            f"{mean_idle_baseline:.3f} mA"
        )
        print(
            f"Active mean: "
            f"{mean_active_current:.3f} mA"
        )
        print(
            f"Measured delta: "
            f"{measured_mean_delta:.3f} mA"
        )
        print(
            f"Model delta: "
            f"{GAIN_MODEL_MA:.3f} mA"
        )

        print(
            "Steady-active adjusted MAE: "
            f"{workload_summary['mean_adjusted_steady_active_mae_mA']:.3f} mA"
        )

        print(
            "Steady-active adjusted mean error: "
            f"{workload_summary['mean_adjusted_steady_active_mean_error_mA']:.3f} mA"
        )

        print(
            "Steady-active adjusted error std: "
            f"{workload_summary['mean_adjusted_steady_active_error_std_mA']:.3f} mA"
        )

    # ========================================================
    # Combined summary
    # ========================================================

    if not all_workload_summary_rows:
        raise RuntimeError(
            "No workload datasets were processed."
        )

    workload_summary_df = pd.DataFrame(
        all_workload_summary_rows
    )

    out_combined_summary = (
        BASE_OUT_DIR
        / "core_only_workload_validation_summary.csv"
    )

    workload_summary_df.to_csv(
        out_combined_summary,
        index=False,
    )

    compact_columns = [
        "workload",
        "run_count",

        "measured_mean_idle_baseline_mA",
        "measured_mean_steady_active_mA",
        "measured_mean_delta_mA",

        "delta_error_model_minus_measured_mA",

        "mean_adjusted_steady_active_mae_mA",
        "mean_adjusted_steady_active_mean_error_mA",
        "mean_adjusted_steady_active_error_std_mA",

        "mean_adjusted_full_active_mae_mA",
        "mean_adjusted_full_active_mean_error_mA",
        "mean_adjusted_full_active_error_std_mA",

        "mean_fixed_steady_active_mae_mA",
        "mean_fixed_steady_active_mean_error_mA",
        "mean_fixed_steady_active_error_std_mA",
    ]

    print()
    print("=" * 100)
    print("Core-only workload validation summary")
    print("=" * 100)

    print(
        workload_summary_df[
            compact_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Saved: {out_combined_summary}")

    print()
    print("Error convention:")
    print(
        "error = prediction - measurement"
    )
    print(
        "positive mean error = overestimate"
    )
    print(
        "negative mean error = underestimate"
    )


if __name__ == "__main__":
    main()