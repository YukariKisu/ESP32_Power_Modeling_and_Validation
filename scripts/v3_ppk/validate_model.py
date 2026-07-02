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

# Model estimated from CPU_100 training data
I_IDLE_MODEL_MA = 47.2668
GAIN_MODEL_MA = 20.1888
TAU_S = 0.00049

# PPK2 raw data: 100 kS/s
# Processed waveform: 10 kS/s = 0.1 ms interval
DOWNSAMPLE_STEP = 10

# Common aligned time axis
# active start = 0 s
GRID_DT_S = 0.0001
ALIGNED_GRID = np.arange(-10.0, 30.0, GRID_DT_S)

# Evaluation windows
TRANSIENT_WINDOW_S = (0.0, 0.005)      # 0 to 5 ms
STEADY_WINDOW_S = (0.010, 10.0)        # 10 ms to 10 s
FULL_ACTIVE_WINDOW_S = (0.0, 10.0)     # 0 to 10 s

# Smoothing for smoothed waveform MAE
# 10 ms smoothing = 100 samples at 10 kS/s
SMOOTH_WINDOW_S = 0.010
SMOOTH_WINDOW_SAMPLES = int(SMOOTH_WINDOW_S / GRID_DT_S)

if SMOOTH_WINDOW_SAMPLES % 2 == 0:
    SMOOTH_WINDOW_SAMPLES += 1

# If True, plots are shown and script waits until you close each figure.
# If False, plots are only saved and the script continues automatically.
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

    # Downsample: 100 kS/s -> about 10 kS/s
    # 0.1 ms interval
    return (
        time_s.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True),
        current_mA.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True),
    )


def detect_active_edges(time_s, current_mA):
    """
    Current-based time alignment.

    Detect:
    - power-on time
    - active rising edge
    - active falling edge
    - idle level
    - active level

    This follows the same logic used for CPU_100 training data.
    """

    # Smooth current for robust edge detection
    smooth = current_mA.rolling(window=200, center=True, min_periods=1).mean()

    # Detect power-on roughly
    max_level = smooth.quantile(0.99)
    power_threshold = max_level * 0.1

    power_candidates = np.where(smooth > power_threshold)[0]
    if len(power_candidates) == 0:
        raise ValueError("Could not detect power-on")

    power_on_time = time_s.iloc[power_candidates[0]]

    # Estimate idle and active levels relative to power-on
    # These windows assume the same measurement timing as CPU_100.
    idle_mask = (time_s > power_on_time + 3) & (time_s < power_on_time + 9)
    active_mask = (time_s > power_on_time + 13) & (time_s < power_on_time + 25)

    idle_level = smooth[idle_mask].median()
    active_level = smooth[active_mask].median()

    if np.isnan(idle_level) or np.isnan(active_level):
        raise ValueError("Could not estimate idle or active level")

    delta = active_level - idle_level

    if delta <= 0:
        raise ValueError(
            f"Active level is not higher than idle level. "
            f"idle={idle_level:.3f} mA, active={active_level:.3f} mA"
        )

    # Midpoint threshold
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


def model_prediction_current_mA(time_from_active_start_s, u):
    """
    First-order dynamic current model.

    For t < 0:
        predicted current = idle current

    For t >= 0:
        I(t) = I_inf - (I_inf - I_idle) * exp(-t / tau)

    where:
        I_inf = I_idle + gain * u
    """

    I_inf = I_IDLE_MODEL_MA + GAIN_MODEL_MA * u

    pred = np.full_like(
        time_from_active_start_s,
        fill_value=I_IDLE_MODEL_MA,
        dtype=float,
    )

    active_mask = time_from_active_start_s >= 0
    t = time_from_active_start_s[active_mask]

    pred[active_mask] = (
        I_inf
        - (I_inf - I_IDLE_MODEL_MA) * np.exp(-t / TAU_S)
    )

    return pred


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


def smooth_waveform(values, window_samples):
    """
    Smooth waveform using centered rolling mean.

    This is useful when comparing an averaged ODE model against measured data
    that contains fast switching.
    """

    values = np.asarray(values, dtype=float)

    original_nan_mask = ~np.isfinite(values)

    smoothed = (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
        .copy()
    )

    # Keep outside-interpolation regions as NaN
    smoothed[original_nan_mask] = np.nan

    return smoothed


def calculate_level_error(measured_level_mA, predicted_level_mA):
    """
    Steady-state level error.

    Positive signed error:
        measured current is higher than predicted.

    Negative signed error:
        measured current is lower than predicted.
    """

    signed_error = measured_level_mA - predicted_level_mA
    abs_error = abs(signed_error)

    return signed_error, abs_error


def save_validation_plot(
    intensity,
    aligned_currents,
    mean_current,
    smoothed_mean_current,
    std_current,
    predicted_current,
    out_dir,
):
    """
    Save full-range and transient-zoom validation plots.
    """

    # ------------------------------------------------------------
    # Full plot
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    for i, current in enumerate(aligned_currents):
        plt.plot(
            ALIGNED_GRID,
            current,
            alpha=0.20,
            linewidth=0.6,
            label="individual aligned runs" if i == 0 else None,
        )

    plt.plot(
        ALIGNED_GRID,
        mean_current,
        linewidth=2.2,
        label="mean measured waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        smoothed_mean_current,
        linewidth=2.0,
        linestyle="-.",
        label="smoothed mean measured waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        predicted_current,
        linestyle="--",
        linewidth=2.0,
        label="model prediction",
    )

    plt.fill_between(
        ALIGNED_GRID,
        mean_current - std_current,
        mean_current + std_current,
        alpha=0.20,
        label="measured mean ± 1 std",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"PPK2 CPU-only {intensity}% busy: "
        "aligned measured waveform vs model prediction"
    )
    plt.xlim(-5, 25)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_plot = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_aligned_measured_vs_predicted.png",
    )

    plt.savefig(out_plot, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # Transient zoom plot
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    for i, current in enumerate(aligned_currents):
        plt.plot(
            ALIGNED_GRID,
            current,
            alpha=0.20,
            linewidth=0.6,
            label="individual aligned runs" if i == 0 else None,
        )

    plt.plot(
        ALIGNED_GRID,
        mean_current,
        linewidth=2.2,
        label="mean measured waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        smoothed_mean_current,
        linewidth=2.0,
        linestyle="-.",
        label="smoothed mean measured waveform",
    )

    plt.plot(
        ALIGNED_GRID,
        predicted_current,
        linestyle="--",
        linewidth=2.0,
        label="model prediction",
    )

    plt.fill_between(
        ALIGNED_GRID,
        mean_current - std_current,
        mean_current + std_current,
        alpha=0.20,
        label="measured mean ± 1 std",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time from active start [s]")
    plt.ylabel("Current [mA]")
    plt.title(
        f"PPK2 CPU-only {intensity}% busy: "
        "transient zoom"
    )
    plt.xlim(-0.005, 0.030)
    plt.ylim(40, 90)
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out_zoom_plot = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_transient_zoom_measured_vs_predicted.png",
    )

    plt.savefig(out_zoom_plot, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return out_plot, out_zoom_plot


# ============================================================
# Main processing
# ============================================================

all_validation_summary_rows = []

for intensity in INTENSITIES:
    u = intensity / 100.0

    data_pattern = (
        f"{BASE_RAW_DIR}/cpu_{intensity}/ppk_cpu_{intensity}_run*.csv"
    )

    out_dir = f"{BASE_OUT_DIR}/cpu_{intensity}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(data_pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No CSV files found: {data_pattern}")

    print("\n" + "=" * 70)
    print(f"Processing CPU_{intensity}")
    print("=" * 70)

    print("Files:")
    for f in files:
        print(f)

    aligned_currents = []
    summary_rows = []
    mae_rows = []

    predicted_current = model_prediction_current_mA(ALIGNED_GRID, u)
    model_final_current_mA = I_IDLE_MODEL_MA + GAIN_MODEL_MA * u

    # ------------------------------------------------------------
    # Process each run
    # ------------------------------------------------------------

    for csv_path in files:
        label = os.path.basename(csv_path).replace(".csv", "")

        print(f"\nProcessing: {label}")

        time_s, current_mA = load_ppk2_csv(csv_path)

        (
            power_on_time,
            rising_time,
            falling_time,
            idle_level,
            active_level,
            threshold,
        ) = detect_active_edges(time_s, current_mA)

        aligned_time = time_s - rising_time

        # Interpolate each run onto common relative time grid
        interpolated_current = np.interp(
            ALIGNED_GRID,
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

        # ------------------------------------------------------------
        # 1. Raw waveform MAE
        # ------------------------------------------------------------

        raw_mae_full_mA = calculate_mae(
            ALIGNED_GRID,
            interpolated_current,
            predicted_current,
            FULL_ACTIVE_WINDOW_S,
        )

        raw_mae_transient_mA = calculate_mae(
            ALIGNED_GRID,
            interpolated_current,
            predicted_current,
            TRANSIENT_WINDOW_S,
        )

        raw_mae_steady_mA = calculate_mae(
            ALIGNED_GRID,
            interpolated_current,
            predicted_current,
            STEADY_WINDOW_S,
        )

        # ------------------------------------------------------------
        # 2. Smoothed waveform MAE
        # ------------------------------------------------------------

        smoothed_current = smooth_waveform(
            interpolated_current,
            SMOOTH_WINDOW_SAMPLES,
        )

        smoothed_mae_full_mA = calculate_mae(
            ALIGNED_GRID,
            smoothed_current,
            predicted_current,
            FULL_ACTIVE_WINDOW_S,
        )

        smoothed_mae_transient_mA = calculate_mae(
            ALIGNED_GRID,
            smoothed_current,
            predicted_current,
            TRANSIENT_WINDOW_S,
        )

        smoothed_mae_steady_mA = calculate_mae(
            ALIGNED_GRID,
            smoothed_current,
            predicted_current,
            STEADY_WINDOW_S,
        )

        # ------------------------------------------------------------
        # 3. Steady-state level error
        # ------------------------------------------------------------

        level_error_signed_mA, level_error_abs_mA = calculate_level_error(
            active_level,
            model_final_current_mA,
        )

        # ------------------------------------------------------------
        # Save per-run summaries
        # ------------------------------------------------------------

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

        mae_rows.append({
            "run": label,
            "cpu_intensity_percent": intensity,
            "u": u,

            "model_idle_current_mA": I_IDLE_MODEL_MA,
            "model_gain_mA": GAIN_MODEL_MA,
            "model_tau_s": TAU_S,
            "model_final_current_mA": model_final_current_mA,

            "measured_idle_level_mA": idle_level,
            "measured_active_level_mA": active_level,

            # 1. Raw waveform MAE
            "raw_mae_full_active_0_to_10s_mA": raw_mae_full_mA,
            "raw_mae_transient_0_to_5ms_mA": raw_mae_transient_mA,
            "raw_mae_steady_10ms_to_10s_mA": raw_mae_steady_mA,

            # 2. Smoothed waveform MAE
            "smoothed_mae_full_active_0_to_10s_mA": smoothed_mae_full_mA,
            "smoothed_mae_transient_0_to_5ms_mA": smoothed_mae_transient_mA,
            "smoothed_mae_steady_10ms_to_10s_mA": smoothed_mae_steady_mA,

            # 3. Steady-state level error
            "steady_level_error_signed_mA": level_error_signed_mA,
            "steady_level_error_abs_mA": level_error_abs_mA,
        })

    # ------------------------------------------------------------
    # Mean and std waveform
    # ------------------------------------------------------------

    aligned_array = np.array(aligned_currents)

    mean_current = np.nanmean(aligned_array, axis=0)
    std_current = np.nanstd(aligned_array, axis=0, ddof=1)

    smoothed_mean_current = smooth_waveform(
        mean_current,
        SMOOTH_WINDOW_SAMPLES,
    )

    # ------------------------------------------------------------
    # Mean waveform MAE
    # ------------------------------------------------------------

    mean_waveform_mae_full_mA = calculate_mae(
        ALIGNED_GRID,
        mean_current,
        predicted_current,
        FULL_ACTIVE_WINDOW_S,
    )

    mean_waveform_mae_transient_mA = calculate_mae(
        ALIGNED_GRID,
        mean_current,
        predicted_current,
        TRANSIENT_WINDOW_S,
    )

    mean_waveform_mae_steady_mA = calculate_mae(
        ALIGNED_GRID,
        mean_current,
        predicted_current,
        STEADY_WINDOW_S,
    )

    smoothed_mean_waveform_mae_full_mA = calculate_mae(
        ALIGNED_GRID,
        smoothed_mean_current,
        predicted_current,
        FULL_ACTIVE_WINDOW_S,
    )

    smoothed_mean_waveform_mae_transient_mA = calculate_mae(
        ALIGNED_GRID,
        smoothed_mean_current,
        predicted_current,
        TRANSIENT_WINDOW_S,
    )

    smoothed_mean_waveform_mae_steady_mA = calculate_mae(
        ALIGNED_GRID,
        smoothed_mean_current,
        predicted_current,
        STEADY_WINDOW_S,
    )

    # ------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------

    out_plot, out_zoom_plot = save_validation_plot(
        intensity=intensity,
        aligned_currents=aligned_currents,
        mean_current=mean_current,
        smoothed_mean_current=smoothed_mean_current,
        std_current=std_current,
        predicted_current=predicted_current,
        out_dir=out_dir,
    )

    # ------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    out_summary = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_run_summary.csv",
    )
    summary_df.to_csv(out_summary, index=False)

    mae_df = pd.DataFrame(mae_rows)

    out_mae = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_validation_mae_by_run.csv",
    )
    mae_df.to_csv(out_mae, index=False)

    mean_df = pd.DataFrame({
        "time_from_active_start_s": ALIGNED_GRID,
        "mean_current_mA": mean_current,
        "smoothed_mean_current_mA": smoothed_mean_current,
        "std_current_mA": std_current,
        "predicted_current_mA": predicted_current,
    })

    out_mean = os.path.join(
        out_dir,
        f"ppk_cpu_{intensity}_aligned_mean_waveform_with_prediction.csv",
    )
    mean_df.to_csv(out_mean, index=False)

    # ------------------------------------------------------------
    # Summary for this intensity
    # ------------------------------------------------------------

    intensity_summary = {
        "cpu_intensity_percent": intensity,
        "u": u,
        "n_runs": len(files),

        "model_final_current_mA": model_final_current_mA,

        "measured_mean_idle_level_mA": summary_df["idle_level_mA"].mean(),
        "measured_mean_active_level_mA": summary_df["active_level_mA"].mean(),
        "measured_std_active_level_mA": summary_df["active_level_mA"].std(ddof=1),

        # 1. Raw waveform MAE
        "mean_raw_mae_full_active_0_to_10s_mA": mae_df["raw_mae_full_active_0_to_10s_mA"].mean(),
        "std_raw_mae_full_active_0_to_10s_mA": mae_df["raw_mae_full_active_0_to_10s_mA"].std(ddof=1),

        "mean_raw_mae_transient_0_to_5ms_mA": mae_df["raw_mae_transient_0_to_5ms_mA"].mean(),
        "std_raw_mae_transient_0_to_5ms_mA": mae_df["raw_mae_transient_0_to_5ms_mA"].std(ddof=1),

        "mean_raw_mae_steady_10ms_to_10s_mA": mae_df["raw_mae_steady_10ms_to_10s_mA"].mean(),
        "std_raw_mae_steady_10ms_to_10s_mA": mae_df["raw_mae_steady_10ms_to_10s_mA"].std(ddof=1),

        # 2. Smoothed waveform MAE
        "mean_smoothed_mae_full_active_0_to_10s_mA": mae_df["smoothed_mae_full_active_0_to_10s_mA"].mean(),
        "std_smoothed_mae_full_active_0_to_10s_mA": mae_df["smoothed_mae_full_active_0_to_10s_mA"].std(ddof=1),

        "mean_smoothed_mae_transient_0_to_5ms_mA": mae_df["smoothed_mae_transient_0_to_5ms_mA"].mean(),
        "std_smoothed_mae_transient_0_to_5ms_mA": mae_df["smoothed_mae_transient_0_to_5ms_mA"].std(ddof=1),

        "mean_smoothed_mae_steady_10ms_to_10s_mA": mae_df["smoothed_mae_steady_10ms_to_10s_mA"].mean(),
        "std_smoothed_mae_steady_10ms_to_10s_mA": mae_df["smoothed_mae_steady_10ms_to_10s_mA"].std(ddof=1),

        # 2b. Mean waveform MAE
        "mean_waveform_mae_full_active_0_to_10s_mA": mean_waveform_mae_full_mA,
        "mean_waveform_mae_transient_0_to_5ms_mA": mean_waveform_mae_transient_mA,
        "mean_waveform_mae_steady_10ms_to_10s_mA": mean_waveform_mae_steady_mA,

        "smoothed_mean_waveform_mae_full_active_0_to_10s_mA": smoothed_mean_waveform_mae_full_mA,
        "smoothed_mean_waveform_mae_transient_0_to_5ms_mA": smoothed_mean_waveform_mae_transient_mA,
        "smoothed_mean_waveform_mae_steady_10ms_to_10s_mA": smoothed_mean_waveform_mae_steady_mA,

        # 3. Steady-state level error
        "mean_steady_level_error_signed_mA": mae_df["steady_level_error_signed_mA"].mean(),
        "std_steady_level_error_signed_mA": mae_df["steady_level_error_signed_mA"].std(ddof=1),

        "mean_steady_level_error_abs_mA": mae_df["steady_level_error_abs_mA"].mean(),
        "std_steady_level_error_abs_mA": mae_df["steady_level_error_abs_mA"].std(ddof=1),
    }

    all_validation_summary_rows.append(intensity_summary)

    print("\nSaved:")
    print(out_plot)
    print(out_zoom_plot)
    print(out_summary)
    print(out_mae)
    print(out_mean)

    print("\nIntensity summary:")
    print(pd.DataFrame([intensity_summary]))


# ============================================================
# Save combined validation summary
# ============================================================

validation_summary_df = pd.DataFrame(all_validation_summary_rows)

out_validation_summary = os.path.join(
    BASE_OUT_DIR,
    "cpu_validation_mae_summary_cpu75_cpu50_cpu25.csv",
)

validation_summary_df.to_csv(out_validation_summary, index=False)

print("\n" + "=" * 70)
print("Combined validation summary")
print("=" * 70)
print(validation_summary_df)

print("\nSaved:")
print(out_validation_summary)


# ============================================================
# Print compact summary for quick reading
# ============================================================

compact_cols = [
    "cpu_intensity_percent",
    "model_final_current_mA",
    "measured_mean_active_level_mA",
    "mean_raw_mae_full_active_0_to_10s_mA",
    "mean_smoothed_mae_full_active_0_to_10s_mA",
    "mean_waveform_mae_full_active_0_to_10s_mA",
    "smoothed_mean_waveform_mae_full_active_0_to_10s_mA",
    "mean_steady_level_error_signed_mA",
    "mean_steady_level_error_abs_mA",
]

print("\n" + "=" * 70)
print("Compact summary")
print("=" * 70)
print(validation_summary_df[compact_cols])