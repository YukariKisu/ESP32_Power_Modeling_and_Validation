from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

CONDITION = "adc_single_1ms"

# Search this root recursively for adc_single_1ms_run*.csv
INPUT_ROOT = Path("data/raw/v3_ppk/peripheral/adc/final_predictioned")

# Output directory
OUTPUT_DIR = Path(
    "results/adc/adc_single_1ms_measured_vs_original"
)

# Model parameters (original remodeled ADC model)
DELTA_I_MA = 12.943
TAU_RISE_S = 0.579e-3
TAU_FALL_S = 0.579e-3

# Firmware timing
SYNC_TO_ACTIVE_START_S = 16.0
ACTIVE_DURATION_S = 20.0

# Original Single 1 ms input definition
PULSE_PERIOD_S = 1.0e-3
PULSE_ON_S = 44.0e-6

# Time ranges to plot
ACTIVE_START_PLOT_START_S = -10e-3
ACTIVE_START_PLOT_END_S = 30e-3

STEADY_ZOOM_START_S = 5.000
STEADY_ZOOM_END_S = 5.010

# Uniform resampling
UNIFORM_DT_S = 10e-6

# Baseline windows relative to active start
INITIAL_BASELINE_START_S = -10.0
INITIAL_BASELINE_END_S = -0.5

FINAL_BASELINE_START_S = 20.5
FINAL_BASELINE_END_S = 29.5

# Sync detection
SYNC_SEARCH_START_S = 2.0
SYNC_SEARCH_END_S = 8.0
SYNC_SMOOTH_WINDOW_S = 5e-3
SYNC_THRESHOLD_RATIO = 0.40
MIN_SYNC_DURATION_S = 0.20


# ============================================================
# Helpers
# ============================================================

def collect_csv_files(
    input_root: Path,
    condition: str,
) -> list[Path]:
    pattern = f"{condition}_run*.csv"
    return sorted(input_root.rglob(pattern))


def extract_run_id(path: Path) -> str:
    match = re.search(r"(run\d+)", path.stem)
    return match.group(1) if match else path.stem


def _normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = [
        str(col).strip() for col in dataframe.columns
    ]
    return dataframe


def load_measurement_csv(
    csv_path: Path,
) -> pd.DataFrame:

    # First try normal comma-separated CSV
    dataframe = pd.read_csv(
        csv_path,
        comment="#",
    )

    # If only one column was detected, try semicolon-separated
    if len(dataframe.columns) == 1:
        dataframe = pd.read_csv(
            csv_path,
            comment="#",
            sep=";",
        )

    dataframe.columns = [
        str(col).strip()
        for col in dataframe.columns
    ]

    print(
        f"{csv_path.name}: columns = "
        f"{list(dataframe.columns)}"
    )

    # --------------------------------------------------------
    # Find time column
    # --------------------------------------------------------

    time_col = None

    for col in dataframe.columns:
        name = col.lower().replace(" ", "")

        if (
            "timestamp" in name
            or "time" in name
        ):
            time_col = col
            break

    if time_col is None:
        raise ValueError(
            f"Could not find time column in {csv_path}. "
            f"Columns: {list(dataframe.columns)}"
        )

    # --------------------------------------------------------
    # Find current column
    # --------------------------------------------------------

    current_col = None

    for col in dataframe.columns:
        name = col.lower().replace(" ", "")

        if "current" in name:
            current_col = col
            break

    if current_col is None:
        raise ValueError(
            f"Could not find current column in {csv_path}. "
            f"Columns: {list(dataframe.columns)}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    time_raw = pd.to_numeric(
        dataframe[time_col],
        errors="coerce",
    )

    current_raw = pd.to_numeric(
        dataframe[current_col],
        errors="coerce",
    )

    valid = (
        time_raw.notna()
        & current_raw.notna()
    )

    time_values = (
        time_raw[valid]
        .to_numpy(dtype=float)
    )

    current_values = (
        current_raw[valid]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Time -> seconds
    # --------------------------------------------------------

    time_name = (
        time_col
        .lower()
        .replace(" ", "")
    )

    if "ms" in time_name:
        time_s = (
            time_values
            / 1000.0
        )

    elif (
        "us" in time_name
        or "µs" in time_name
    ):
        time_s = (
            time_values
            / 1_000_000.0
        )

    else:
        time_s = time_values

    # Start each raw recording from t = 0
    time_s = (
        time_s
        - time_s[0]
    )

    # --------------------------------------------------------
    # Current -> mA
    # --------------------------------------------------------

    current_name = (
        current_col
        .lower()
        .replace(" ", "")
    )

    if (
        "ua" in current_name
        or "µa" in current_name
    ):
        current_ma = (
            current_values
            / 1000.0
        )

    elif "ma" in current_name:
        current_ma = (
            current_values
        )

    elif (
        "[a]" in current_name
        or "(a)" in current_name
    ):
        current_ma = (
            current_values
            * 1000.0
        )

    else:
        # Fallback based on magnitude
        median_abs = float(
            np.median(
                np.abs(current_values)
            )
        )

        if median_abs < 1.0:
            current_ma = (
                current_values
                * 1000.0
            )

        elif median_abs > 1000.0:
            current_ma = (
                current_values
                / 1000.0
            )

        else:
            current_ma = (
                current_values
            )

    return pd.DataFrame(
        {
            "time_s": time_s,
            "current_ma": current_ma,
        }
    )


def moving_average_same(
    values: np.ndarray,
    window_len: int,
) -> np.ndarray:
    if window_len <= 1:
        return values.copy()

    kernel = np.ones(window_len, dtype=float) / window_len
    return np.convolve(values, kernel, mode="same")


def find_first_true_run(
    mask: np.ndarray,
    min_len: int,
) -> tuple[int, int] | None:
    start_idx = None

    for idx, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = idx

        if (not flag) and (start_idx is not None):
            end_idx = idx
            if (end_idx - start_idx) >= min_len:
                return start_idx, end_idx
            start_idx = None

    if start_idx is not None:
        end_idx = len(mask)
        if (end_idx - start_idx) >= min_len:
            return start_idx, end_idx

    return None


def detect_sync_start_end(
    time_s: np.ndarray,
    current_ma: np.ndarray,
) -> tuple[float, float]:
    dt_s = float(np.median(np.diff(time_s)))
    smooth_len = max(
        1,
        int(round(SYNC_SMOOTH_WINDOW_S / dt_s)),
    )

    smoothed = moving_average_same(current_ma, smooth_len)

    search_mask = (
        (time_s >= SYNC_SEARCH_START_S)
        & (time_s <= SYNC_SEARCH_END_S)
    )

    if not np.any(search_mask):
        raise ValueError("Sync search window is empty")

    baseline_mask = (
        (time_s >= 0.5)
        & (time_s <= 2.5)
    )
    if not np.any(baseline_mask):
        baseline_mask = time_s < SYNC_SEARCH_START_S

    baseline = float(np.median(smoothed[baseline_mask]))
    peak = float(np.max(smoothed[search_mask]))

    threshold = baseline + SYNC_THRESHOLD_RATIO * (
        peak - baseline
    )

    active_mask = (
        (smoothed >= threshold)
        & search_mask
    )

    min_len = max(
        1,
        int(round(MIN_SYNC_DURATION_S / dt_s)),
    )

    run = find_first_true_run(active_mask, min_len)
    if run is None:
        raise ValueError("Could not detect sync pulse")

    start_idx, end_idx = run
    sync_start_s = float(time_s[start_idx])
    sync_end_s = float(time_s[end_idx - 1])

    return sync_start_s, sync_end_s


def calculate_mean_in_window(
    time_rel_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    mask = (
        (time_rel_s >= start_s)
        & (time_rel_s <= end_s)
    )
    if not np.any(mask):
        raise ValueError(
            f"No samples in window [{start_s}, {end_s}]"
        )
    return float(np.mean(values[mask]))


def align_run_to_active_start(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    time_s = dataframe["time_s"].to_numpy()
    current_ma = dataframe["current_ma"].to_numpy()

    sync_start_s, _ = detect_sync_start_end(
        time_s=time_s,
        current_ma=current_ma,
    )

    active_start_s = sync_start_s + SYNC_TO_ACTIVE_START_S
    time_rel_s = time_s - active_start_s

    initial_mean_ma = calculate_mean_in_window(
        time_rel_s,
        current_ma,
        INITIAL_BASELINE_START_S,
        INITIAL_BASELINE_END_S,
    )

    final_mean_ma = calculate_mean_in_window(
        time_rel_s,
        current_ma,
        FINAL_BASELINE_START_S,
        FINAL_BASELINE_END_S,
    )

    baseline_ma = (
        initial_mean_ma + final_mean_ma
    ) / 2.0

    delta_ma = current_ma - baseline_ma

    return time_rel_s, delta_ma, baseline_ma


def interpolate_segment(
    time_rel_s: np.ndarray,
    delta_ma: np.ndarray,
    start_s: float,
    end_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    uniform_time_s = np.arange(
        start_s,
        end_s + 0.5 * dt_s,
        dt_s,
    )

    interpolated = np.interp(
        uniform_time_s,
        time_rel_s,
        delta_ma,
    )

    return uniform_time_s, interpolated


def input_u_original(
    time_s: np.ndarray,
) -> np.ndarray:
    u = np.zeros_like(time_s, dtype=float)

    active_mask = (
        (time_s >= 0.0)
        & (time_s <= ACTIVE_DURATION_S)
    )
    active_time = time_s[active_mask]

    phase = np.mod(active_time, PULSE_PERIOD_S)
    pulse_mask = phase < PULSE_ON_S

    tmp = np.zeros_like(active_time, dtype=float)
    tmp[pulse_mask] = 1.0
    u[active_mask] = tmp

    return u


def simulate_first_order_delta(
    time_s: np.ndarray,
    delta_i_ma: float,
    tau_rise_s: float,
    tau_fall_s: float,
) -> np.ndarray:
    dt_s = float(np.median(np.diff(time_s)))
    u = input_u_original(time_s)

    predicted = np.zeros_like(time_s, dtype=float)

    for idx in range(1, len(time_s)):
        target = delta_i_ma * u[idx - 1]
        tau = tau_rise_s if target >= predicted[idx - 1] else tau_fall_s

        predicted[idx] = (
            predicted[idx - 1]
            + dt_s * (target - predicted[idx - 1]) / tau
        )

    return predicted


def build_measured_mean(
    csv_files: list[Path],
    start_s: float,
    end_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_segments = []

    common_time_s = None

    for csv_path in csv_files:
        dataframe = load_measurement_csv(csv_path)
        time_rel_s, delta_ma, _ = align_run_to_active_start(
            dataframe
        )

        uniform_time_s, uniform_delta_ma = interpolate_segment(
            time_rel_s=time_rel_s,
            delta_ma=delta_ma,
            start_s=start_s,
            end_s=end_s,
            dt_s=dt_s,
        )

        if common_time_s is None:
            common_time_s = uniform_time_s

        all_segments.append(uniform_delta_ma)

    measured_matrix = np.vstack(all_segments)
    measured_mean = np.mean(measured_matrix, axis=0)

    return common_time_s, measured_mean


def plot_active_start_view(
    time_s: np.ndarray,
    measured_ma: np.ndarray,
    predicted_ma: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_s * 1e3,
        measured_ma,
        label="Measured mean",
    )
    plt.plot(
        time_s * 1e3,
        predicted_ma,
        "--",
        label="Original prediction (44 µs pulse / 1 ms)",
    )
    plt.axvline(
        0.0,
        linestyle=":",
        label="Firmware active start",
    )
    plt.xlabel("Time relative to firmware ADC active start [ms]")
    plt.ylabel("Current increase from idle baseline [mA]")
    plt.title("ADC Single 1 ms — Measured vs Original model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_steady_zoom_view(
    time_s: np.ndarray,
    measured_ma: np.ndarray,
    predicted_ma: np.ndarray,
    output_path: Path,
) -> None:
    mask = (
        (time_s >= STEADY_ZOOM_START_S)
        & (time_s <= STEADY_ZOOM_END_S)
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        (time_s[mask] - STEADY_ZOOM_START_S) * 1e3,
        measured_ma[mask],
        label="Measured mean",
    )
    plt.plot(
        (time_s[mask] - STEADY_ZOOM_START_S) * 1e3,
        predicted_ma[mask],
        "--",
        label="Original prediction (44 µs pulse / 1 ms)",
    )
    plt.xlabel(
        f"Time within {STEADY_ZOOM_START_S:.3f}–{STEADY_ZOOM_END_S:.3f} s active window [ms]"
    )
    plt.ylabel("Current increase from idle baseline [mA]")
    plt.title("ADC Single 1 ms — 10 ms steady-state zoom")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(
        INPUT_ROOT,
        CONDITION,
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for {CONDITION} under {INPUT_ROOT}"
        )

    print(f"Found {len(csv_files)} CSV files")
    for path in csv_files:
        print(f"  {path}")

    # 1) Active-start view
    active_time_s, measured_active_mean_ma = build_measured_mean(
        csv_files=csv_files,
        start_s=ACTIVE_START_PLOT_START_S,
        end_s=ACTIVE_START_PLOT_END_S,
        dt_s=UNIFORM_DT_S,
    )

    predicted_active_ma = simulate_first_order_delta(
        time_s=active_time_s,
        delta_i_ma=DELTA_I_MA,
        tau_rise_s=TAU_RISE_S,
        tau_fall_s=TAU_FALL_S,
    )

    active_plot_path = OUTPUT_DIR / (
        "adc_single_1ms_measured_vs_original_active_start.png"
    )
    plot_active_start_view(
        time_s=active_time_s,
        measured_ma=measured_active_mean_ma,
        predicted_ma=predicted_active_ma,
        output_path=active_plot_path,
    )
    print(f"Saved: {active_plot_path}")

    # 2) Steady-state zoom
    steady_time_s, measured_steady_mean_ma = build_measured_mean(
        csv_files=csv_files,
        start_s=STEADY_ZOOM_START_S,
        end_s=STEADY_ZOOM_END_S,
        dt_s=UNIFORM_DT_S,
    )

    predicted_steady_ma = simulate_first_order_delta(
        time_s=steady_time_s,
        delta_i_ma=DELTA_I_MA,
        tau_rise_s=TAU_RISE_S,
        tau_fall_s=TAU_FALL_S,
    )

    steady_plot_path = OUTPUT_DIR / (
        "adc_single_1ms_measured_vs_original_steady_zoom.png"
    )
    plot_steady_zoom_view(
        time_s=steady_time_s,
        measured_ma=measured_steady_mean_ma,
        predicted_ma=predicted_steady_ma,
        output_path=steady_plot_path,
    )
    print(f"Saved: {steady_plot_path}")

    # 3) Simple summary
    measured_steady_mean = float(np.mean(measured_steady_mean_ma))
    predicted_steady_mean = float(np.mean(predicted_steady_ma))
    mae_steady = float(
        np.mean(np.abs(measured_steady_mean_ma - predicted_steady_ma))
    )
    me_steady = float(
        np.mean(measured_steady_mean_ma - predicted_steady_ma)
    )
    rmse_steady = float(
        np.sqrt(
            np.mean(
                (measured_steady_mean_ma - predicted_steady_ma) ** 2
            )
        )
    )

    summary = pd.DataFrame(
        [
            {
                "case": "Measured vs Original",
                "measured_delta_mean_ma": measured_steady_mean,
                "predicted_delta_mean_ma": predicted_steady_mean,
                "mae_ma": mae_steady,
                "me_ma": me_steady,
                "rmse_ma": rmse_steady,
            }
        ]
    )

    summary_path = OUTPUT_DIR / (
        "adc_single_1ms_measured_vs_original_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()