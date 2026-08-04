from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/v3_ppk/peripheral/adc/final_predictioned")

INPUT_DIRS = {
    "adc_single_1ms": RAW_ROOT / "adc_periodic_single" / "adc_single_1ms",
    "adc_single_10ms": RAW_ROOT / "adc_periodic_single" / "adc_single_10ms",
    "adc_single_100ms": RAW_ROOT / "adc_periodic_single" / "adc_single_100ms",
    "adc_burst_100ms_10samples": RAW_ROOT / "adc_periodic_burst" / "adc_burst_100ms_10samples",
    "adc_burst_100ms_100samples": RAW_ROOT / "adc_periodic_burst" / "adc_burst_100ms_100samples",
    "adc_burst_100ms_1000samples": RAW_ROOT / "adc_periodic_burst" / "adc_burst_100ms_1000samples",
}

CONDITION_TO_PLOT = "all"
RUN_ID = None  # None = first CSV found
OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/waveform_comparison"
)

ODE_GAIN_MA = 20.1888
TAU_RISE_S = 0.00049
TAU_FALL_S = 0.00049

WORKLOADS = {
    "adc_single_1ms": {"period_s": 0.001, "active_duration_s": 44e-6},
    "adc_single_10ms": {"period_s": 0.010, "active_duration_s": 44e-6},
    "adc_single_100ms": {"period_s": 0.100, "active_duration_s": 44e-6},
    "adc_burst_100ms_10samples": {"period_s": 0.100, "active_duration_s": 537e-6},
    "adc_burst_100ms_100samples": {"period_s": 0.100, "active_duration_s": 4396e-6},
    "adc_burst_100ms_1000samples": {"period_s": 0.100, "active_duration_s": 43984e-6},
}

INITIAL_START_S = 5.5
INITIAL_END_S = 15.5
ACTIVE_START_S = 15.5
ACTIVE_END_S = 35.5
FINAL_START_S = 35.5
FINAL_END_S = 45.5
WINDOW_TRIM_S = 0.2

FULL_XLIM = (5.0, 46.0)
ACTIVE_START_XLIM = (15.45, 15.60)
# Per-workload detail windows. Each one shows several complete cycles.
ACTIVE_ZOOM_WINDOWS = {
    "adc_single_1ms": (20.000, 20.010),          # 10 cycles
    "adc_single_10ms": (20.000, 20.050),         # 5 cycles
    "adc_single_100ms": (20.000, 20.300),        # 3 cycles
    "adc_burst_100ms_10samples": (20.000, 20.300),
    "adc_burst_100ms_100samples": (20.000, 20.300),
    "adc_burst_100ms_1000samples": (20.000, 20.300),
}
MEASURED_DISPLAY_SMOOTH_S = 0.00010
MAX_PLOT_POINTS = 200_000

SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_MARGIN_S = 1.5
SYNC_SMOOTH_WINDOW_S = 0.02

TIME_COLUMN_CANDIDATES = (
    "time", "timestamp", "timestamp_s", "time_s", "time (s)", "time[s]", "timestamp(ms)"
)
CURRENT_COLUMN_CANDIDATES = (
    "current", "current_ma", "current (ma)", "current[ma]",
    "current_ua", "current (ua)", "current[ua]",
    "current_a", "current (a)", "current[a]",
)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    for column in columns:
        key = normalize_name(column)
        if kind == "time" and "time" in key:
            return column
        if kind == "current" and "current" in key:
            return column
    raise ValueError(f"Could not find {kind} column in columns: {list(columns)}")


def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if "ms" in name:
        return values / 1000.0
    if "us" in name or "µs" in name:
        return values / 1_000_000.0
    return values


def convert_current_to_ma(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if "ua" in name or "µa" in name or "micro" in name:
        return values / 1000.0
    if "ma" in name:
        return values
    if re.search(r"\ba\b", name) or "(a" in name or "[a" in name:
        return values * 1000.0
    median_abs = float(np.nanmedian(np.abs(values)))
    if median_abs < 1.0:
        return values * 1000.0
    if median_abs > 1000.0:
        return values / 1000.0
    return values


def read_ppk_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, comment="#")
    if dataframe.empty:
        raise ValueError(f"CSV is empty: {path}")
    time_column = find_column(dataframe.columns, TIME_COLUMN_CANDIDATES, "time")
    current_column = find_column(dataframe.columns, CURRENT_COLUMN_CANDIDATES, "current")
    time_raw = pd.to_numeric(dataframe[time_column], errors="coerce").to_numpy(dtype=float)
    current_raw = pd.to_numeric(dataframe[current_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_raw) & np.isfinite(current_raw)
    time_raw = time_raw[valid]
    current_raw = current_raw[valid]
    if len(time_raw) < 10:
        raise ValueError(f"Not enough valid samples in {path}")
    time_s = convert_time_to_s(time_raw, time_column)
    time_s = time_s - time_s[0]
    current_ma = convert_current_to_ma(current_raw, current_column)
    return pd.DataFrame({"time_s": time_s, "current_ma": current_ma})


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values
    return (
        pd.Series(values)
        .rolling(window=samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def detect_sync_midpoint(dataframe: pd.DataFrame) -> float:
    time_s = dataframe["time_s"].to_numpy()
    current_ma = dataframe["current_ma"].to_numpy()
    dt_s = float(np.median(np.diff(time_s)))
    smooth_samples = max(1, round(SYNC_SMOOTH_WINDOW_S / dt_s))
    smooth_current = rolling_mean(current_ma, smooth_samples)
    search_start_s = max(0.0, SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S)
    search_end_s = SYNC_EXPECTED_START_S + SYNC_DURATION_S + SYNC_SEARCH_MARGIN_S
    search_mask = (time_s >= search_start_s) & (time_s <= search_end_s)
    if not np.any(search_mask):
        raise ValueError("Sync-pulse search window contains no samples")
    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]
    baseline_mask = (
        (time_s >= max(0.0, SYNC_EXPECTED_START_S - 2.5))
        & (time_s <= SYNC_EXPECTED_START_S - 0.3)
    )
    baseline = (
        float(np.median(smooth_current[baseline_mask]))
        if np.any(baseline_mask)
        else float(np.percentile(search_current, 20))
    )
    peak = float(np.max(search_current))
    threshold = baseline + 0.45 * (peak - baseline)
    high = search_current >= threshold
    segments: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, is_high in enumerate(high):
        if is_high and start_index is None:
            start_index = index
        elif not is_high and start_index is not None:
            segments.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        segments.append((start_index, len(high) - 1))
    if not segments:
        return float(search_time[int(np.argmax(search_current))])
    expected_mid_s = SYNC_EXPECTED_START_S + SYNC_DURATION_S / 2.0
    def score(segment: tuple[int, int]) -> tuple[float, float]:
        start_i, end_i = segment
        start_s = search_time[start_i]
        end_s = search_time[end_i]
        midpoint_s = (start_s + end_s) / 2.0
        return (
            abs((end_s - start_s) - SYNC_DURATION_S),
            abs(midpoint_s - expected_mid_s),
        )
    best_start, best_end = min(segments, key=score)
    return float((search_time[best_start] + search_time[best_end]) / 2.0)


def select_csv(input_dir: Path, run_id: str | None) -> Path:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    if run_id is None:
        return csv_files[0]
    target = normalize_name(run_id).replace(" ", "")
    for path in csv_files:
        if target in normalize_name(path.stem).replace(" ", ""):
            return path
    raise FileNotFoundError(f"No CSV matching RUN_ID={run_id!r} in {input_dir}")


def calculate_adc_idle_mean(dataframe: pd.DataFrame) -> float:
    time_s = dataframe["aligned_time_s"].to_numpy()
    current_ma = dataframe["current_ma"].to_numpy()
    initial_mask = (
        (time_s >= INITIAL_START_S + WINDOW_TRIM_S)
        & (time_s <= INITIAL_END_S - WINDOW_TRIM_S)
    )
    final_mask = (
        (time_s >= FINAL_START_S + WINDOW_TRIM_S)
        & (time_s <= FINAL_END_S - WINDOW_TRIM_S)
    )
    return (
        float(np.mean(current_ma[initial_mask]))
        + float(np.mean(current_ma[final_mask]))
    ) / 2.0


def create_occupancy_input(time_s: np.ndarray, period_s: float, active_duration_s: float) -> np.ndarray:
    u = np.zeros_like(time_s, dtype=float)
    active_mask = (time_s >= ACTIVE_START_S) & (time_s < ACTIVE_END_S)
    u[active_mask] = active_duration_s / period_s
    return u


def create_pulse_input(time_s: np.ndarray, period_s: float, active_duration_s: float) -> np.ndarray:
    u = np.zeros_like(time_s, dtype=float)
    active_mask = (time_s >= ACTIVE_START_S) & (time_s < ACTIVE_END_S)
    active_time_s = time_s[active_mask] - ACTIVE_START_S
    u[active_mask] = (np.mod(active_time_s, period_s) < active_duration_s).astype(float)
    return u


def simulate_ode(time_s: np.ndarray, input_u: np.ndarray, baseline_ma: float) -> np.ndarray:
    prediction_ma = np.empty_like(time_s, dtype=float)
    prediction_ma[0] = baseline_ma
    for index in range(1, len(time_s)):
        dt_s = time_s[index] - time_s[index - 1]
        if not np.isfinite(dt_s) or dt_s <= 0:
            prediction_ma[index] = prediction_ma[index - 1]
            continue
        target_ma = baseline_ma + ODE_GAIN_MA * input_u[index]
        tau_s = TAU_RISE_S if target_ma >= prediction_ma[index - 1] else TAU_FALL_S
        alpha = 1.0 - np.exp(-dt_s / tau_s)
        prediction_ma[index] = prediction_ma[index - 1] + alpha * (
            target_ma - prediction_ma[index - 1]
        )
    return prediction_ma


def downsample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size)
    step = int(np.ceil(size / max_points))
    return np.arange(0, size, step)


def plot_window(
    time_s: np.ndarray,
    measured_ma: np.ndarray,
    occupancy_prediction_ma: np.ndarray,
    pulse_prediction_ma: np.ndarray,
    xlim: tuple[float, float],
    title: str,
    output_path: Path,
    max_points: int | None = None,
) -> None:
    mask = (time_s >= xlim[0]) & (time_s <= xlim[1])
    x = time_s[mask]
    measured = measured_ma[mask]
    occupancy = occupancy_prediction_ma[mask]
    pulse = pulse_prediction_ma[mask]
    if x.size == 0:
        raise ValueError(f"No samples in plot window {xlim}")
    if max_points is not None:
        idx = downsample_indices(x.size, max_points)
        x, measured, occupancy, pulse = x[idx], measured[idx], occupancy[idx], pulse[idx]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(x, measured, label="Measured", linewidth=1.0)
    axis.plot(x, occupancy, label="Occupancy prediction", linewidth=1.5)
    axis.plot(x, pulse, label="Pulse prediction", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Aligned time after sync midpoint (s)")
    axis.set_ylabel("Current (mA)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze_and_plot_condition(condition: str) -> None:
    csv_path = select_csv(INPUT_DIRS[condition], RUN_ID)
    print(f"\n[{condition}]\n  Reading: {csv_path}")
    dataframe = read_ppk_csv(csv_path)
    sync_mid_s = detect_sync_midpoint(dataframe)
    dataframe["aligned_time_s"] = dataframe["time_s"] - sync_mid_s
    time_s = dataframe["aligned_time_s"].to_numpy()
    measured_raw_ma = dataframe["current_ma"].to_numpy()
    dt_s = float(np.median(np.diff(time_s)))
    display_smooth_samples = max(1, round(MEASURED_DISPLAY_SMOOTH_S / dt_s))
    measured_display_ma = rolling_mean(measured_raw_ma, display_smooth_samples)
    adc_idle_mean_ma = calculate_adc_idle_mean(dataframe)
    workload = WORKLOADS[condition]
    occupancy_u = create_occupancy_input(time_s, workload["period_s"], workload["active_duration_s"])
    pulse_u = create_pulse_input(time_s, workload["period_s"], workload["active_duration_s"])
    occupancy_prediction_ma = simulate_ode(time_s, occupancy_u, adc_idle_mean_ma)
    pulse_prediction_ma = simulate_ode(time_s, pulse_u, adc_idle_mean_ma)

    condition_output_dir = OUTPUT_DIR / condition
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    plot_window(
        time_s, measured_display_ma, occupancy_prediction_ma, pulse_prediction_ma,
        FULL_XLIM,
        f"{condition}: measured vs ADC model predictions ({stem})",
        condition_output_dir / f"{stem}_full_waveform.png",
        max_points=MAX_PLOT_POINTS,
    )
    plot_window(
        time_s, measured_display_ma, occupancy_prediction_ma, pulse_prediction_ma,
        ACTIVE_START_XLIM,
        f"{condition}: active-phase transition ({stem})",
        condition_output_dir / f"{stem}_active_start.png",
        max_points=MAX_PLOT_POINTS,
    )
    detail_xlim = ACTIVE_ZOOM_WINDOWS[condition]

    plot_window(
        time_s, measured_display_ma, occupancy_prediction_ma, pulse_prediction_ma,
        detail_xlim,
        f"{condition}: active-phase waveform detail ({stem})",
        condition_output_dir / f"{stem}_active_zoom.png",
        max_points=None,
    )

    prediction_output = condition_output_dir / f"{stem}_measured_and_predictions.csv"
    pd.DataFrame(
        {
            "aligned_time_s": time_s,
            "measured_raw_ma": measured_raw_ma,
            "measured_display_ma": measured_display_ma,
            "occupancy_input": occupancy_u,
            "pulse_input": pulse_u,
            "occupancy_prediction_ma": occupancy_prediction_ma,
            "pulse_prediction_ma": pulse_prediction_ma,
        }
    ).to_csv(prediction_output, index=False)

    print(f"  ADC idle baseline: {adc_idle_mean_ma:.6f} mA")
    print(f"  Occupancy: {workload['active_duration_s'] / workload['period_s']:.8f}")
    print(f"  Saved plots to: {condition_output_dir}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONDITION_TO_PLOT == "all":
        conditions = list(INPUT_DIRS)
    else:
        if CONDITION_TO_PLOT not in INPUT_DIRS:
            raise ValueError(f"Unknown condition: {CONDITION_TO_PLOT}")
        conditions = [CONDITION_TO_PLOT]
    print(f"Plotting {len(conditions)} workload(s)...")

    for index, condition in enumerate(conditions, start=1):
        print(f"\nWorkload {index}/{len(conditions)}")
        analyze_and_plot_condition(condition)

    print("\nAll requested workload plots are complete.")


if __name__ == "__main__":
    main()