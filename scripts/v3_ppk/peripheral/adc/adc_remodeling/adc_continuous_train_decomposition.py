from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths and experiment timing
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path(
    "data/raw/v3_ppk/peripheral/adc/final_predictioned/adc_remodel_train"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/v3_ppk/peripheral/adc/final_predictioned/adc_remodel_train"
)

SYNC_DURATION_S = 1.0
RECOVERY_IDLE_S = 5.0
INITIAL_IDLE_S = 10.0
ACTIVE_S = 20.0
FINAL_IDLE_S = 10.0

# Phase boundaries after alignment to the sync-pulse START.
# The 1 s sync pulse spans 0.0 to 1.0 s after alignment.
INITIAL_IDLE_START_S = SYNC_DURATION_S + RECOVERY_IDLE_S            # 6.0 s
INITIAL_IDLE_END_S = INITIAL_IDLE_START_S + INITIAL_IDLE_S          # 16.0 s
ACTIVE_START_S = INITIAL_IDLE_END_S                                 # 16.0 s
ACTIVE_END_S = ACTIVE_START_S + ACTIVE_S                            # 36.0 s
FINAL_IDLE_START_S = ACTIVE_END_S                                   # 36.0 s
FINAL_IDLE_END_S = FINAL_IDLE_START_S + FINAL_IDLE_S                # 46.0 s

# Exclude transition edges when calculating plateau means.
INITIAL_IDLE_MARGIN_START_S = 1.0
INITIAL_IDLE_MARGIN_END_S = 1.0
ACTIVE_MARGIN_START_S = 1.0
ACTIVE_MARGIN_END_S = 1.0
FINAL_IDLE_MARGIN_START_S = 1.0
FINAL_IDLE_MARGIN_END_S = 1.0

# Sync search: firmware places sync approximately 3 s after file start.
SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_MARGIN_S = 3.0
SYNC_SMOOTH_WINDOW_S = 0.02

# Average waveform output resolution.
AVERAGE_DT_S = 0.0001  # 10 kS/s


# ---------------------------------------------------------------------------
# CSV loading and column detection
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_time_column(columns: Iterable[str]) -> str:
    normalized = {col: normalize_name(col) for col in columns}
    preferred = (
        "timestamp",
        "time",
        "times",
        "timeus",
        "timems",
        "timestampus",
        "timestampms",
    )

    for target in preferred:
        for col, norm in normalized.items():
            if norm == target:
                return col

    for col, norm in normalized.items():
        if "time" in norm:
            return col

    raise ValueError(
        "Could not detect the time column. Available columns: "
        + ", ".join(map(str, columns))
    )


def detect_current_column(columns: Iterable[str]) -> str:
    normalized = {col: normalize_name(col) for col in columns}
    preferred = (
        "current",
        "currentma",
        "currentua",
        "currenta",
        "currentamps",
    )

    for target in preferred:
        for col, norm in normalized.items():
            if norm == target:
                return col

    for col, norm in normalized.items():
        if "current" in norm:
            return col

    raise ValueError(
        "Could not detect the current column. Available columns: "
        + ", ".join(map(str, columns))
    )


def infer_time_seconds(series: pd.Series, column_name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    norm = normalize_name(column_name)

    if "us" in norm or "micro" in norm:
        return values / 1e6
    if "ms" in norm or "milli" in norm:
        return values / 1e3
    if "ns" in norm or "nano" in norm:
        return values / 1e9

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError("Time column does not contain enough valid values.")

    diffs = np.diff(finite)
    positive_diffs = diffs[diffs > 0]
    median_dt = float(np.median(positive_diffs)) if positive_diffs.size else math.nan
    total_span = float(finite[-1] - finite[0])

    # Heuristic for unlabeled numeric time columns.
    if total_span > 1e7 or median_dt > 1000:
        return values / 1e6
    if total_span > 1e4 or median_dt > 1:
        return values / 1e3
    return values


def infer_current_ma(series: pd.Series, column_name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    norm = normalize_name(column_name)

    if "ua" in norm or "microamp" in norm:
        return values / 1000.0
    if norm.endswith("a") and "ma" not in norm:
        return values * 1000.0

    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("Current column does not contain valid values.")

    median_abs = float(np.median(finite))

    # Typical ESP32 current:
    #   A  -> around 0.05
    #   mA -> around 50
    #   uA -> around 50000
    if median_abs < 1.0:
        return values * 1000.0
    if median_abs > 5000.0:
        return values / 1000.0
    return values


def load_ppk_csv(path: Path) -> pd.DataFrame:
    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    )

    last_error: Exception | None = None
    frame: pd.DataFrame | None = None

    for options in attempts:
        try:
            candidate = pd.read_csv(path, comment="#", **options)
            if candidate.shape[1] >= 2:
                frame = candidate
                break
        except Exception as exc:
            last_error = exc

    if frame is None:
        raise ValueError(f"Could not read {path}: {last_error}")

    time_col = detect_time_column(frame.columns)
    current_col = detect_current_column(frame.columns)

    result = pd.DataFrame(
        {
            "time_s": infer_time_seconds(frame[time_col], time_col),
            "current_ma": infer_current_ma(frame[current_col], current_col),
        }
    )

    result = (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("time_s")
        .drop_duplicates("time_s")
        .reset_index(drop=True)
    )

    if len(result) < 100:
        raise ValueError(f"{path} contains too few valid samples.")

    result["time_s"] -= result["time_s"].iloc[0]
    return result


# ---------------------------------------------------------------------------
# Sync-pulse detection
# ---------------------------------------------------------------------------

def rolling_mean_by_time(
    time_s: np.ndarray,
    current_ma: np.ndarray,
    window_ms: float,
) -> np.ndarray:
    if len(time_s) < 3:
        return current_ma.copy()

    dt = np.median(np.diff(time_s))
    if not np.isfinite(dt) or dt <= 0:
        return current_ma.copy()

    window_samples = max(1, int(round((window_ms / 1000.0) / dt)))
    return (
        pd.Series(current_ma)
        .rolling(window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def detect_sync_start(df: pd.DataFrame) -> tuple[float, float, float]:
    """Detect the 1 s CPU sync pulse from the waveform itself.

    Returns (sync_start_s, sync_end_s, sync_mid_s).

    The expected raw start time is used only to define a broad search region.
    Alignment itself uses the detected rising edge (sync_start_s), not the
    segment midpoint.
    """
    time_s = df["time_s"].to_numpy()
    current_ma = df["current_ma"].to_numpy()

    dt = float(np.median(np.diff(time_s)))
    smooth_samples = max(1, round(SYNC_SMOOTH_WINDOW_S / dt))
    smooth_current = rolling_mean_by_time(
        time_s, current_ma, SYNC_SMOOTH_WINDOW_S * 1000.0
    )

    search_start = max(0.0, SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S)
    search_end = (
        SYNC_EXPECTED_START_S + SYNC_DURATION_S + SYNC_SEARCH_MARGIN_S
    )
    search_mask = (time_s >= search_start) & (time_s <= search_end)
    if not np.any(search_mask):
        raise ValueError("Sync-pulse search window does not contain samples")

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline_mask = (
        (time_s >= max(0.0, SYNC_EXPECTED_START_S - 2.5))
        & (time_s <= SYNC_EXPECTED_START_S - 0.3)
    )
    if np.any(baseline_mask):
        baseline = float(np.median(smooth_current[baseline_mask]))
    else:
        baseline = float(np.percentile(search_current, 20))

    peak = float(np.max(search_current))
    if peak <= baseline:
        raise ValueError("No positive sync pulse found above baseline")

    threshold = baseline + 0.45 * (peak - baseline)
    high = search_current >= threshold

    segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    for idx, is_high in enumerate(high):
        if is_high and seg_start is None:
            seg_start = idx
        elif not is_high and seg_start is not None:
            segments.append((seg_start, idx - 1))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, len(high) - 1))

    if not segments:
        raise ValueError("No threshold-crossing sync segment found")

    expected_mid = SYNC_EXPECTED_START_S + SYNC_DURATION_S / 2.0

    def segment_score(segment: tuple[int, int]) -> tuple[float, float]:
        start_idx, end_idx = segment
        seg_start_s = float(search_time[start_idx])
        seg_end_s = float(search_time[end_idx])
        seg_mid_s = (seg_start_s + seg_end_s) / 2.0
        duration_error = abs((seg_end_s - seg_start_s) - SYNC_DURATION_S)
        timing_error = abs(seg_mid_s - expected_mid)
        return duration_error, timing_error

    best_start_idx, best_end_idx = min(segments, key=segment_score)

    # Refine the detected start using linear interpolation at the upward
    # threshold crossing. This gives a more precise common alignment point.
    if best_start_idx > 0:
        t1 = float(search_time[best_start_idx - 1])
        t2 = float(search_time[best_start_idx])
        y1 = float(search_current[best_start_idx - 1])
        y2 = float(search_current[best_start_idx])

        if y2 != y1:
            sync_start_s = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
        else:
            sync_start_s = t2
    else:
        sync_start_s = float(search_time[best_start_idx])

    # End time is diagnostic only; alignment does not depend on it.
    if best_end_idx + 1 < len(search_time):
        t1 = float(search_time[best_end_idx])
        t2 = float(search_time[best_end_idx + 1])
        y1 = float(search_current[best_end_idx])
        y2 = float(search_current[best_end_idx + 1])

        if y2 != y1:
            sync_end_s = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
        else:
            sync_end_s = t1
    else:
        sync_end_s = float(search_time[best_end_idx])

    sync_mid_s = (sync_start_s + sync_end_s) / 2.0

    return sync_start_s, sync_end_s, sync_mid_s


# ---------------------------------------------------------------------------
# Phase statistics
# ---------------------------------------------------------------------------

def phase_values(
    df: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    mask = (
        (df["aligned_time_s"] >= start_s)
        & (df["aligned_time_s"] < end_s)
    )
    values = df.loc[mask, "current_ma"].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(
            f"No samples found in phase window {start_s:.3f}–{end_s:.3f} s."
        )
    return values


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else math.nan


def analyze_run(
    path: Path,
    i1_ma: float,
    i2_ma: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    df = load_ppk_csv(path)
    sync_start_s, sync_end_s, sync_mid_s = detect_sync_start(df)
    df["aligned_time_s"] = df["time_s"] - sync_start_s

    initial_idle = phase_values(
        df,
        INITIAL_IDLE_START_S + INITIAL_IDLE_MARGIN_START_S,
        INITIAL_IDLE_END_S - INITIAL_IDLE_MARGIN_END_S,
    )
    active = phase_values(
        df,
        ACTIVE_START_S + ACTIVE_MARGIN_START_S,
        ACTIVE_END_S - ACTIVE_MARGIN_END_S,
    )
    final_idle = phase_values(
        df,
        FINAL_IDLE_START_S + FINAL_IDLE_MARGIN_START_S,
        FINAL_IDLE_END_S - FINAL_IDLE_MARGIN_END_S,
    )

    i3_ma = safe_mean(initial_idle)
    i4_ma = safe_mean(active)
    final_idle_ma = safe_mean(final_idle)

    result: dict[str, float | str] = {
        "file": path.name,
        "sync_start_in_raw_s": sync_start_s,
        "sync_end_in_raw_s": sync_end_s,
        "sync_mid_in_raw_s": sync_mid_s,
        "detected_sync_duration_s": sync_end_s - sync_start_s,
        "predicted_active_start_raw_s": sync_start_s + ACTIVE_START_S,
        "predicted_active_end_raw_s": sync_start_s + ACTIVE_END_S,
        "I1_esp_idle_baseline_ma": i1_ma,
        "I2_adc_init_ma": i2_ma,
        "I3_adc_preconditioned_idle_ma": i3_ma,
        "I4_continuous_adc_active_ma": i4_ma,
        "adc_init_overhead_I2_minus_I1_ma": i2_ma - i1_ma,
        "first_read_shift_I3_minus_I2_ma": i3_ma - i2_ma,
        "continuous_read_overhead_I4_minus_I3_ma": i4_ma - i3_ma,
        "total_active_overhead_I4_minus_I1_ma": i4_ma - i1_ma,
        "initial_idle_std_ma": safe_std(initial_idle),
        "active_plateau_std_ma": safe_std(active),
        "final_idle_mean_ma": final_idle_ma,
        "final_minus_initial_idle_ma": final_idle_ma - i3_ma,
        "initial_idle_samples": int(initial_idle.size),
        "active_plateau_samples": int(active.size),
        "final_idle_samples": int(final_idle.size),
    }

    return df, result


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    numeric = results.select_dtypes(include=[np.number])
    rows = []

    for column in numeric.columns:
        rows.append(
            {
                "metric": column,
                "mean": numeric[column].mean(),
                "std": numeric[column].std(ddof=1),
                "min": numeric[column].min(),
                "max": numeric[column].max(),
                "median": numeric[column].median(),
            }
        )

    return pd.DataFrame(rows)


def build_average_waveform(
    aligned_runs: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    grid = np.arange(
        -0.5,
        FINAL_IDLE_END_S,
        AVERAGE_DT_S,
        dtype=float,
    )

    interpolated = []
    for _, df in aligned_runs:
        t = df["aligned_time_s"].to_numpy(dtype=float)
        y = df["current_ma"].to_numpy(dtype=float)

        valid = (grid >= t[0]) & (grid <= t[-1])
        run_values = np.full(grid.shape, np.nan, dtype=float)
        run_values[valid] = np.interp(grid[valid], t, y)
        interpolated.append(run_values)

    matrix = np.vstack(interpolated)

    return pd.DataFrame(
        {
            "aligned_time_s": grid,
            "mean_current_ma": np.nanmean(matrix, axis=0),
            "std_current_ma": np.nanstd(matrix, axis=0, ddof=1),
            "n_runs": np.sum(np.isfinite(matrix), axis=0),
        }
    )


def save_run_plot(
    name: str,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["aligned_time_s"], df["current_ma"], linewidth=0.7)

    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="Sync start")
    ax.axvline(INITIAL_IDLE_START_S, linestyle="--", linewidth=1.0)
    ax.axvline(ACTIVE_START_S, linestyle="--", linewidth=1.0)
    ax.axvline(ACTIVE_END_S, linestyle="--", linewidth=1.0)
    ax.axvline(FINAL_IDLE_END_S, linestyle="--", linewidth=1.0)

    ax.set_xlim(-0.5, FINAL_IDLE_END_S)
    ax.set_xlabel("Aligned time from sync start [s]")
    ax.set_ylabel("Current [mA]")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{Path(name).stem}_aligned.png", dpi=160)
    plt.close(fig)


def save_average_plot(
    average: pd.DataFrame,
    output_dir: Path,
) -> None:
    t = average["aligned_time_s"].to_numpy()
    mean = average["mean_current_ma"].to_numpy()
    std = average["std_current_ma"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, mean, linewidth=1.0, label="10-run mean")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, label="±1 SD")

    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="Sync start")
    ax.axvline(INITIAL_IDLE_START_S, linestyle="--", linewidth=1.0)
    ax.axvline(ACTIVE_START_S, linestyle="--", linewidth=1.0)
    ax.axvline(ACTIVE_END_S, linestyle="--", linewidth=1.0)
    ax.axvline(FINAL_IDLE_END_S, linestyle="--", linewidth=1.0)

    ax.set_xlim(-0.5, FINAL_IDLE_END_S)
    ax.set_xlabel("Aligned time from sync start [s]")
    ax.set_ylabel("Current [mA]")
    ax.set_title("ADC continuous-read train data: aligned average")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "adc_continuous_train_aligned_average.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align and decompose ADC continuous-read PPK2 train data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw CSV runs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for processed outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--i1",
        type=float,
        required=True,
        help="Fixed I1 ESP idle baseline [mA] from the previous ADC validation.",
    )
    parser.add_argument(
        "--i2",
        type=float,
        required=True,
        help="Fixed I2 ADC-init level [mA] from the previous ADC validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    csv_paths = sorted(
        p for p in input_dir.glob("*.csv")
        if p.is_file()
    )

    if not csv_paths:
        print(f"ERROR: no CSV files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    run_plot_dir = output_dir / "aligned_run_plots"
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    aligned_runs: list[tuple[str, pd.DataFrame]] = []
    results: list[dict[str, float | str]] = []

    print(f"Found {len(csv_paths)} CSV file(s).")

    for path in csv_paths:
        try:
            aligned_df, result = analyze_run(
                path=path,
                i1_ma=args.i1,
                i2_ma=args.i2,
            )
        except Exception as exc:
            print(f"ERROR processing {path.name}: {exc}", file=sys.stderr)
            return 1

        aligned_runs.append((path.name, aligned_df))
        results.append(result)
        save_run_plot(path.name, aligned_df, run_plot_dir)

        print(
            f"{path.name}: "
            f"I3={result['I3_adc_preconditioned_idle_ma']:.6f} mA, "
            f"I4={result['I4_continuous_adc_active_ma']:.6f} mA, "
            f"ΔI={result['continuous_read_overhead_I4_minus_I3_ma']:.6f} mA"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        output_dir / "adc_continuous_train_decomposition_by_run.csv",
        index=False,
    )

    summary_df = make_summary(results_df)
    summary_df.to_csv(
        output_dir / "adc_continuous_train_decomposition_summary.csv",
        index=False,
    )

    average_df = build_average_waveform(aligned_runs)
    average_df.to_csv(
        output_dir / "adc_continuous_train_aligned_average.csv",
        index=False,
    )
    save_average_plot(average_df, output_dir)

    print("\nSaved:")
    print(
        output_dir / "adc_continuous_train_decomposition_by_run.csv"
    )
    print(
        output_dir / "adc_continuous_train_decomposition_summary.csv"
    )
    print(
        output_dir / "adc_continuous_train_aligned_average.csv"
    )
    print(
        output_dir / "adc_continuous_train_aligned_average.png"
    )
    print(run_plot_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())