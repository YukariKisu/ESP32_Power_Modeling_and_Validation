from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# User settings
# ============================================================

RAW_ROOT = Path(
    "data/raw/v3_ppk/peripheral/adc/final_predictioned"
)

INPUT_DIRS = {
    "adc_single_1ms": (
        RAW_ROOT / "adc_periodic_single" / "adc_single_1ms"
    ),
    "adc_single_10ms": (
        RAW_ROOT / "adc_periodic_single" / "adc_single_10ms"
    ),
    "adc_single_100ms": (
        RAW_ROOT / "adc_periodic_single" / "adc_single_100ms"
    ),
    "adc_burst_100ms_10samples": (
        RAW_ROOT / "adc_periodic_burst"
        / "adc_burst_100ms_10samples"
    ),
    "adc_burst_100ms_100samples": (
        RAW_ROOT / "adc_periodic_burst"
        / "adc_burst_100ms_100samples"
    ),
    "adc_burst_100ms_1000samples": (
        RAW_ROOT / "adc_periodic_burst"
        / "adc_burst_100ms_1000samples"
    ),
}

OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "active_high_level"
)


# ============================================================
# Analysis windows after sync-pulse midpoint
# ============================================================

INITIAL_START_S = 5.5
INITIAL_END_S = 15.5

ACTIVE_START_S = 15.5
ACTIVE_END_S = 35.5

FINAL_START_S = 35.5
FINAL_END_S = 45.5

# Remove phase-edge transients.
WINDOW_TRIM_S = 0.2

HIGH_PERCENTILE = 99.9


# ============================================================
# Sync-pulse settings
# ============================================================

SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_MARGIN_S = 1.5
SYNC_SMOOTH_WINDOW_S = 0.02


TIME_COLUMN_CANDIDATES = (
    "time",
    "timestamp",
    "timestamp_s",
    "time_s",
    "time (s)",
    "time[s]",
    "timestamp(ms)",
)

CURRENT_COLUMN_CANDIDATES = (
    "current",
    "current_ma",
    "current (ma)",
    "current[ma]",
    "current_ua",
    "current (ua)",
    "current[ua]",
    "current_a",
    "current (a)",
    "current[a]",
)


@dataclass(frozen=True)
class RunResult:
    condition: str
    run_id: str
    source_file: str
    sync_mid_s: float

    initial_idle_mean_ma: float
    final_idle_mean_ma: float
    adc_idle_mean_ma: float

    active_mean_ma: float
    active_max_ma: float
    active_p99_9_ma: float
    active_top_0_1pct_mean_ma: float

    active_mean_delta_ma: float
    active_max_delta_ma: float
    active_p99_9_delta_ma: float
    active_top_0_1pct_mean_delta_ma: float

    active_sample_count: int
    top_0_1pct_sample_count: int


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    kind: str,
) -> str:
    normalized = {
        normalize_name(column): column
        for column in columns
    }

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

    raise ValueError(
        f"Could not find {kind} column in columns: {list(columns)}"
    )


def convert_time_to_s(
    values: np.ndarray,
    column_name: str,
) -> np.ndarray:
    name = normalize_name(column_name)

    if "ms" in name:
        return values / 1000.0

    if "us" in name or "µs" in name:
        return values / 1_000_000.0

    return values


def convert_current_to_ma(
    values: np.ndarray,
    column_name: str,
) -> np.ndarray:
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

    time_column = find_column(
        dataframe.columns,
        TIME_COLUMN_CANDIDATES,
        "time",
    )
    current_column = find_column(
        dataframe.columns,
        CURRENT_COLUMN_CANDIDATES,
        "current",
    )

    time_raw = pd.to_numeric(
        dataframe[time_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    current_raw = pd.to_numeric(
        dataframe[current_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(time_raw) & np.isfinite(current_raw)
    time_raw = time_raw[valid]
    current_raw = current_raw[valid]

    if len(time_raw) < 10:
        raise ValueError(f"Not enough valid samples in {path}")

    time_s = convert_time_to_s(time_raw, time_column)
    time_s = time_s - time_s[0]

    current_ma = convert_current_to_ma(
        current_raw,
        current_column,
    )

    return pd.DataFrame(
        {
            "time_s": time_s,
            "current_ma": current_ma,
        }
    )


def rolling_mean(
    values: np.ndarray,
    samples: int,
) -> np.ndarray:
    """Fast centered rolling mean using pandas' C implementation."""
    samples = max(1, int(samples))

    if samples == 1:
        return values

    return (
        pd.Series(values)
        .rolling(
            window=samples,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=float)
    )


def detect_sync_midpoint(
    dataframe: pd.DataFrame,
) -> float:
    time_s = dataframe["time_s"].to_numpy()
    current_ma = dataframe["current_ma"].to_numpy()

    dt_s = float(np.median(np.diff(time_s)))
    smooth_samples = max(
        1,
        round(SYNC_SMOOTH_WINDOW_S / dt_s),
    )
    smooth_current = rolling_mean(
        current_ma,
        smooth_samples,
    )

    search_start_s = max(
        0.0,
        SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S,
    )
    search_end_s = (
        SYNC_EXPECTED_START_S
        + SYNC_DURATION_S
        + SYNC_SEARCH_MARGIN_S
    )

    search_mask = (
        (time_s >= search_start_s)
        & (time_s <= search_end_s)
    )

    if not np.any(search_mask):
        raise ValueError(
            "Sync-pulse search window contains no samples"
        )

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline_mask = (
        (
            time_s
            >= max(0.0, SYNC_EXPECTED_START_S - 2.5)
        )
        & (
            time_s
            <= SYNC_EXPECTED_START_S - 0.3
        )
    )

    if np.any(baseline_mask):
        baseline = float(
            np.median(smooth_current[baseline_mask])
        )
    else:
        baseline = float(
            np.percentile(search_current, 20)
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

    expected_mid_s = (
        SYNC_EXPECTED_START_S + SYNC_DURATION_S / 2.0
    )

    def score(
        segment: tuple[int, int],
    ) -> tuple[float, float]:
        start_i, end_i = segment
        start_s = search_time[start_i]
        end_s = search_time[end_i]
        midpoint_s = (start_s + end_s) / 2.0

        duration_error_s = abs(
            (end_s - start_s) - SYNC_DURATION_S
        )
        timing_error_s = abs(
            midpoint_s - expected_mid_s
        )

        return duration_error_s, timing_error_s

    best_start, best_end = min(segments, key=score)

    return float(
        (
            search_time[best_start]
            + search_time[best_end]
        )
        / 2.0
    )


def infer_run_id(path: Path) -> str:
    match = re.search(
        r"(run[_-]?\d+|\d+)$",
        path.stem,
        flags=re.IGNORECASE,
    )

    return match.group(1) if match else path.stem


def phase_values(
    dataframe: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    mask = (
        (
            dataframe["aligned_time_s"]
            >= start_s + WINDOW_TRIM_S
        )
        & (
            dataframe["aligned_time_s"]
            <= end_s - WINDOW_TRIM_S
        )
    )

    values = dataframe.loc[
        mask,
        "current_ma",
    ].to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError(
            f"No valid samples in phase {start_s}–{end_s} s"
        )

    return values


def analyze_run(
    condition: str,
    path: Path,
) -> RunResult:
    dataframe = read_ppk_csv(path)

    sync_mid_s = detect_sync_midpoint(dataframe)
    dataframe["aligned_time_s"] = (
        dataframe["time_s"] - sync_mid_s
    )

    initial_values = phase_values(
        dataframe,
        INITIAL_START_S,
        INITIAL_END_S,
    )
    active_values = phase_values(
        dataframe,
        ACTIVE_START_S,
        ACTIVE_END_S,
    )
    final_values = phase_values(
        dataframe,
        FINAL_START_S,
        FINAL_END_S,
    )

    initial_mean_ma = float(np.mean(initial_values))
    final_mean_ma = float(np.mean(final_values))

    # Same ADC idle definition as the baseline decomposition:
    # average of the initial and final idle levels.
    adc_idle_mean_ma = (
        initial_mean_ma + final_mean_ma
    ) / 2.0

    active_mean_ma = float(np.mean(active_values))
    active_max_ma = float(np.max(active_values))

    active_p99_9_ma = float(
        np.percentile(
            active_values,
            HIGH_PERCENTILE,
        )
    )

    top_mask = active_values >= active_p99_9_ma
    top_values = active_values[top_mask]

    active_top_0_1pct_mean_ma = float(
        np.mean(top_values)
    )

    return RunResult(
        condition=condition,
        run_id=infer_run_id(path),
        source_file=str(path),
        sync_mid_s=sync_mid_s,

        initial_idle_mean_ma=initial_mean_ma,
        final_idle_mean_ma=final_mean_ma,
        adc_idle_mean_ma=adc_idle_mean_ma,

        active_mean_ma=active_mean_ma,
        active_max_ma=active_max_ma,
        active_p99_9_ma=active_p99_9_ma,
        active_top_0_1pct_mean_ma=(
            active_top_0_1pct_mean_ma
        ),

        active_mean_delta_ma=(
            active_mean_ma - adc_idle_mean_ma
        ),
        active_max_delta_ma=(
            active_max_ma - adc_idle_mean_ma
        ),
        active_p99_9_delta_ma=(
            active_p99_9_ma - adc_idle_mean_ma
        ),
        active_top_0_1pct_mean_delta_ma=(
            active_top_0_1pct_mean_ma
            - adc_idle_mean_ma
        ),

        active_sample_count=int(active_values.size),
        top_0_1pct_sample_count=int(top_values.size),
    )


def summarize_results(
    run_results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "adc_idle_mean_ma",
        "active_mean_ma",
        "active_max_ma",
        "active_p99_9_ma",
        "active_top_0_1pct_mean_ma",
        "active_mean_delta_ma",
        "active_max_delta_ma",
        "active_p99_9_delta_ma",
        "active_top_0_1pct_mean_delta_ma",
    ]

    rows: list[dict[str, object]] = []

    for condition, group in run_results.groupby(
        "condition",
        sort=False,
    ):
        row: dict[str, object] = {
            "condition": condition,
            "n_runs": len(group),
        }

        for column in metric_columns:
            row[f"{column}_mean"] = float(
                group[column].mean()
            )
            row[f"{column}_sd"] = float(
                group[column].std(ddof=1)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    failures: list[dict[str, str]] = []

    for condition, input_dir in INPUT_DIRS.items():
        csv_files = sorted(input_dir.glob("*.csv"))

        if not csv_files:
            failures.append(
                {
                    "condition": condition,
                    "source_file": str(input_dir),
                    "error": "No CSV files found",
                }
            )
            continue

        print(
            f"\n[{condition}] "
            f"{len(csv_files)} CSV file(s)"
        )

        for file_index, path in enumerate(csv_files, start=1):
            print(
                f"  Processing {file_index}/{len(csv_files)}: "
                f"{path.name}",
                flush=True,
            )

            try:
                results.append(
                    analyze_run(condition, path)
                )
            except Exception as error:
                failures.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(error),
                    }
                )
                print(
                    f"    FAILED: {error}",
                    flush=True,
                )

    if not results:
        raise RuntimeError(
            "No runs were analyzed successfully. "
            "Check RAW_ROOT, folder names, and CSV columns."
        )

    run_results = pd.DataFrame(
        [result.__dict__ for result in results]
    )

    condition_order = list(INPUT_DIRS)
    run_results["condition"] = pd.Categorical(
        run_results["condition"],
        categories=condition_order,
        ordered=True,
    )
    run_results = run_results.sort_values(
        ["condition", "run_id"]
    ).reset_index(drop=True)

    summary = summarize_results(run_results)

    run_output = OUTPUT_DIR / "adc_active_high_level_per_run.csv"
    summary_output = (
        OUTPUT_DIR / "adc_active_high_level_summary.csv"
    )
    failure_output = OUTPUT_DIR / "analysis_failures.csv"

    run_results.to_csv(run_output, index=False)
    summary.to_csv(summary_output, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            failure_output,
            index=False,
        )

    display_columns = [
        "condition",
        "n_runs",
        "adc_idle_mean_ma_mean",
        "active_p99_9_ma_mean",
        "active_p99_9_delta_ma_mean",
        "active_top_0_1pct_mean_ma_mean",
        "active_top_0_1pct_mean_delta_ma_mean",
    ]

    print("\nADC active high-level summary")
    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved per-run results:\n  {run_output}")
    print(f"Saved condition summary:\n  {summary_output}")

    if failures:
        print(f"Saved failures:\n  {failure_output}")


if __name__ == "__main__":
    main()