from __future__ import annotations

import math
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
        RAW_ROOT
        / "adc_periodic_single"
        / "adc_single_1ms"
    ),
    "adc_single_10ms": (
        RAW_ROOT
        / "adc_periodic_single"
        / "adc_single_10ms"
    ),
    "adc_single_100ms": (
        RAW_ROOT
        / "adc_periodic_single"
        / "adc_single_100ms"
    ),
    "adc_burst_100ms_10samples": (
        RAW_ROOT
        / "adc_periodic_burst"
        / "adc_burst_100ms_10samples"
    ),
    "adc_burst_100ms_100samples": (
        RAW_ROOT
        / "adc_periodic_burst"
        / "adc_burst_100ms_100samples"
    ),
    "adc_burst_100ms_1000samples": (
        RAW_ROOT
        / "adc_periodic_burst"
        / "adc_burst_100ms_1000samples"
    ),
}

OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "mae_input_definitions"
)


# ============================================================
# CPU-trained first-order ODE
# ============================================================

# Replace these only if a different final CPU model is selected.
ODE_GAIN_MA = 20.1888
ODE_TAU_RISE_S = 0.00049
ODE_TAU_FALL_S = 0.00049


# ============================================================
# ADC input definitions
# ============================================================

ACTIVE_DURATION_S = 20.0

# Measured / representative duration of one ADC activity block.
#
# Periodic single:
#   one adc1_get_raw() call, approximately 44 us.
#
# Periodic burst:
#   measured total burst durations.
EVENT_DURATION_S = {
    "adc_single_1ms": 44e-6,
    "adc_single_10ms": 44e-6,
    "adc_single_100ms": 44e-6,

    "adc_burst_100ms_10samples": 537e-6,
    "adc_burst_100ms_100samples": 4396e-6,
    "adc_burst_100ms_1000samples": 43984e-6,
}

EVENT_INTERVAL_S = {
    "adc_single_1ms": 0.001,
    "adc_single_10ms": 0.010,
    "adc_single_100ms": 0.100,

    "adc_burst_100ms_10samples": 0.100,
    "adc_burst_100ms_100samples": 0.100,
    "adc_burst_100ms_1000samples": 0.100,
}

# Constant occupancy input:
# u = ADC activity duration / repetition interval
OCCUPANCY_U = {
    condition: EVENT_DURATION_S[condition] / EVENT_INTERVAL_S[condition]
    for condition in EVENT_DURATION_S
}


# ============================================================
# Analysis windows after sync-pulse midpoint
# ============================================================

INITIAL_START_S = 5.5
INITIAL_END_S = 15.5

ACTIVE_START_S = 15.5
ACTIVE_END_S = 35.5

FINAL_START_S = 35.5
FINAL_END_S = 45.5

WINDOW_TRIM_S = 0.2

# 10 us preserves the 44 us single-read pulse with several samples.
UNIFORM_DT_S = 0.00001


# ============================================================
# Sync pulse settings
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
    input_definition: str
    run_id: str
    source_file: str

    sync_mid_s: float

    initial_mean_ma: float
    final_mean_ma: float
    adc_idle_mean_ma: float

    measured_active_delta_mean_ma: float
    predicted_active_delta_mean_ma: float

    mae_ma: float
    me_ma: float
    rmse_ma: float


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
        candidate_normalized = normalize_name(candidate)

        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    for column in columns:
        column_normalized = normalize_name(column)

        if kind == "time" and "time" in column_normalized:
            return column

        if kind == "current" and "current" in column_normalized:
            return column

    raise ValueError(
        f"Could not find {kind} column in columns: "
        f"{list(columns)}"
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
    dataframe = pd.read_csv(
        path,
        comment="#",
    )

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

    valid = (
        np.isfinite(time_raw)
        & np.isfinite(current_raw)
    )

    time_raw = time_raw[valid]
    current_raw = current_raw[valid]

    if len(time_raw) < 10:
        raise ValueError(
            f"Not enough valid samples in {path}"
        )

    time_s = convert_time_to_s(
        time_raw,
        time_column,
    )

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
    samples = max(1, int(samples))

    if samples == 1:
        return values

    kernel = np.ones(
        samples,
        dtype=float,
    ) / samples

    return np.convolve(
        values,
        kernel,
        mode="same",
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
            "Sync-pulse search window does not contain samples"
        )

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline_mask = (
        (
            time_s
            >= max(
                0.0,
                SYNC_EXPECTED_START_S - 2.5,
            )
        )
        & (
            time_s
            <= SYNC_EXPECTED_START_S - 0.3
        )
    )

    if np.any(baseline_mask):
        baseline = float(
            np.median(
                smooth_current[baseline_mask]
            )
        )
    else:
        baseline = float(
            np.percentile(
                search_current,
                20,
            )
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
            segments.append(
                (start_index, index - 1)
            )
            start_index = None

    if start_index is not None:
        segments.append(
            (start_index, len(high) - 1)
        )

    if not segments:
        peak_index = int(np.argmax(search_current))
        return float(search_time[peak_index])

    expected_mid_s = (
        SYNC_EXPECTED_START_S
        + SYNC_DURATION_S / 2.0
    )

    def score(
        segment: tuple[int, int],
    ) -> tuple[float, float]:
        segment_start_index, segment_end_index = segment

        segment_start_s = search_time[
            segment_start_index
        ]
        segment_end_s = search_time[
            segment_end_index
        ]

        segment_mid_s = (
            segment_start_s + segment_end_s
        ) / 2.0

        duration_error_s = abs(
            (
                segment_end_s
                - segment_start_s
            )
            - SYNC_DURATION_S
        )

        timing_error_s = abs(
            segment_mid_s - expected_mid_s
        )

        return duration_error_s, timing_error_s

    best_start_index, best_end_index = min(
        segments,
        key=score,
    )

    return float(
        (
            search_time[best_start_index]
            + search_time[best_end_index]
        )
        / 2.0
    )


def infer_run_id(path: Path) -> str:
    match = re.search(
        r"(run[_-]?\d+|\d+)$",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return path.stem


def make_occupancy_input(
    condition: str,
    time_s: np.ndarray,
) -> np.ndarray:
    return np.full_like(
        time_s,
        OCCUPANCY_U[condition],
        dtype=float,
    )


def make_pulse_input(
    condition: str,
    time_s: np.ndarray,
) -> np.ndarray:
    interval_s = EVENT_INTERVAL_S[condition]
    event_duration_s = EVENT_DURATION_S[condition]

    phase_s = np.mod(
        time_s,
        interval_s,
    )

    return (
        phase_s < event_duration_s
    ).astype(float)


def simulate_first_order_ode(
    input_u: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    prediction_delta_ma = np.zeros_like(
        input_u,
        dtype=float,
    )

    for index in range(
        1,
        len(input_u),
    ):
        target_delta_ma = (
            ODE_GAIN_MA
            * input_u[index]
        )

        tau_s = (
            ODE_TAU_RISE_S
            if target_delta_ma
            >= prediction_delta_ma[index - 1]
            else ODE_TAU_FALL_S
        )

        alpha = 1.0 - math.exp(
            -dt_s / tau_s
        )

        prediction_delta_ma[index] = (
            prediction_delta_ma[index - 1]
            + alpha
            * (
                target_delta_ma
                - prediction_delta_ma[index - 1]
            )
        )

    return prediction_delta_ma


def calculate_phase_mean(
    dataframe: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> float:
    phase_mask = (
        (
            dataframe["aligned_time_s"]
            >= start_s + WINDOW_TRIM_S
        )
        & (
            dataframe["aligned_time_s"]
            <= end_s - WINDOW_TRIM_S
        )
    )

    if not np.any(phase_mask):
        raise ValueError(
            f"No samples in phase {start_s}–{end_s} s"
        )

    return float(
        dataframe.loc[
            phase_mask,
            "current_ma",
        ].mean()
    )


def active_uniform_measurement(
    dataframe: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    initial_mean_ma = calculate_phase_mean(
        dataframe,
        INITIAL_START_S,
        INITIAL_END_S,
    )

    final_mean_ma = calculate_phase_mean(
        dataframe,
        FINAL_START_S,
        FINAL_END_S,
    )

    # Same baseline definition as the final ADC decomposition:
    # ADC idle = mean of initial and final idle.
    adc_idle_mean_ma = (
        initial_mean_ma + final_mean_ma
    ) / 2.0

    active_mask = (
        (
            dataframe["aligned_time_s"]
            >= ACTIVE_START_S
        )
        & (
            dataframe["aligned_time_s"]
            <= ACTIVE_END_S
        )
    )

    active_time_s = (
        dataframe.loc[
            active_mask,
            "aligned_time_s",
        ].to_numpy()
        - ACTIVE_START_S
    )

    active_current_ma = dataframe.loc[
        active_mask,
        "current_ma",
    ].to_numpy()

    measured_delta_ma = (
        active_current_ma
        - adc_idle_mean_ma
    )

    if len(active_time_s) < 10:
        raise ValueError(
            "Active window does not contain enough samples"
        )

    uniform_time_s = np.arange(
        0.0,
        ACTIVE_DURATION_S,
        UNIFORM_DT_S,
    )

    uniform_measured_delta_ma = np.interp(
        uniform_time_s,
        active_time_s,
        measured_delta_ma,
    )

    return (
        uniform_time_s,
        uniform_measured_delta_ma,
        initial_mean_ma,
        final_mean_ma,
        adc_idle_mean_ma,
    )


def analyze_one_file(
    condition: str,
    path: Path,
) -> list[RunResult]:
    dataframe = read_ppk_csv(path)

    sync_mid_s = detect_sync_midpoint(
        dataframe
    )

    dataframe["aligned_time_s"] = (
        dataframe["time_s"]
        - sync_mid_s
    )

    (
        time_s,
        measured_delta_ma,
        initial_mean_ma,
        final_mean_ma,
        adc_idle_mean_ma,
    ) = active_uniform_measurement(
        dataframe
    )

    run_id = infer_run_id(path)

    results: list[RunResult] = []

    for input_definition, input_builder in (
        ("occupancy", make_occupancy_input),
        ("pulse", make_pulse_input),
    ):
        input_u = input_builder(
            condition,
            time_s,
        )

        predicted_delta_ma = simulate_first_order_ode(
            input_u,
            UNIFORM_DT_S,
        )

        # Sign convention:
        # positive error means overprediction.
        error_ma = (
            predicted_delta_ma
            - measured_delta_ma
        )

        results.append(
            RunResult(
                condition=condition,
                input_definition=input_definition,
                run_id=run_id,
                source_file=str(path),

                sync_mid_s=sync_mid_s,

                initial_mean_ma=initial_mean_ma,
                final_mean_ma=final_mean_ma,
                adc_idle_mean_ma=adc_idle_mean_ma,

                measured_active_delta_mean_ma=float(
                    np.mean(measured_delta_ma)
                ),
                predicted_active_delta_mean_ma=float(
                    np.mean(predicted_delta_ma)
                ),

                mae_ma=float(
                    np.mean(
                        np.abs(error_ma)
                    )
                ),
                me_ma=float(
                    np.mean(error_ma)
                ),
                rmse_ma=float(
                    np.sqrt(
                        np.mean(
                            error_ma * error_ma
                        )
                    )
                ),
            )
        )

    return results


def collect_csv_files(
    input_dir: Path,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    return sorted(
        path
        for path in input_dir.rglob("*.csv")
        if path.is_file()
    )


def results_to_frame(
    results: list[RunResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            result.__dict__
            for result in results
        ]
    )


def summarize(
    per_run_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = per_run_dataframe.groupby(
        [
            "condition",
            "input_definition",
        ],
        sort=True,
    )

    for (
        condition,
        input_definition,
    ), group in grouped:
        rows.append(
            {
                "condition": condition,
                "input_definition": input_definition,
                "n_runs": int(len(group)),

                "mae_mean_ma": float(
                    group["mae_ma"].mean()
                ),
                "mae_std_ma": float(
                    group["mae_ma"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
                "mae_min_ma": float(
                    group["mae_ma"].min()
                ),
                "mae_max_ma": float(
                    group["mae_ma"].max()
                ),

                "me_mean_ma": float(
                    group["me_ma"].mean()
                ),
                "me_std_ma": float(
                    group["me_ma"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,

                "rmse_mean_ma": float(
                    group["rmse_ma"].mean()
                ),
                "rmse_std_ma": float(
                    group["rmse_ma"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,

                "measured_delta_mean_ma": float(
                    group[
                        "measured_active_delta_mean_ma"
                    ].mean()
                ),
                "measured_delta_std_ma": float(
                    group[
                        "measured_active_delta_mean_ma"
                    ].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,

                "predicted_delta_mean_ma": float(
                    group[
                        "predicted_active_delta_mean_ma"
                    ].mean()
                ),
                "predicted_delta_std_ma": float(
                    group[
                        "predicted_active_delta_mean_ma"
                    ].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_input_definition_table() -> pd.DataFrame:
    rows = []

    for condition in INPUT_DIRS:
        rows.append(
            {
                "condition": condition,
                "event_duration_s":
                    EVENT_DURATION_S[condition],
                "event_interval_s":
                    EVENT_INTERVAL_S[condition],
                "occupancy_u":
                    OCCUPANCY_U[condition],
                "cpu_gain_scaled_occupancy_ma":
                    ODE_GAIN_MA
                    * OCCUPANCY_U[condition],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    all_results: list[RunResult] = []
    errors: list[dict[str, str]] = []

    for condition, input_dir in INPUT_DIRS.items():
        for path in collect_csv_files(
            input_dir
        ):
            try:
                print(
                    f"Processing {condition}: {path}",
                    flush=True,
                )

                all_results.extend(
                    analyze_one_file(
                        condition,
                        path,
                    )
                )

            except Exception as exception:
                errors.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(exception),
                    }
                )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not all_results:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )

        raise SystemExit(
            "No runs could be analyzed. See errors.csv."
        )

    per_run_dataframe = results_to_frame(
        all_results
    )

    summary_dataframe = summarize(
        per_run_dataframe
    )

    input_definition_dataframe = (
        build_input_definition_table()
    )

    per_run_path = (
        OUTPUT_DIR / "per_run_metrics.csv"
    )
    summary_path = (
        OUTPUT_DIR / "summary_metrics.csv"
    )
    input_definition_path = (
        OUTPUT_DIR / "input_definitions.csv"
    )

    per_run_dataframe.to_csv(
        per_run_path,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    input_definition_dataframe.to_csv(
        input_definition_path,
        index=False,
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )

    print(
        f"Analyzed validation rows: "
        f"{len(per_run_dataframe)}"
    )
    print(
        f"Skipped files: {len(errors)}"
    )
    print(f"Wrote: {per_run_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {input_definition_path}")


if __name__ == "__main__":
    main()