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
    "ppk_vs_scope_transition_rmse_50pct_aligned"
)


# ============================================================
# Model parameter sets
# ============================================================

# Keep Delta I fixed so this isolates the effect of tau only.
MODEL_PARAMETERS = {
    "ppk2": {
        "delta_i_ma": 12.943,
        "tau_rise_s": 0.000579,
        "tau_fall_s": 0.000579,
    },
    "scope": {
        "delta_i_ma": 12.943,
        "tau_rise_s": 0.000440,
        "tau_fall_s": 0.000440,
    },
}


# ============================================================
# Workload definitions
# ============================================================

ACTIVE_DURATION_S = 20.0

EVENT_DURATION_S = {
    "adc_burst_100ms_100samples": 4397e-6,
    "adc_burst_100ms_1000samples": 43987e-6,
}

EVENT_INTERVAL_S = {
    "adc_burst_100ms_100samples": 0.100,
    "adc_burst_100ms_1000samples": 0.100,
}


# ============================================================
# Transition-alignment settings
# ============================================================

# Search for the measured/predicted 50% crossing around each nominal edge.
CROSSING_SEARCH_S = 0.0025

# Compare waveform shape on a common relative-time axis after 50% alignment.
ALIGN_PRE_S = 0.0015
ALIGN_POST_S = 0.0015

# Small smoothing used only for measured 50% crossing detection.
CROSSING_SMOOTH_S = 0.00010

# Ignore event edges too close to the beginning/end of the active window.
ACTIVE_EDGE_GUARD_S = 0.005

# Uniform grid used by the original validation.
UNIFORM_DT_S = 0.00001


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
class TransitionResult:
    condition: str
    model_name: str
    transition_type: str
    run_id: str
    source_file: str

    n_transitions: int
    n_samples_in_windows: int

    model_delta_i_ma: float
    model_tau_ms: float

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
    delta_i_ma: float,
    tau_rise_s: float,
    tau_fall_s: float,
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
            delta_i_ma
            * input_u[index]
        )

        tau_s = (
            tau_rise_s
            if target_delta_ma
            >= prediction_delta_ma[index - 1]
            else tau_fall_s
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
        adc_idle_mean_ma,
    )


def nominal_transition_times(
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return nominal rise and fall edge times inside the 20 s active window.
    """
    interval_s = EVENT_INTERVAL_S[condition]
    duration_s = EVENT_DURATION_S[condition]

    rise_times = np.arange(
        0.0,
        ACTIVE_DURATION_S,
        interval_s,
    )

    fall_times = rise_times + duration_s

    rise_times = rise_times[
        (rise_times >= ACTIVE_EDGE_GUARD_S)
        & (
            rise_times
            <= ACTIVE_DURATION_S - ACTIVE_EDGE_GUARD_S
        )
    ]

    fall_times = fall_times[
        (fall_times >= ACTIVE_EDGE_GUARD_S)
        & (
            fall_times
            <= ACTIVE_DURATION_S - ACTIVE_EDGE_GUARD_S
        )
    ]

    return rise_times, fall_times


def find_local_50_crossing(
    time_s: np.ndarray,
    signal_ma: np.ndarray,
    nominal_edge_s: float,
    transition_type: str,
    event_duration_s: float,
) -> float | None:
    """
    Detect the measured 50% crossing around one nominal transition.

    Local low/high levels are estimated from regions away from the edge.
    The crossing itself is then detected in the expected direction.
    """
    dt_s = float(np.median(np.diff(time_s)))

    smooth_samples = max(
        1,
        round(CROSSING_SMOOTH_S / dt_s),
    )

    smooth_signal = rolling_mean(
        signal_ma,
        smooth_samples,
    )

    if transition_type == "rise":
        low_mask = (
            (time_s >= nominal_edge_s - 0.0030)
            & (time_s <= nominal_edge_s - 0.0008)
        )

        high_start_s = nominal_edge_s + min(
            0.0020,
            event_duration_s * 0.50,
        )
        high_end_s = nominal_edge_s + min(
            0.0035,
            event_duration_s * 0.80,
        )

        high_mask = (
            (time_s >= high_start_s)
            & (time_s <= high_end_s)
        )

    elif transition_type == "fall":
        high_mask = (
            (time_s >= nominal_edge_s - 0.0030)
            & (time_s <= nominal_edge_s - 0.0008)
        )

        low_mask = (
            (time_s >= nominal_edge_s + 0.0015)
            & (time_s <= nominal_edge_s + 0.0035)
        )

    else:
        raise ValueError(
            f"Unknown transition type: {transition_type}"
        )

    if (
        np.sum(low_mask) < 5
        or np.sum(high_mask) < 5
    ):
        return None

    local_low = float(
        np.median(
            smooth_signal[low_mask]
        )
    )

    local_high = float(
        np.median(
            smooth_signal[high_mask]
        )
    )

    if local_high <= local_low:
        return None

    level_50 = (
        local_low
        + 0.5 * (local_high - local_low)
    )

    search_mask = (
        (time_s >= nominal_edge_s - CROSSING_SEARCH_S)
        & (time_s <= nominal_edge_s + CROSSING_SEARCH_S)
    )

    indices = np.where(search_mask)[0]

    if len(indices) < 2:
        return None

    for index in indices[:-1]:
        y0 = smooth_signal[index]
        y1 = smooth_signal[index + 1]

        if transition_type == "rise":
            crossed = (
                y0 < level_50
                and y1 >= level_50
            )
        else:
            crossed = (
                y0 > level_50
                and y1 <= level_50
            )

        if crossed:
            if y1 == y0:
                return float(time_s[index])

            fraction = (
                (level_50 - y0)
                / (y1 - y0)
            )

            return float(
                time_s[index]
                + fraction
                * (
                    time_s[index + 1]
                    - time_s[index]
                )
            )

    return None


def find_prediction_50_crossing(
    time_s: np.ndarray,
    prediction_ma: np.ndarray,
    nominal_edge_s: float,
    transition_type: str,
    delta_i_ma: float,
) -> float | None:
    """
    Detect the model's own 50% crossing around one nominal transition.
    """
    level_50 = 0.5 * delta_i_ma

    search_mask = (
        (time_s >= nominal_edge_s - CROSSING_SEARCH_S)
        & (time_s <= nominal_edge_s + CROSSING_SEARCH_S)
    )

    indices = np.where(search_mask)[0]

    if len(indices) < 2:
        return None

    for index in indices[:-1]:
        y0 = prediction_ma[index]
        y1 = prediction_ma[index + 1]

        if transition_type == "rise":
            crossed = (
                y0 < level_50
                and y1 >= level_50
            )
        else:
            crossed = (
                y0 > level_50
                and y1 <= level_50
            )

        if crossed:
            if y1 == y0:
                return float(time_s[index])

            fraction = (
                (level_50 - y0)
                / (y1 - y0)
            )

            return float(
                time_s[index]
                + fraction
                * (
                    time_s[index + 1]
                    - time_s[index]
                )
            )

    return None


def aligned_transition_error(
    time_s: np.ndarray,
    measured_ma: np.ndarray,
    predicted_ma: np.ndarray,
    nominal_edges_s: np.ndarray,
    transition_type: str,
    event_duration_s: float,
    delta_i_ma: float,
) -> tuple[float, float, float, int, int]:
    """
    Align measured and predicted transitions independently at their 50%
    crossings, then calculate error on the same relative-time axis.

    This removes horizontal timing offset and focuses the comparison on
    transition shape / tau.
    """
    relative_time_s = np.arange(
        -ALIGN_PRE_S,
        ALIGN_POST_S + UNIFORM_DT_S,
        UNIFORM_DT_S,
    )

    all_errors: list[np.ndarray] = []
    valid_transitions = 0

    for nominal_edge_s in nominal_edges_s:
        measured_crossing_s = find_local_50_crossing(
            time_s=time_s,
            signal_ma=measured_ma,
            nominal_edge_s=nominal_edge_s,
            transition_type=transition_type,
            event_duration_s=event_duration_s,
        )

        predicted_crossing_s = find_prediction_50_crossing(
            time_s=time_s,
            prediction_ma=predicted_ma,
            nominal_edge_s=nominal_edge_s,
            transition_type=transition_type,
            delta_i_ma=delta_i_ma,
        )

        if (
            measured_crossing_s is None
            or predicted_crossing_s is None
        ):
            continue

        measured_sample_times = (
            measured_crossing_s
            + relative_time_s
        )

        predicted_sample_times = (
            predicted_crossing_s
            + relative_time_s
        )

        if (
            measured_sample_times[0] < time_s[0]
            or measured_sample_times[-1] > time_s[-1]
            or predicted_sample_times[0] < time_s[0]
            or predicted_sample_times[-1] > time_s[-1]
        ):
            continue

        measured_aligned = np.interp(
            measured_sample_times,
            time_s,
            measured_ma,
        )

        predicted_aligned = np.interp(
            predicted_sample_times,
            time_s,
            predicted_ma,
        )

        all_errors.append(
            predicted_aligned
            - measured_aligned
        )
        valid_transitions += 1

    if not all_errors:
        raise ValueError(
            f"No valid aligned {transition_type} transitions"
        )

    errors = np.concatenate(
        all_errors
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    me = float(
        np.mean(errors)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors * errors
            )
        )
    )

    return (
        mae,
        me,
        rmse,
        valid_transitions,
        len(errors),
    )


def analyze_one_file(
    condition: str,
    path: Path,
) -> list[TransitionResult]:
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
        _adc_idle_mean_ma,
    ) = active_uniform_measurement(
        dataframe
    )

    input_u = make_pulse_input(
        condition,
        time_s,
    )

    rise_times, fall_times = nominal_transition_times(
        condition
    )

    run_id = infer_run_id(path)
    results: list[TransitionResult] = []

    for model_name, parameters in MODEL_PARAMETERS.items():
        predicted_delta_ma = simulate_first_order_ode(
            input_u=input_u,
            dt_s=UNIFORM_DT_S,
            delta_i_ma=parameters["delta_i_ma"],
            tau_rise_s=parameters["tau_rise_s"],
            tau_fall_s=parameters["tau_fall_s"],
        )

        # --------------------------------------------------------
        # Rise: align measured and predicted 50% crossings
        # --------------------------------------------------------
        (
            rise_mae,
            rise_me,
            rise_rmse,
            rise_valid_transitions,
            rise_n_samples,
        ) = aligned_transition_error(
            time_s=time_s,
            measured_ma=measured_delta_ma,
            predicted_ma=predicted_delta_ma,
            nominal_edges_s=rise_times,
            transition_type="rise",
            event_duration_s=EVENT_DURATION_S[condition],
            delta_i_ma=parameters["delta_i_ma"],
        )

        results.append(
            TransitionResult(
                condition=condition,
                model_name=model_name,
                transition_type="rise",
                run_id=run_id,
                source_file=str(path),

                n_transitions=rise_valid_transitions,
                n_samples_in_windows=rise_n_samples,

                model_delta_i_ma=parameters["delta_i_ma"],
                model_tau_ms=(
                    parameters["tau_rise_s"]
                    * 1000.0
                ),

                mae_ma=rise_mae,
                me_ma=rise_me,
                rmse_ma=rise_rmse,
            )
        )

        # --------------------------------------------------------
        # Fall: align measured and predicted 50% crossings
        # --------------------------------------------------------
        (
            fall_mae,
            fall_me,
            fall_rmse,
            fall_valid_transitions,
            fall_n_samples,
        ) = aligned_transition_error(
            time_s=time_s,
            measured_ma=measured_delta_ma,
            predicted_ma=predicted_delta_ma,
            nominal_edges_s=fall_times,
            transition_type="fall",
            event_duration_s=EVENT_DURATION_S[condition],
            delta_i_ma=parameters["delta_i_ma"],
        )

        results.append(
            TransitionResult(
                condition=condition,
                model_name=model_name,
                transition_type="fall",
                run_id=run_id,
                source_file=str(path),

                n_transitions=fall_valid_transitions,
                n_samples_in_windows=fall_n_samples,

                model_delta_i_ma=parameters["delta_i_ma"],
                model_tau_ms=(
                    parameters["tau_fall_s"]
                    * 1000.0
                ),

                mae_ma=fall_mae,
                me_ma=fall_me,
                rmse_ma=fall_rmse,
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


def summarize(
    per_run_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = per_run_dataframe.groupby(
        [
            "condition",
            "transition_type",
            "model_name",
        ],
        sort=True,
    )

    for (
        condition,
        transition_type,
        model_name,
    ), group in grouped:
        rows.append(
            {
                "condition": condition,
                "transition_type": transition_type,
                "model_name": model_name,
                "n_runs": int(len(group)),
                "n_transitions_per_run": int(
                    group["n_transitions"].iloc[0]
                ),

                "model_delta_i_ma": float(
                    group["model_delta_i_ma"].iloc[0]
                ),
                "model_tau_ms": (
                    float(
                        group["model_tau_ms"].dropna().iloc[0]
                    )
                    if group["model_tau_ms"].notna().any()
                    else np.nan
                ),

                "mae_mean_ma": float(
                    group["mae_ma"].mean()
                ),
                "mae_std_ma": float(
                    group["mae_ma"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,

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
            }
        )

    return pd.DataFrame(rows)


def build_comparison_table(
    summary_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Put PPK2 and scope RMSE side-by-side and report scope-minus-PPK2.
    Negative delta means scope is better.
    """
    subset = summary_dataframe[
        [
            "condition",
            "transition_type",
            "model_name",
            "rmse_mean_ma",
            "rmse_std_ma",
            "mae_mean_ma",
            "me_mean_ma",
        ]
    ].copy()

    wide = subset.pivot(
        index=[
            "condition",
            "transition_type",
        ],
        columns="model_name",
        values=[
            "rmse_mean_ma",
            "rmse_std_ma",
            "mae_mean_ma",
            "me_mean_ma",
        ],
    )

    wide.columns = [
        f"{metric}_{model}"
        for metric, model in wide.columns
    ]

    wide = wide.reset_index()

    if (
        "rmse_mean_ma_scope" in wide.columns
        and "rmse_mean_ma_ppk2" in wide.columns
    ):
        wide["rmse_scope_minus_ppk2_ma"] = (
            wide["rmse_mean_ma_scope"]
            - wide["rmse_mean_ma_ppk2"]
        )

        wide["rmse_improvement_scope_percent"] = (
            (
                wide["rmse_mean_ma_ppk2"]
                - wide["rmse_mean_ma_scope"]
            )
            / wide["rmse_mean_ma_ppk2"]
            * 100.0
        )

        wide["better_tau"] = np.where(
            wide["rmse_scope_minus_ppk2_ma"] < 0,
            "scope_0.440ms",
            np.where(
                wide["rmse_scope_minus_ppk2_ma"] > 0,
                "ppk2_0.579ms",
                "tie",
            ),
        )

    return wide


def main() -> None:
    all_results: list[TransitionResult] = []
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

    per_run_dataframe = pd.DataFrame(
        [
            result.__dict__
            for result in all_results
        ]
    )

    summary_dataframe = summarize(
        per_run_dataframe
    )

    comparison_dataframe = build_comparison_table(
        summary_dataframe
    )

    per_run_path = (
        OUTPUT_DIR
        / "transition_per_run_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "transition_summary_metrics.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "transition_ppk_vs_scope_comparison.csv"
    )

    per_run_dataframe.to_csv(
        per_run_path,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    comparison_dataframe.to_csv(
        comparison_path,
        index=False,
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )

    print()
    print(f"Analyzed rows: {len(per_run_dataframe)}")
    print(f"Skipped files: {len(errors)}")
    print(f"Wrote: {per_run_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {comparison_path}")


if __name__ == "__main__":
    main()