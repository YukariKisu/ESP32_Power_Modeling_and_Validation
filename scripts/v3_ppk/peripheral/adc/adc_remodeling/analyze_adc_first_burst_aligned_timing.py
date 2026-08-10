from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "first_burst_aligned_timing_diagnostics"
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
# Transition timing diagnostics
# ============================================================

# Search measured 50% crossing around each firmware-defined nominal edge.
CROSSING_SEARCH_S = 0.0025

# Smoothing used only for 50% crossing detection.
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
    search_center_s: float,
    transition_type: str,
    event_duration_s: float,
    search_half_width_s: float = 0.0100,
) -> float | None:
    """
    Detect a measured 50% crossing around a supplied search center.
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
            (time_s >= search_center_s - 0.0030)
            & (time_s <= search_center_s - 0.0008)
        )

        high_start_s = search_center_s + min(
            0.0020,
            event_duration_s * 0.50,
        )
        high_end_s = search_center_s + min(
            0.0035,
            event_duration_s * 0.80,
        )

        high_mask = (
            (time_s >= high_start_s)
            & (time_s <= high_end_s)
        )

    elif transition_type == "fall":
        high_mask = (
            (time_s >= search_center_s - 0.0030)
            & (time_s <= search_center_s - 0.0008)
        )

        low_mask = (
            (time_s >= search_center_s + 0.0015)
            & (time_s <= search_center_s + 0.0035)
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
        (time_s >= search_center_s - search_half_width_s)
        & (time_s <= search_center_s + search_half_width_s)
    )

    indices = np.where(search_mask)[0]

    if len(indices) < 2:
        return None

    candidates: list[float] = []

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
                crossing_s = float(time_s[index])
            else:
                fraction = (
                    (level_50 - y0)
                    / (y1 - y0)
                )

                crossing_s = float(
                    time_s[index]
                    + fraction
                    * (
                        time_s[index + 1]
                        - time_s[index]
                    )
                )

            candidates.append(crossing_s)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda value: abs(
            value - search_center_s
        ),
    )


def collect_first_burst_aligned_offsets(
    time_s: np.ndarray,
    measured_ma: np.ndarray,
    nominal_edges_s: np.ndarray,
    transition_type: str,
    event_duration_s: float,
) -> pd.DataFrame:
    """
    Align each run to its first burst.

    relative_offset_from_burst1_ms =
        (measured_50[k] - nominal[k])
        - (measured_50[1] - nominal[1])

    Therefore Burst 1 is always 0 ms.
    """
    rows: list[dict[str, object]] = []

    if len(nominal_edges_s) == 0:
        return pd.DataFrame(rows)

    first_nominal_s = float(
        nominal_edges_s[0]
    )

    first_crossing_s = find_local_50_crossing(
        time_s=time_s,
        signal_ma=measured_ma,
        search_center_s=first_nominal_s,
        transition_type=transition_type,
        event_duration_s=event_duration_s,
        search_half_width_s=0.0100,
    )

    if first_crossing_s is None:
        for event_index, nominal_edge_s in enumerate(
            nominal_edges_s,
            start=1,
        ):
            rows.append(
                {
                    "event_index": event_index,
                    "transition_type": transition_type,
                    "nominal_edge_s": float(nominal_edge_s),
                    "measured_50_s": np.nan,
                    "absolute_offset_ms": np.nan,
                    "relative_offset_from_burst1_ms": np.nan,
                    "detected": False,
                }
            )

        return pd.DataFrame(rows)

    first_absolute_offset_ms = (
        first_crossing_s
        - first_nominal_s
    ) * 1000.0

    previous_crossing_s = first_crossing_s
    previous_nominal_s = first_nominal_s

    for event_index, nominal_edge_s in enumerate(
        nominal_edges_s,
        start=1,
    ):
        nominal_edge_s = float(
            nominal_edge_s
        )

        if event_index == 1:
            measured_crossing_s = first_crossing_s

        else:
            nominal_step_s = (
                nominal_edge_s
                - previous_nominal_s
            )

            # Track from the previous actual transition rather than repeatedly
            # returning to the global CPU-sync-based nominal timing.
            adaptive_center_s = (
                previous_crossing_s
                + nominal_step_s
            )

            measured_crossing_s = find_local_50_crossing(
                time_s=time_s,
                signal_ma=measured_ma,
                search_center_s=adaptive_center_s,
                transition_type=transition_type,
                event_duration_s=event_duration_s,
                search_half_width_s=0.0100,
            )

        if measured_crossing_s is None:
            rows.append(
                {
                    "event_index": event_index,
                    "transition_type": transition_type,
                    "nominal_edge_s": nominal_edge_s,
                    "measured_50_s": np.nan,
                    "absolute_offset_ms": np.nan,
                    "relative_offset_from_burst1_ms": np.nan,
                    "detected": False,
                }
            )

            previous_nominal_s = nominal_edge_s
            continue

        absolute_offset_ms = (
            measured_crossing_s
            - nominal_edge_s
        ) * 1000.0

        relative_offset_ms = (
            absolute_offset_ms
            - first_absolute_offset_ms
        )

        rows.append(
            {
                "event_index": event_index,
                "transition_type": transition_type,
                "nominal_edge_s": nominal_edge_s,
                "measured_50_s": measured_crossing_s,
                "absolute_offset_ms": absolute_offset_ms,
                "relative_offset_from_burst1_ms": relative_offset_ms,
                "detected": True,
            }
        )

        previous_crossing_s = measured_crossing_s
        previous_nominal_s = nominal_edge_s

    return pd.DataFrame(rows)


def summarize_first_burst_aligned(
    offsets_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = offsets_dataframe.groupby(
        [
            "condition",
            "run_id",
            "transition_type",
        ],
        sort=True,
    )

    for (
        condition,
        run_id,
        transition_type,
    ), group in grouped:
        detected_group = group[
            group["detected"].astype(bool)
        ].copy()

        total = int(len(group))
        detected = int(len(detected_group))

        first_row = group[
            group["event_index"] == 1
        ]

        burst1_absolute_offset_ms = (
            float(
                first_row[
                    "absolute_offset_ms"
                ].iloc[0]
            )
            if (
                not first_row.empty
                and pd.notna(
                    first_row[
                        "absolute_offset_ms"
                    ].iloc[0]
                )
            )
            else np.nan
        )

        if detected > 1:
            event_index = detected_group[
                "event_index"
            ].to_numpy(dtype=float)

            relative_offset = detected_group[
                "relative_offset_from_burst1_ms"
            ].to_numpy(dtype=float)

            slope, intercept = np.polyfit(
                event_index,
                relative_offset,
                1,
            )

            fitted = (
                slope * event_index
                + intercept
            )

            ss_res = float(
                np.sum(
                    (
                        relative_offset
                        - fitted
                    ) ** 2
                )
            )

            ss_tot = float(
                np.sum(
                    (
                        relative_offset
                        - np.mean(relative_offset)
                    ) ** 2
                )
            )

            drift_ms_per_event = float(
                slope
            )

            drift_r2 = (
                1.0 - ss_res / ss_tot
                if ss_tot > 0
                else 0.0
            )

            final_relative_offset_ms = float(
                relative_offset[-1]
            )

            relative_offset_min_ms = float(
                np.min(relative_offset)
            )

            relative_offset_max_ms = float(
                np.max(relative_offset)
            )

        elif detected == 1:
            drift_ms_per_event = np.nan
            drift_r2 = np.nan
            final_relative_offset_ms = 0.0
            relative_offset_min_ms = 0.0
            relative_offset_max_ms = 0.0

        else:
            drift_ms_per_event = np.nan
            drift_r2 = np.nan
            final_relative_offset_ms = np.nan
            relative_offset_min_ms = np.nan
            relative_offset_max_ms = np.nan

        rows.append(
            {
                "condition": condition,
                "run_id": run_id,
                "transition_type": transition_type,
                "n_nominal_transitions": total,
                "n_detected_transitions": detected,
                "detection_rate_percent": (
                    100.0 * detected / total
                    if total > 0
                    else np.nan
                ),
                "burst1_absolute_offset_ms": (
                    burst1_absolute_offset_ms
                ),
                "final_relative_offset_ms": (
                    final_relative_offset_ms
                ),
                "relative_offset_min_ms": (
                    relative_offset_min_ms
                ),
                "relative_offset_max_ms": (
                    relative_offset_max_ms
                ),
                "drift_ms_per_event": (
                    drift_ms_per_event
                ),
                "drift_r2": drift_r2,
            }
        )

    return pd.DataFrame(rows)


def plot_all_runs_first_burst_aligned(
    offsets_dataframe: pd.DataFrame,
    condition: str,
    transition_type: str,
    output_dir: Path,
) -> None:
    """
    Overlay all runs after setting Burst 1 = 0 ms.
    """
    plot_dir = (
        output_dir
        / "plots_all_runs"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset = offsets_dataframe[
        (
            offsets_dataframe["condition"]
            == condition
        )
        & (
            offsets_dataframe["transition_type"]
            == transition_type
        )
    ].copy()

    if subset.empty:
        return

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for run_id, group in subset.groupby(
        "run_id",
        sort=True,
    ):
        group = group[
            group["detected"].astype(bool)
        ]

        if group.empty:
            continue

        axis.plot(
            group["event_index"],
            group[
                "relative_offset_from_burst1_ms"
            ],
            marker="o",
            markersize=2.0,
            linewidth=1.0,
            label=run_id,
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        f"{condition} | {transition_type} | "
        "all runs aligned to Burst 1"
    )

    axis.set_xlabel(
        "Burst / event index"
    )

    axis.set_ylabel(
        "Relative timing offset from Burst 1 [ms]"
    )

    axis.legend(
        ncol=2,
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        plot_dir
        / (
            f"{condition}_{transition_type}_"
            "all_runs_first_burst_aligned.png"
        ),
        dpi=170,
    )

    plt.close(figure)



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

    rise_mask = transition_mask(
        time_s,
        rise_times,
    )

    fall_mask = transition_mask(
        time_s,
        fall_times,
    )

    combined_mask = (
        rise_mask | fall_mask
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

        for transition_type, mask, n_transitions, tau_s in (
            (
                "rise",
                rise_mask,
                len(rise_times),
                parameters["tau_rise_s"],
            ),
            (
                "fall",
                fall_mask,
                len(fall_times),
                parameters["tau_fall_s"],
            ),
            (
                "combined",
                combined_mask,
                len(rise_times) + len(fall_times),
                np.nan,
            ),
        ):
            mae, me, rmse, n_window_samples = error_metrics(
                predicted_delta_ma,
                measured_delta_ma,
                mask,
            )

            results.append(
                TransitionResult(
                    condition=condition,
                    model_name=model_name,
                    transition_type=transition_type,
                    run_id=run_id,
                    source_file=str(path),

                    n_transitions=int(n_transitions),
                    n_samples_in_windows=n_window_samples,

                    model_delta_i_ma=parameters["delta_i_ma"],
                    model_tau_ms=(
                        float(tau_s * 1000.0)
                        if np.isfinite(tau_s)
                        else np.nan
                    ),

                    mae_ma=mae,
                    me_ma=me,
                    rmse_ma=rmse,
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
    all_offsets: list[pd.DataFrame] = []
    sync_diagnostics: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for condition, input_dir in INPUT_DIRS.items():
        for path in collect_csv_files(
            input_dir
        ):
            try:
                print(
                    f"Processing {condition}: {path}",
                    flush=True,
                )

                dataframe = read_ppk_csv(path)

                sync_mid_s = detect_sync_midpoint(
                    dataframe
                )

                run_id = infer_run_id(path)
                expected_sync_mid_s = (
                    SYNC_EXPECTED_START_S
                    + SYNC_DURATION_S / 2.0
                )

                sync_diagnostics.append(
                    {
                        "condition": condition,
                        "run_id": run_id,
                        "source_file": str(path),
                        "sync_mid_s": sync_mid_s,
                        "expected_sync_mid_s": expected_sync_mid_s,
                        "sync_mid_offset_ms": (
                            sync_mid_s - expected_sync_mid_s
                        ) * 1000.0,
                    }
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

                rise_times, fall_times = nominal_transition_times(
                    condition
                )

                rise_offsets = collect_first_burst_aligned_offsets(
                    time_s=time_s,
                    measured_ma=measured_delta_ma,
                    nominal_edges_s=rise_times,
                    transition_type="rise",
                    event_duration_s=EVENT_DURATION_S[condition],
                )

                fall_offsets = collect_first_burst_aligned_offsets(
                    time_s=time_s,
                    measured_ma=measured_delta_ma,
                    nominal_edges_s=fall_times,
                    transition_type="fall",
                    event_duration_s=EVENT_DURATION_S[condition],
                )

                run_offsets = pd.concat(
                    [
                        rise_offsets,
                        fall_offsets,
                    ],
                    ignore_index=True,
                )

                run_offsets.insert(
                    0,
                    "condition",
                    condition,
                )

                run_offsets.insert(
                    1,
                    "run_id",
                    run_id,
                )

                run_offsets.insert(
                    2,
                    "source_file",
                    str(path),
                )

                all_offsets.append(
                    run_offsets
                )

            except Exception as exception:
                errors.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(exception),
                    }
                )

    sync_diagnostics_path = (
        OUTPUT_DIR / "sync_midpoint_diagnostics.csv"
    )

    if sync_diagnostics:
        sync_diagnostics_dataframe = pd.DataFrame(
            sync_diagnostics
        ).sort_values(
            ["condition", "run_id"]
        )
        sync_diagnostics_dataframe.to_csv(
            sync_diagnostics_path,
            index=False,
        )

        print()
        print("Sync midpoint diagnostics:")
        print(
            sync_diagnostics_dataframe[
                [
                    "condition",
                    "run_id",
                    "sync_mid_s",
                    "sync_mid_offset_ms",
                ]
            ].to_string(index=False)
        )
        print(f"Wrote: {sync_diagnostics_path}")

    if not all_offsets:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )

        raise SystemExit(
            "No runs could be analyzed. "
            "See errors.csv."
        )

    offsets_dataframe = pd.concat(
        all_offsets,
        ignore_index=True,
    )

    summary_dataframe = summarize_first_burst_aligned(
        offsets_dataframe
    )

    offsets_path = (
        OUTPUT_DIR
        / "first_burst_aligned_offsets.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "first_burst_aligned_summary.csv"
    )

    offsets_dataframe.to_csv(
        offsets_path,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    for condition in sorted(
        offsets_dataframe[
            "condition"
        ].unique()
    ):
        for transition_type in (
            "rise",
            "fall",
        ):
            plot_all_runs_first_burst_aligned(
                offsets_dataframe=offsets_dataframe,
                condition=condition,
                transition_type=transition_type,
                output_dir=OUTPUT_DIR,
            )

    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "errors.csv",
            index=False,
        )

    print()
    print(
        "Analyzed runs:",
        offsets_dataframe[
            ["condition", "run_id"]
        ].drop_duplicates().shape[0],
    )

    print(f"Skipped files: {len(errors)}")
    print(f"Wrote: {offsets_path}")
    print(f"Wrote: {summary_path}")
    print(
        "Wrote overlay plots to:",
        OUTPUT_DIR / "plots_all_runs",
    )




if __name__ == "__main__":
    main()