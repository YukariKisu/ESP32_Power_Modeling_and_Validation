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
    "raw_waveform_overlay_diagnostics"
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


def collect_csv_files(input_dir: Path) -> list[Path]:
    """
    Collect raw run CSV files from one condition folder.

    The condition folders are expected to contain files such as:
        adc_burst_100ms_100samples_run1.csv
        ...
        adc_burst_100ms_100samples_run10.csv
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    csv_files = sorted(
        input_dir.glob("*.csv"),
        key=lambda path: (
            int(
                re.search(
                    r"run[_-]?(\d+)",
                    path.stem,
                    flags=re.IGNORECASE,
                ).group(1)
            )
            if re.search(
                r"run[_-]?(\d+)",
                path.stem,
                flags=re.IGNORECASE,
            )
            else 999999,
            path.name,
        ),
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {input_dir}"
        )

    return csv_files


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



# ============================================================
# Raw waveform overlay settings
# ============================================================

FULL_START_S = 0.0
FULL_END_S = 20.0

# Zoom around the first nominal burst.
RISE_PRE_S = 0.006
RISE_POST_S = 0.012

FALL_PRE_S = 0.012
FALL_POST_S = 0.012

# Plot measured current above the run-specific idle baseline.
# This makes waveform shape/timing easier to compare across runs.
PLOT_DELTA_CURRENT = True

# Status is used only for line style / legend.
RUN_STATUS = {
    "adc_burst_100ms_100samples": {
        "run1": "complete",
        "run2": "complete",
        "run3": "incomplete",
        "run4": "complete",
        "run5": "incomplete",
        "run6": "complete",
        "run7": "incomplete",
        "run8": "complete",
        "run9": "complete",
        "run10": "incomplete",
    },
    "adc_burst_100ms_1000samples": {
        "run1": "incomplete",
        "run2": "partial",
        "run3": "complete",
        "run4": "incomplete",
        "run5": "incomplete",
        "run6": "incomplete",
        "run7": "complete",
        "run8": "complete",
        "run9": "incomplete",
        "run10": "incomplete",
    },
}


def status_linestyle(status: str) -> str:
    if status == "complete":
        return "-"
    if status == "partial":
        return "-."
    return "--"


def load_runs(condition: str, input_dir: Path) -> list[dict]:
    runs = []

    for path in collect_csv_files(input_dir):
        df = read_ppk_csv(path)

        sync_mid_s = detect_sync_midpoint(df)
        df["aligned_time_s"] = df["time_s"] - sync_mid_s

        time_s, measured_delta_ma, idle_ma = active_uniform_measurement(df)

        time_s = np.asarray(time_s, dtype=float)
        measured_delta_ma = np.asarray(measured_delta_ma, dtype=float)

        # active_uniform_measurement normally returns 0...20 s.
        # Normalize defensively in case the helper returns an absolute axis.
        if len(time_s) and time_s[0] > 1.0:
            time_s = time_s - time_s[0]

        if PLOT_DELTA_CURRENT:
            current_ma = measured_delta_ma
            ylabel = "Measured current above idle [mA]"
        else:
            current_ma = measured_delta_ma + float(idle_ma)
            ylabel = "Measured current [mA]"

        runs.append(
            {
                "run_id": infer_run_id(path),
                "time_s": time_s,
                "current_ma": current_ma,
                "idle_ma": float(idle_ma),
                "ylabel": ylabel,
            }
        )

    return sorted(
        runs,
        key=lambda x: int(str(x["run_id"]).replace("run", "")),
    )



# ============================================================
# Stacked waveform diagnostics
# ============================================================

STACK_SPACING_MA = 20.0

# Three views:
# 1) entire 20 s active phase
# 2) early section containing several bursts
# 3) late section containing several bursts
STACK_WINDOWS = {
    "full": (0.0, 20.0),
    "early": (0.0, 0.50),
    "late": (19.50, 20.0),
}


def plot_stacked_waveforms(
    condition: str,
    runs: list[dict],
    start_s: float,
    end_s: float,
    suffix: str,
) -> Path:
    """
    Plot each run on its own vertically shifted baseline.

    Important:
    - Time is NOT realigned to each burst.
    - Every run retains its CPU-sync-based timing.
    - Vertical offset is display-only.
    - This makes run-to-run phase differences visible without waveform overlap.
    """
    output_dir = (
        OUTPUT_DIR
        / "stacked_waveforms"
        / condition
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(15, 9)
    )

    tick_positions = []
    tick_labels = []

    for row_index, run in enumerate(runs):
        run_id = str(run["run_id"])
        status = RUN_STATUS.get(
            condition,
            {},
        ).get(
            run_id,
            "unknown",
        )

        # Put run1 at the top and run10 at the bottom.
        vertical_offset = (
            (len(runs) - 1 - row_index)
            * STACK_SPACING_MA
        )

        mask = (
            (run["time_s"] >= start_s)
            & (run["time_s"] <= end_s)
        )

        t = run["time_s"][mask]
        y = run["current_ma"][mask]

        ax.plot(
            t,
            y + vertical_offset,
            linewidth=0.9,
            linestyle=status_linestyle(
                status
            ),
        )

        # Baseline reference for each row.
        ax.hlines(
            vertical_offset,
            start_s,
            end_s,
            linewidth=0.5,
            alpha=0.25,
        )

        tick_positions.append(
            vertical_offset
        )
        tick_labels.append(
            f"{run_id} ({status})"
        )

    ax.set_xlim(
        start_s,
        end_s,
    )

    ax.set_yticks(
        tick_positions
    )
    ax.set_yticklabels(
        tick_labels
    )

    ax.set_xlabel(
        "Time from active-window start [s]"
    )
    ax.set_ylabel(
        "Run (vertical offsets are display-only)"
    )

    ax.set_title(
        f"{condition} | stacked raw waveforms | {suffix}"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / f"{condition}_stacked_{suffix}.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for condition, input_dir in INPUT_DIRS.items():
        print(
            f"\nLoading {condition}",
            flush=True,
        )

        runs = load_runs(
            condition,
            input_dir,
        )

        if not runs:
            print(
                f"No runs found for {condition}"
            )
            continue

        for suffix, (
            start_s,
            end_s,
        ) in STACK_WINDOWS.items():
            path = plot_stacked_waveforms(
                condition=condition,
                runs=runs,
                start_s=start_s,
                end_s=end_s,
                suffix=suffix,
            )
            generated.append(path)
            print(f"Wrote: {path}")

    print()
    print(
        f"Generated {len(generated)} stacked plots."
    )


if __name__ == "__main__":
    main()