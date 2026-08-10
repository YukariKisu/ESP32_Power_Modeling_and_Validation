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


def collect_csv_files(input_dir: Path) -> list[Path]:
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



# ============================================================
# Firmware-vs-measured timing diagnostic
# ============================================================

# Firmware timing measured from the ACTUAL sync-start edge:
#   1.0 s sync pulse
# + 5.0 s recovery idle
# +10.0 s initial idle
# =16.0 s until the first ADC burst begins.
FIRMWARE_SYNC_TO_FIRST_BURST_S = (
    SYNC_DURATION_S + 5.0 + 10.0
)

# Search regions are deliberately broad because the purpose is to test timing,
# not to force the measured waveform onto the firmware timing.
SYNC_SEARCH_START_S = 1.0
SYNC_SEARCH_END_S = 6.0

# The first ADC burst should be preceded by the long initial-idle phase.
# Search broadly around the firmware expectation while still remaining far
# from earlier experiment phases.
FIRST_BURST_SEARCH_BEFORE_S = 0.50
FIRST_BURST_SEARCH_AFTER_S = 0.50

# Smoothing only for edge detection; raw current is retained for plotting.
EDGE_SMOOTH_S = 0.00020

# Diagnostic plot windows.
SYNC_PLOT_PRE_S = 0.30
SYNC_PLOT_POST_S = 1.30
BURST_PLOT_PRE_S = 0.20
BURST_PLOT_POST_S = 0.30

TIMING_OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "sync_start_to_first_burst_diagnostics"
)


# ============================================================
# Early / Middle / Late measured-vs-predicted comparison
# ============================================================

EARLY_MIDDLE_LATE_OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "early_middle_late_measured_vs_predicted"
)

# Windows are relative to the firmware-defined ADC active-window start.
# No actual ADC-edge re-alignment is applied.
COMPARISON_WINDOWS_S = {
    "Early": (0.0, 0.5),
    "Middle": (9.75, 10.25),
    "Late": (19.5, 20.0),
}

# Plot both parameter sets separately so the waveform is not overcrowded.
MODELS_TO_PLOT = ("ppk2", "scope")

# Vertical separation between runs in the stacked plots.
STACK_SPACING_MA = 18.0



def detect_sync_segment(
    dataframe: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Detect the actual CPU-sync pulse and return:
        actual_sync_start_s,
        actual_sync_end_s,
        actual_sync_mid_s

    Selection prefers a segment whose duration is closest to the 1 s firmware
    sync duration. The expected absolute recording time is not used as the
    primary criterion, because PPK2 recording start is not firmware-synchronized.
    """
    time_s = dataframe["time_s"].to_numpy(dtype=float)
    current_ma = dataframe["current_ma"].to_numpy(dtype=float)

    dt_s = float(np.median(np.diff(time_s)))
    smooth_n = max(1, round(SYNC_SMOOTH_WINDOW_S / dt_s))
    smooth = rolling_mean(current_ma, smooth_n)

    search_mask = (
        (time_s >= SYNC_SEARCH_START_S)
        & (time_s <= SYNC_SEARCH_END_S)
    )
    if not np.any(search_mask):
        raise ValueError("Sync search window has no samples")

    st = time_s[search_mask]
    sy = smooth[search_mask]

    # Estimate local baseline from the lower part of the search window.
    baseline = float(np.percentile(sy, 20))
    peak = float(np.percentile(sy, 99))
    threshold = baseline + 0.45 * (peak - baseline)

    high = sy >= threshold

    segments: list[tuple[int, int]] = []
    start_idx = None
    for i, is_high in enumerate(high):
        if is_high and start_idx is None:
            start_idx = i
        elif (not is_high) and start_idx is not None:
            segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(high) - 1))

    if not segments:
        raise ValueError("No sync-like high-current segment detected")

    def score(seg: tuple[int, int]) -> tuple[float, float]:
        i0, i1 = seg
        dur = float(st[i1] - st[i0])
        # First criterion: duration near 1 s.
        duration_error = abs(dur - SYNC_DURATION_S)
        # Second criterion: prefer stronger segment.
        strength = float(np.mean(sy[i0:i1 + 1]) - baseline)
        return duration_error, -strength

    i0, i1 = min(segments, key=score)

    sync_start_s = float(st[i0])
    sync_end_s = float(st[i1])
    sync_mid_s = 0.5 * (sync_start_s + sync_end_s)

    return sync_start_s, sync_end_s, sync_mid_s


def detect_first_adc_rise(
    dataframe: pd.DataFrame,
    sync_start_s: float,
    condition: str,
) -> tuple[float, float]:
    """
    Detect the first measured ADC burst rise.

    For short 100-sample bursts, percentile-based active-level estimation is
    unreliable because the burst occupies only ~4.4% of each 100 ms period.
    Instead, estimate the idle baseline before the expected test-window start,
    detect real high-current pulse segments, and require a periodic sequence
    with the firmware-defined 100 ms spacing.

    Returns:
        actual_first_burst_s,
        threshold_ma
    """
    time_s = dataframe["time_s"].to_numpy(dtype=float)
    current_ma = dataframe["current_ma"].to_numpy(dtype=float)

    expected_s = sync_start_s + FIRMWARE_SYNC_TO_FIRST_BURST_S
    interval_s = EVENT_INTERVAL_S[condition]
    expected_duration_s = EVENT_DURATION_S[condition]

    # Wide enough to reveal a real timing mismatch, but detection is based on
    # recurring pulse structure rather than simply taking the earliest crossing.
    search_start_s = expected_s - FIRST_BURST_SEARCH_BEFORE_S
    search_end_s = expected_s + FIRST_BURST_SEARCH_AFTER_S

    mask = (
        (time_s >= search_start_s)
        & (time_s <= search_end_s)
    )
    if not np.any(mask):
        raise ValueError("First-burst search window has no samples")

    t = time_s[mask]
    y_raw = current_ma[mask]

    dt_s = float(np.median(np.diff(t)))
    smooth_n = max(1, round(EDGE_SMOOTH_S / dt_s))
    y = rolling_mean(y_raw, smooth_n)

    # Use the pre-test region to estimate idle. This avoids the occupancy
    # problem that caused the old 90th-percentile method to fail for 100 samples.
    baseline_mask = (
        (time_s >= expected_s - 0.40)
        & (time_s <= expected_s - 0.05)
    )
    if not np.any(baseline_mask):
        raise ValueError("No baseline samples before expected first burst")

    baseline_values = current_ma[baseline_mask]
    baseline = float(np.median(baseline_values))

    # Robust noise estimate from the idle region.
    mad = float(
        np.median(np.abs(baseline_values - np.median(baseline_values)))
    )
    robust_sigma = 1.4826 * mad

    # ADC burst amplitude is around the 10 mA scale; use a conservative
    # threshold that stays well above idle fluctuation without requiring a
    # global active percentile.
    threshold_delta_ma = max(2.0, 6.0 * robust_sigma)
    threshold = baseline + threshold_delta_ma

    high = y >= threshold

    # Convert thresholded samples into contiguous pulse segments.
    segments: list[tuple[int, int]] = []
    seg_start = None
    for i, is_high in enumerate(high):
        if is_high and seg_start is None:
            seg_start = i
        elif (not is_high) and seg_start is not None:
            segments.append((seg_start, i - 1))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, len(high) - 1))

    # Keep physically plausible ADC burst segments.
    # The lower bound rejects noise spikes; the upper bound remains broad
    # enough for the 44 ms / 1000-sample case.
    min_duration_s = max(0.00030, 0.15 * expected_duration_s)
    max_duration_s = min(
        0.080,
        max(0.010, 1.8 * expected_duration_s),
    )

    pulse_starts: list[float] = []
    for i0, i1 in segments:
        duration_s = float(t[i1] - t[i0])
        if min_duration_s <= duration_s <= max_duration_s:
            # Refine the start using interpolation across the threshold.
            if i0 > 0:
                y0, y1 = y[i0 - 1], y[i0]
                t0, t1 = t[i0 - 1], t[i0]
                if y1 != y0:
                    frac = (threshold - y0) / (y1 - y0)
                    start_s = float(t0 + frac * (t1 - t0))
                else:
                    start_s = float(t[i0])
            else:
                start_s = float(t[i0])
            pulse_starts.append(start_s)

    if len(pulse_starts) < 3:
        raise ValueError(
            f"Too few ADC-like pulse segments detected: {len(pulse_starts)}"
        )

    # Find the earliest pulse that begins a recurring sequence at ~100 ms.
    # Requiring several successive matches prevents a random idle fluctuation
    # from being mislabeled as Burst 1.
    period_tol_s = 0.010
    required_matches = 3

    for first in pulse_starts:
        matches = 1
        target = first + interval_s

        for candidate in pulse_starts:
            if candidate <= first:
                continue

            if abs(candidate - target) <= period_tol_s:
                matches += 1
                target += interval_s
                if matches >= required_matches:
                    return float(first), float(threshold)

            elif candidate > target + period_tol_s:
                # Candidate skipped the current target; move on to trying
                # another possible sequence start.
                break

    raise ValueError(
        "No recurring ADC pulse sequence with ~100 ms spacing detected"
    )



def make_pulse_input(
    condition: str,
    time_from_active_start_s: np.ndarray,
) -> np.ndarray:
    """
    Firmware-defined binary ADC activity input.

    t=0 is the firmware-defined ADC active-window start, so Burst 1 starts at
    t=0 and subsequent bursts repeat at EVENT_INTERVAL_S.
    """
    interval_s = EVENT_INTERVAL_S[condition]
    duration_s = EVENT_DURATION_S[condition]

    phase_s = np.mod(
        time_from_active_start_s,
        interval_s,
    )
    return (phase_s < duration_s).astype(float)


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

    for i in range(1, len(input_u)):
        target_ma = delta_i_ma * input_u[i]

        tau_s = (
            tau_rise_s
            if target_ma >= prediction_delta_ma[i - 1]
            else tau_fall_s
        )

        alpha = 1.0 - math.exp(-dt_s / tau_s)

        prediction_delta_ma[i] = (
            prediction_delta_ma[i - 1]
            + alpha
            * (
                target_ma
                - prediction_delta_ma[i - 1]
            )
        )

    return prediction_delta_ma


def estimate_run_idle_ma(
    dataframe: pd.DataFrame,
    sync_start_s: float,
) -> float:
    """
    Estimate a run-specific idle level from the initial and final idle phases,
    using the actual measured sync-start edge as the time anchor.
    """
    aligned_s = (
        dataframe["time_s"].to_numpy(dtype=float)
        - sync_start_s
    )
    current_ma = dataframe["current_ma"].to_numpy(dtype=float)

    initial_mask = (
        (aligned_s >= 6.2)
        & (aligned_s <= 15.8)
    )
    final_mask = (
        (aligned_s >= 36.2)
        & (aligned_s <= 45.8)
    )

    means = []

    if np.any(initial_mask):
        means.append(
            float(np.mean(current_ma[initial_mask]))
        )

    if np.any(final_mask):
        means.append(
            float(np.mean(current_ma[final_mask]))
        )

    if not means:
        raise ValueError(
            "Could not estimate idle level from initial/final idle phases"
        )

    return float(np.mean(means))


def build_prediction_for_active_window(
    condition: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the firmware-timed prediction over the full 20 s active window.
    """
    dt_s = UNIFORM_DT_S
    time_s = np.arange(
        0.0,
        ACTIVE_DURATION_S,
        dt_s,
    )

    input_u = make_pulse_input(
        condition,
        time_s,
    )

    params = MODEL_PARAMETERS[model_name]

    predicted_delta_ma = simulate_first_order_ode(
        input_u=input_u,
        dt_s=dt_s,
        delta_i_ma=params["delta_i_ma"],
        tau_rise_s=params["tau_rise_s"],
        tau_fall_s=params["tau_fall_s"],
    )

    return time_s, predicted_delta_ma


def plot_early_middle_late_measured_vs_predicted(
    diagnostics: pd.DataFrame,
    condition: str,
    model_name: str,
) -> Path:
    """
    Stacked measured-vs-predicted comparison at Early / Middle / Late times.

    Critical point:
    - every run is anchored ONLY by the measured CPU sync-start edge;
    - the prediction follows the firmware timeline;
    - actual ADC edges are NOT re-aligned.

    Therefore, if measured ADC bursts progressively drift relative to the
    firmware/model, that drift remains visible from Early -> Middle -> Late.
    """
    subset = diagnostics[
        diagnostics["condition"] == condition
    ].copy()

    subset["run_number"] = (
        subset["run_id"]
        .str.extract(r"(\d+)")
        .astype(int)
    )
    subset = subset.sort_values("run_number")

    pred_t_s, pred_delta_ma = (
        build_prediction_for_active_window(
            condition,
            model_name,
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 10),
        sharey=True,
    )

    y_ticks = []
    y_labels = []

    for row_idx, row in enumerate(
        subset.itertuples(index=False)
    ):
        dataframe = read_ppk_csv(
            Path(row.source_file)
        )

        sync_start_s = float(
            row.actual_sync_start_s
        )

        idle_ma = estimate_run_idle_ma(
            dataframe,
            sync_start_s,
        )

        # Time relative to firmware-defined ADC active-window start.
        measured_active_t_s = (
            dataframe["time_s"].to_numpy(dtype=float)
            - sync_start_s
            - FIRMWARE_SYNC_TO_FIRST_BURST_S
        )

        measured_delta_ma = (
            dataframe["current_ma"].to_numpy(dtype=float)
            - idle_ma
        )

        offset = (
            len(subset) - 1 - row_idx
        ) * STACK_SPACING_MA

        y_ticks.append(offset)
        y_labels.append(row.run_id)

        for ax, (
            window_name,
            (start_s, end_s),
        ) in zip(
            axes,
            COMPARISON_WINDOWS_S.items(),
        ):
            measured_mask = (
                (measured_active_t_s >= start_s)
                & (measured_active_t_s <= end_s)
            )

            pred_mask = (
                (pred_t_s >= start_s)
                & (pred_t_s <= end_s)
            )

            # Show local time within each Early/Middle/Late window.
            mx_ms = (
                measured_active_t_s[measured_mask]
                - start_s
            ) * 1000.0

            px_ms = (
                pred_t_s[pred_mask]
                - start_s
            ) * 1000.0

            ax.plot(
                mx_ms,
                measured_delta_ma[measured_mask]
                + offset,
                linewidth=0.75,
                label=(
                    "Measured"
                    if row_idx == 0
                    else None
                ),
            )

            ax.plot(
                px_ms,
                pred_delta_ma[pred_mask]
                + offset,
                linewidth=1.0,
                linestyle="--",
                label=(
                    f"Predicted ({model_name})"
                    if row_idx == 0
                    else None
                ),
            )

            ax.set_title(
                f"{window_name}: "
                f"{start_s:.2f}–{end_s:.2f} s"
            )
            ax.set_xlabel(
                "Time within window [ms]"
            )
            ax.grid(
                True,
                axis="x",
                alpha=0.25,
            )

    axes[0].set_ylabel(
        "Run (vertical offset for display)"
    )

    for ax in axes:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    axes[-1].legend(
        loc="upper right",
    )

    fig.suptitle(
        f"{condition} | {model_name} | "
        "Measured vs predicted: Early / Middle / Late\n"
        "Sync-start anchored; no ADC-edge re-alignment"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    out_dir = (
        EARLY_MIDDLE_LATE_OUTPUT_DIR
        / condition
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / (
            f"{condition}_{model_name}_"
            "early_middle_late_stacked.png"
        )
    )

    fig.savefig(
        out_path,
        dpi=180,
    )
    plt.close(fig)

    return out_path


def analyze_timing_file(
    condition: str,
    path: Path,
) -> dict[str, object]:
    dataframe = read_ppk_csv(path)

    sync_start_s, sync_end_s, sync_mid_s = detect_sync_segment(
        dataframe
    )

    actual_first_burst_s, adc_threshold_ma = detect_first_adc_rise(
        dataframe,
        sync_start_s,
        condition,
    )

    firmware_first_burst_s = (
        sync_start_s + FIRMWARE_SYNC_TO_FIRST_BURST_S
    )

    actual_delay_s = actual_first_burst_s - sync_start_s
    timing_error_s = (
        actual_delay_s - FIRMWARE_SYNC_TO_FIRST_BURST_S
    )

    return {
        "condition": condition,
        "run_id": infer_run_id(path),
        "source_file": str(path),

        "actual_sync_start_s": sync_start_s,
        "actual_sync_end_s": sync_end_s,
        "actual_sync_mid_s": sync_mid_s,
        "actual_sync_duration_s": sync_end_s - sync_start_s,

        "firmware_sync_start_relative_s": 0.0,
        "firmware_first_burst_relative_s": (
            FIRMWARE_SYNC_TO_FIRST_BURST_S
        ),

        "firmware_first_burst_raw_s": firmware_first_burst_s,
        "actual_first_burst_raw_s": actual_first_burst_s,

        "actual_sync_to_first_burst_s": actual_delay_s,
        "timing_error_ms": timing_error_s * 1000.0,
        "abs_timing_error_ms": abs(timing_error_s) * 1000.0,

        "adc_detection_threshold_ma": adc_threshold_ma,
    }


def plot_timing_error(
    diagnostics: pd.DataFrame,
    condition: str,
) -> Path:
    subset = diagnostics[
        diagnostics["condition"] == condition
    ].copy()

    subset["run_number"] = subset["run_id"].str.extract(
        r"(\d+)"
    ).astype(int)
    subset = subset.sort_values("run_number")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )
    ax.plot(
        subset["run_number"],
        subset["timing_error_ms"],
        marker="o",
        linewidth=1.2,
    )

    ax.set_xticks(subset["run_number"])
    ax.set_xlabel("Run")
    ax.set_ylabel(
        "Measured first-burst timing error vs firmware [ms]"
    )
    ax.set_title(
        f"{condition} | actual sync start → actual first ADC burst"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    out_dir = TIMING_OUTPUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}_timing_error_by_run.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_measured_vs_firmware_delay(
    diagnostics: pd.DataFrame,
    condition: str,
) -> Path:
    subset = diagnostics[
        diagnostics["condition"] == condition
    ].copy()

    subset["run_number"] = subset["run_id"].str.extract(
        r"(\d+)"
    ).astype(int)
    subset = subset.sort_values("run_number")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.axhline(
        FIRMWARE_SYNC_TO_FIRST_BURST_S,
        linewidth=1.2,
        linestyle="--",
        label="Firmware ideal = 16.0 s",
    )
    ax.plot(
        subset["run_number"],
        subset["actual_sync_to_first_burst_s"],
        marker="o",
        linewidth=1.2,
        label="Measured",
    )

    ax.set_xticks(subset["run_number"])
    ax.set_xlabel("Run")
    ax.set_ylabel("Sync start → first ADC burst [s]")
    ax.set_title(
        f"{condition} | measured delay vs firmware ideal"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    out_dir = TIMING_OUTPUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}_measured_vs_firmware_delay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_stacked_event_windows(
    diagnostics: pd.DataFrame,
    condition: str,
) -> Path:
    """
    Visual verification plot.

    Left: actual measured sync-start edge for every run.
    Right: firmware-predicted first-burst time and measured first ADC rise.

    Each run is vertically offset only for display. Raw time differences are
    preserved; no event re-alignment is used to compute the diagnostic CSV.
    """
    subset = diagnostics[
        diagnostics["condition"] == condition
    ].copy()
    subset["run_number"] = subset["run_id"].str.extract(
        r"(\d+)"
    ).astype(int)
    subset = subset.sort_values("run_number")

    fig, (ax_sync, ax_burst) = plt.subplots(
        1,
        2,
        figsize=(16, 9),
    )

    spacing = 20.0
    y_ticks = []
    y_labels = []

    for row_idx, row in enumerate(subset.itertuples(index=False)):
        df = read_ppk_csv(Path(row.source_file))
        offset = (len(subset) - 1 - row_idx) * spacing

        # Plot sync relative to the measured sync-start edge.
        sync_t0 = row.actual_sync_start_s
        sync_mask = (
            (df["time_s"] >= sync_t0 - SYNC_PLOT_PRE_S)
            & (df["time_s"] <= sync_t0 + SYNC_PLOT_POST_S)
        )
        sx = (
            df.loc[sync_mask, "time_s"].to_numpy()
            - sync_t0
        )
        sy = df.loc[sync_mask, "current_ma"].to_numpy()
        sy = sy - np.median(sy[:max(10, len(sy)//10)])

        ax_sync.plot(
            sx,
            sy + offset,
            linewidth=0.8,
        )

        # Plot first-burst window relative to the FIRMWARE-predicted time.
        fw_first = row.firmware_first_burst_raw_s
        burst_mask = (
            (df["time_s"] >= fw_first - BURST_PLOT_PRE_S)
            & (df["time_s"] <= fw_first + BURST_PLOT_POST_S)
        )
        bx = (
            df.loc[burst_mask, "time_s"].to_numpy()
            - fw_first
        )
        by = df.loc[burst_mask, "current_ma"].to_numpy()
        by = by - np.median(by[:max(10, len(by)//10)])

        ax_burst.plot(
            bx,
            by + offset,
            linewidth=0.8,
        )

        actual_rel_to_fw = (
            row.actual_first_burst_raw_s
            - row.firmware_first_burst_raw_s
        )

        ax_burst.plot(
            [actual_rel_to_fw, actual_rel_to_fw],
            [offset - 2.0, offset + 14.0],
            linestyle=":",
            linewidth=0.9,
        )

        y_ticks.append(offset)
        y_labels.append(row.run_id)

    ax_sync.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax_sync.set_title("Measured CPU sync start")
    ax_sync.set_xlabel(
        "Time relative to detected sync start [s]"
    )
    ax_sync.set_ylabel("Run (vertical offset for display)")
    ax_sync.set_yticks(y_ticks)
    ax_sync.set_yticklabels(y_labels)
    ax_sync.grid(True, axis="x", alpha=0.25)

    ax_burst.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Firmware first-burst time",
    )
    ax_burst.set_title(
        "First ADC burst relative to firmware expectation"
    )
    ax_burst.set_xlabel(
        "Time relative to firmware first-burst time [s]"
    )
    ax_burst.set_yticks(y_ticks)
    ax_burst.set_yticklabels(y_labels)
    ax_burst.grid(True, axis="x", alpha=0.25)
    ax_burst.legend()

    fig.suptitle(
        f"{condition} | measured event timing diagnostic"
    )
    fig.tight_layout()

    out_dir = TIMING_OUTPUT_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}_stacked_sync_and_first_burst.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    TIMING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for condition, input_dir in INPUT_DIRS.items():
        for path in collect_csv_files(input_dir):
            try:
                print(
                    f"Processing {condition}: {path}",
                    flush=True,
                )
                rows.append(
                    analyze_timing_file(
                        condition,
                        path,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(exc),
                    }
                )

    if not rows:
        pd.DataFrame(errors).to_csv(
            TIMING_OUTPUT_DIR / "errors.csv",
            index=False,
        )
        raise SystemExit(
            "No files could be analyzed. See errors.csv."
        )

    diagnostics = pd.DataFrame(rows)

    diagnostics["run_number"] = (
        diagnostics["run_id"]
        .str.extract(r"(\d+)")
        .astype(int)
    )
    diagnostics = diagnostics.sort_values(
        ["condition", "run_number"]
    ).drop(columns=["run_number"])

    csv_path = (
        TIMING_OUTPUT_DIR
        / "sync_start_to_first_burst_diagnostics.csv"
    )
    diagnostics.to_csv(
        csv_path,
        index=False,
    )

    summary = (
        diagnostics.groupby("condition")
        .agg(
            n_runs=("run_id", "count"),
            measured_delay_mean_s=(
                "actual_sync_to_first_burst_s",
                "mean",
            ),
            measured_delay_std_ms=(
                "actual_sync_to_first_burst_s",
                lambda s: float(s.std(ddof=1) * 1000.0),
            ),
            timing_error_mean_ms=(
                "timing_error_ms",
                "mean",
            ),
            timing_error_std_ms=(
                "timing_error_ms",
                "std",
            ),
            timing_error_min_ms=(
                "timing_error_ms",
                "min",
            ),
            timing_error_max_ms=(
                "timing_error_ms",
                "max",
            ),
        )
        .reset_index()
    )

    summary_path = (
        TIMING_OUTPUT_DIR
        / "sync_start_to_first_burst_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n=== Timing diagnostics ===")
    print(
        diagnostics[
            [
                "condition",
                "run_id",
                "actual_sync_start_s",
                "actual_first_burst_raw_s",
                "actual_sync_to_first_burst_s",
                "timing_error_ms",
            ]
        ].to_string(index=False)
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    for condition in sorted(diagnostics["condition"].unique()):
        p1 = plot_timing_error(
            diagnostics,
            condition,
        )
        p2 = plot_measured_vs_firmware_delay(
            diagnostics,
            condition,
        )
        p3 = plot_stacked_event_windows(
            diagnostics,
            condition,
        )

        print(f"Wrote: {p1}")
        print(f"Wrote: {p2}")
        print(f"Wrote: {p3}")

        for model_name in MODELS_TO_PLOT:
            p4 = plot_early_middle_late_measured_vs_predicted(
                diagnostics,
                condition,
                model_name,
            )
            print(f"Wrote: {p4}")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {summary_path}")

    if errors:
        errors_path = TIMING_OUTPUT_DIR / "errors.csv"
        pd.DataFrame(errors).to_csv(
            errors_path,
            index=False,
        )
        print(f"Wrote: {errors_path}")


if __name__ == "__main__":
    main()