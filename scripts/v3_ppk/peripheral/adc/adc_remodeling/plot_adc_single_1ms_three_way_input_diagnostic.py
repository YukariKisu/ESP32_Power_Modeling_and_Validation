from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

RAW_ROOT = Path("data/raw/v3_ppk/peripheral/adc/final_predictioned")

SINGLE_1MS_DIR = (
    RAW_ROOT
    / "adc_periodic_single"
    / "adc_single_1ms"
)

OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "single_1ms_input_definition_diagnostic"
)

# ADC-specific remodeled parameters.
DELTA_I_MA = 12.943
TAU_RISE_S = 0.000579
TAU_FALL_S = 0.000579

# Original Single-1ms input definition.
EVENT_INTERVAL_S = 0.001
EVENT_DURATION_S = 44e-6

# Firmware timeline from measured CPU sync start.
SYNC_TO_ACTIVE_START_S = 16.0
ACTIVE_DURATION_S = 20.0

# Uniform prediction/averaging grid.
DT_S = 0.00001

# Plot windows relative to ADC active start.
START_WINDOW_S = (-0.010, 0.030)
ZOOM_WINDOW_S = (5.000, 5.010)
FULL_WINDOW_S = (-0.020, 0.100)

# Sync detection.
SYNC_DURATION_S = 1.0
SYNC_SEARCH_START_S = 1.0
SYNC_SEARCH_END_S = 6.0
SYNC_SMOOTH_WINDOW_S = 0.020

# Measured idle baseline.
BASELINE_START_S = 6.2
BASELINE_END_S = 15.8


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


# ============================================================
# File helpers
# ============================================================

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
        f"Could not find {kind} column in {list(columns)}"
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

    if "ua" in name or "µa" in name:
        return values / 1000.0

    if "ma" in name:
        return values

    if re.search(r"\ba\b", name) or "(a" in name or "[a" in name:
        return values * 1000.0

    median_abs = float(
        np.nanmedian(np.abs(values))
    )

    if median_abs < 1.0:
        return values * 1000.0

    if median_abs > 1000.0:
        return values / 1000.0

    return values


def read_ppk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        comment="#",
    )

    if df.empty:
        raise ValueError(
            f"CSV is empty: {path}"
        )

    time_col = find_column(
        df.columns,
        TIME_COLUMN_CANDIDATES,
        "time",
    )

    current_col = find_column(
        df.columns,
        CURRENT_COLUMN_CANDIDATES,
        "current",
    )

    time_raw = pd.to_numeric(
        df[time_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    current_raw = pd.to_numeric(
        df[current_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(time_raw)
        & np.isfinite(current_raw)
    )

    time_s = convert_time_to_s(
        time_raw[valid],
        time_col,
    )

    time_s = time_s - time_s[0]

    current_ma = convert_current_to_ma(
        current_raw[valid],
        current_col,
    )

    return pd.DataFrame(
        {
            "time_s": time_s,
            "current_ma": current_ma,
        }
    )


def collect_csv_files(
    folder: Path,
) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Input directory not found: {folder}"
        )

    files = list(
        folder.glob("*.csv")
    )

    def sort_key(path: Path):
        match = re.search(
            r"run[_-]?(\d+)",
            path.stem,
            flags=re.IGNORECASE,
        )

        return (
            int(match.group(1))
            if match
            else 999999,
            path.name,
        )

    files = sorted(
        files,
        key=sort_key,
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {folder}"
        )

    return files


def rolling_mean(
    values: np.ndarray,
    samples: int,
) -> np.ndarray:
    samples = max(
        1,
        int(samples),
    )

    if samples == 1:
        return values

    kernel = (
        np.ones(samples, dtype=float)
        / samples
    )

    return np.convolve(
        values,
        kernel,
        mode="same",
    )


# ============================================================
# Sync-start alignment
# ============================================================

def detect_sync_start(
    dataframe: pd.DataFrame,
) -> float:
    """
    Detect only the measured CPU sync pulse.
    ADC waveform is not used for alignment.
    """
    time_s = dataframe[
        "time_s"
    ].to_numpy(dtype=float)

    current_ma = dataframe[
        "current_ma"
    ].to_numpy(dtype=float)

    dt_s = float(
        np.median(
            np.diff(time_s)
        )
    )

    smooth_n = max(
        1,
        round(
            SYNC_SMOOTH_WINDOW_S
            / dt_s
        ),
    )

    smooth = rolling_mean(
        current_ma,
        smooth_n,
    )

    mask = (
        (time_s >= SYNC_SEARCH_START_S)
        & (time_s <= SYNC_SEARCH_END_S)
    )

    t = time_s[mask]
    y = smooth[mask]

    if len(t) < 10:
        raise ValueError(
            "Sync search window contains too few samples"
        )

    baseline = float(
        np.percentile(y, 20)
    )

    peak = float(
        np.percentile(y, 99)
    )

    threshold = (
        baseline
        + 0.45
        * (peak - baseline)
    )

    high = y >= threshold

    segments: list[tuple[int, int]] = []
    start_idx = None

    for i, is_high in enumerate(high):
        if is_high and start_idx is None:
            start_idx = i

        elif (
            not is_high
            and start_idx is not None
        ):
            segments.append(
                (start_idx, i - 1)
            )
            start_idx = None

    if start_idx is not None:
        segments.append(
            (start_idx, len(high) - 1)
        )

    if not segments:
        raise ValueError(
            "No sync-like segment detected"
        )

    def score(
        segment: tuple[int, int],
    ) -> tuple[float, float]:
        i0, i1 = segment

        duration_s = float(
            t[i1] - t[i0]
        )

        strength = float(
            np.mean(
                y[i0:i1 + 1]
            )
            - baseline
        )

        return (
            abs(
                duration_s
                - SYNC_DURATION_S
            ),
            -strength,
        )

    i0, _ = min(
        segments,
        key=score,
    )

    return float(
        t[i0]
    )


# ============================================================
# Measured Single-1ms mean waveform
# ============================================================

def prepare_measured_run(
    path: Path,
) -> dict[str, np.ndarray | float | str]:
    dataframe = read_ppk_csv(
        path
    )

    sync_start_s = detect_sync_start(
        dataframe
    )

    aligned_s = (
        dataframe["time_s"].to_numpy(dtype=float)
        - sync_start_s
    )

    current_ma = dataframe[
        "current_ma"
    ].to_numpy(dtype=float)

    baseline_mask = (
        (aligned_s >= BASELINE_START_S)
        & (aligned_s <= BASELINE_END_S)
    )

    if not np.any(
        baseline_mask
    ):
        raise ValueError(
            f"No idle baseline samples in {path}"
        )

    baseline_ma = float(
        np.mean(
            current_ma[
                baseline_mask
            ]
        )
    )

    return {
        "source_file": str(path),
        "time_active_s": (
            aligned_s
            - SYNC_TO_ACTIVE_START_S
        ),
        "delta_ma": (
            current_ma
            - baseline_ma
        ),
        "baseline_ma": baseline_ma,
    }


def interpolate_measured_mean(
    runs: list[
        dict[
            str,
            np.ndarray | float | str
        ]
    ],
    start_s: float,
    end_s: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    grid_s = np.arange(
        start_s,
        end_s + DT_S / 2,
        DT_S,
    )

    traces = []

    for run in runs:
        time_s = np.asarray(
            run["time_active_s"],
            dtype=float,
        )

        delta_ma = np.asarray(
            run["delta_ma"],
            dtype=float,
        )

        traces.append(
            np.interp(
                grid_s,
                time_s,
                delta_ma,
                left=np.nan,
                right=np.nan,
            )
        )

    matrix = np.vstack(
        traces
    )

    return (
        grid_s,
        np.nanmean(
            matrix,
            axis=0,
        ),
        np.nanstd(
            matrix,
            axis=0,
            ddof=1,
        ),
    )


# ============================================================
# Predictions
# ============================================================

def make_original_pulse_input(
    time_s: np.ndarray,
) -> np.ndarray:
    """
    Original assumption:
    one 44-us active pulse every 1 ms.
    """
    u = np.zeros_like(
        time_s,
        dtype=float,
    )

    active_mask = (
        (time_s >= 0.0)
        & (time_s < ACTIVE_DURATION_S)
    )

    phase_s = np.mod(
        time_s[
            active_mask
        ],
        EVENT_INTERVAL_S,
    )

    u[
        active_mask
    ] = (
        phase_s
        < EVENT_DURATION_S
    ).astype(float)

    return u


def make_continuous_input(
    time_s: np.ndarray,
) -> np.ndarray:
    """
    Diagnostic alternative:
    u(t)=1 during the whole ADC active window.
    """
    return (
        (time_s >= 0.0)
        & (time_s < ACTIVE_DURATION_S)
    ).astype(float)


def simulate_first_order_ode(
    time_s: np.ndarray,
    input_u: np.ndarray,
) -> np.ndarray:
    prediction_ma = np.zeros_like(
        time_s,
        dtype=float,
    )

    for i in range(
        1,
        len(time_s),
    ):
        target_ma = (
            DELTA_I_MA
            * input_u[i]
        )

        tau_s = (
            TAU_RISE_S
            if (
                target_ma
                >= prediction_ma[
                    i - 1
                ]
            )
            else TAU_FALL_S
        )

        dt_s = (
            time_s[i]
            - time_s[i - 1]
        )

        alpha = (
            1.0
            - math.exp(
                -dt_s / tau_s
            )
        )

        prediction_ma[i] = (
            prediction_ma[
                i - 1
            ]
            + alpha
            * (
                target_ma
                - prediction_ma[
                    i - 1
                ]
            )
        )

    return prediction_ma


def build_predictions(
    start_s: float,
    end_s: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    time_s = np.arange(
        start_s,
        end_s + DT_S / 2,
        DT_S,
    )

    original_u = (
        make_original_pulse_input(
            time_s
        )
    )

    continuous_u = (
        make_continuous_input(
            time_s
        )
    )

    original_prediction = (
        simulate_first_order_ode(
            time_s,
            original_u,
        )
    )

    continuous_prediction = (
        simulate_first_order_ode(
            time_s,
            continuous_u,
        )
    )

    return (
        time_s,
        original_prediction,
        continuous_prediction,
    )


# ============================================================
# Plots
# ============================================================

def plot_three_way(
    measured_runs,
    start_s: float,
    end_s: float,
    title: str,
    filename: str,
    local_time_ms: bool = False,
) -> Path:

    measured_t, measured_mean, _ = (
        interpolate_measured_mean(
            measured_runs,
            start_s,
            end_s,
        )
    )

    pred_t, original_pred, continuous_pred = (
        build_predictions(
            start_s,
            end_s,
        )
    )

    if local_time_ms:
        x_measured = (
            measured_t
            - start_s
        ) * 1000.0

        x_pred = (
            pred_t
            - start_s
        ) * 1000.0

        xlabel = (
            f"Time within "
            f"{start_s:.3f}–{end_s:.3f} s "
            f"window [ms]"
        )

    else:
        x_measured = (
            measured_t
            * 1000.0
        )

        x_pred = (
            pred_t
            * 1000.0
        )

        xlabel = (
            "Time relative to firmware "
            "ADC active start [ms]"
        )

    fig, ax = plt.subplots(
        figsize=(13, 6.5)
    )

    ax.plot(
        x_measured,
        measured_mean,
        linewidth=1.2,
        label="Measured Single 1 ms",
    )

    ax.plot(
        x_pred,
        original_pred,
        linewidth=1.1,
        linestyle="--",
        label=(
            "Original prediction "
            "(44 µs pulse / 1 ms)"
        ),
    )

    ax.plot(
        x_pred,
        continuous_pred,
        linewidth=1.1,
        linestyle="-.",
        label=(
            "Continuous-input prediction "
            "(u = 1)"
        ),
    )

    if (
        start_s
        <= 0.0
        <= end_s
        and not local_time_ms
    ):
        ax.axvline(
            0.0,
            linestyle=":",
            linewidth=1.0,
            label=(
                "Firmware active start"
            ),
        )

    if local_time_ms:
        for k in range(
            int(
                round(
                    (end_s - start_s)
                    * 1000.0
                )
            )
            + 1
        ):
            ax.axvline(
                float(k),
                linestyle=":",
                linewidth=0.6,
                alpha=0.25,
            )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Current increase from "
        "idle baseline [mA]"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        out_path,
        dpi=180,
    )

    plt.close(
        fig
    )

    return out_path


# ============================================================
# Numeric diagnostic
# ============================================================

def calculate_summary(
    measured_runs,
) -> pd.DataFrame:
    # Compare steady active period only,
    # avoiding start/end transitions.
    start_s = 1.0
    end_s = 19.0

    measured_t, measured_mean, _ = (
        interpolate_measured_mean(
            measured_runs,
            start_s,
            end_s,
        )
    )

    pred_t, original_pred, continuous_pred = (
        build_predictions(
            start_s,
            end_s,
        )
    )

    # Grids are built identically.
    assert len(
        measured_t
    ) == len(
        pred_t
    )

    measured_level = float(
        np.mean(
            measured_mean
        )
    )

    original_level = float(
        np.mean(
            original_pred
        )
    )

    continuous_level = float(
        np.mean(
            continuous_pred
        )
    )

    return pd.DataFrame(
        [
            {
                "series": (
                    "Measured Single 1 ms"
                ),
                "mean_delta_i_ma": (
                    measured_level
                ),
                "difference_vs_measured_ma": 0.0,
            },
            {
                "series": (
                    "Original 44us pulse prediction"
                ),
                "mean_delta_i_ma": (
                    original_level
                ),
                "difference_vs_measured_ma": (
                    original_level
                    - measured_level
                ),
            },
            {
                "series": (
                    "Continuous-input prediction"
                ),
                "mean_delta_i_ma": (
                    continuous_level
                ),
                "difference_vs_measured_ma": (
                    continuous_level
                    - measured_level
                ),
            },
        ]
    )


def main() -> None:
    files = collect_csv_files(
        SINGLE_1MS_DIR
    )

    print(
        f"Single 1 ms files: {len(files)}"
    )

    measured_runs = [
        prepare_measured_run(
            path
        )
        for path in files
    ]

    p1 = plot_three_way(
        measured_runs,
        start_s=START_WINDOW_S[0],
        end_s=START_WINDOW_S[1],
        title=(
            "ADC Single 1 ms — "
            "Measured vs original pulse input "
            "vs continuous input"
        ),
        filename=(
            "single_1ms_three_way_active_start.png"
        ),
        local_time_ms=False,
    )

    p2 = plot_three_way(
        measured_runs,
        start_s=ZOOM_WINDOW_S[0],
        end_s=ZOOM_WINDOW_S[1],
        title=(
            "ADC Single 1 ms — "
            "10 ms steady-state zoom"
        ),
        filename=(
            "single_1ms_three_way_10ms_zoom.png"
        ),
        local_time_ms=True,
    )

    p3 = plot_three_way(
        measured_runs,
        start_s=FULL_WINDOW_S[0],
        end_s=FULL_WINDOW_S[1],
        title=(
            "ADC Single 1 ms — "
            "120 ms active-start view"
        ),
        filename=(
            "single_1ms_three_way_120ms.png"
        ),
        local_time_ms=False,
    )

    summary = calculate_summary(
        measured_runs
    )

    summary_path = (
        OUTPUT_DIR
        / "single_1ms_three_way_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n=== Three-way steady-state comparison ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nWrote: {p1}"
    )
    print(
        f"Wrote: {p2}"
    )
    print(
        f"Wrote: {p3}"
    )
    print(
        f"Wrote: {summary_path}"
    )


if __name__ == "__main__":
    main()