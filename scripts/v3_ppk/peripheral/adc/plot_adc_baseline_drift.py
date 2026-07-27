from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Fixed paths
# ============================================================

RAW_PATHS = {
    "idle_baseline": Path("data/raw/v3_ppk/idle_baseline"),
    "adc_init_only": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_init_only"
    ),
    "adc_burst_100ms_10samples": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_burst/"
        "adc_burst_100ms_10samples"
    ),
    "adc_burst_100ms_100samples": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_burst/"
        "adc_burst_100ms_100samples"
    ),
    "adc_burst_100ms_1000samples": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_burst/"
        "adc_burst_100ms_1000samples"
    ),
    "adc_single_1ms": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_single/"
        "adc_single_1ms"
    ),
    "adc_single_10ms": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_single/"
        "adc_single_10ms"
    ),
    "adc_single_100ms": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_periodic_single/"
        "adc_single_100ms"
    ),
}

OUTPUT_DIR = Path(
    "data/processed/v3_ppk/peripheral/adc/baseline_drift_plots"
)


# ============================================================
# Experiment timing relative to sync-pulse midpoint
# ============================================================

# sync pulse:      -0.5 to +0.5 s
# recovery idle:   +0.5 to +5.5 s
# initial idle:    +5.5 to +15.5 s
# active/test:     +15.5 to +35.5 s
# final idle:      +35.5 to +45.5 s

INITIAL_IDLE_WINDOW_S = (5.5, 15.5)
ACTIVE_WINDOW_S = (15.5, 35.5)
FINAL_IDLE_WINDOW_S = (35.5, 45.5)

PHASE_GUARD_S = 0.5

SYNC_DURATION_S = 1.0
SYNC_SEARCH_START_S = 1.0
SYNC_SEARCH_END_S = 10.0
SYNC_DETECTION_BIN_S = 0.001

# 10 ms plotting bins:
# enough to show baseline movement without plotting millions of raw samples.
PLOT_BIN_S = 0.010

PLOT_START_S = -1.0
PLOT_END_S = 46.0


# ============================================================
# CSV loading
# ============================================================

def sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        sample = file.read(8192)

    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=",;\t",
        ).delimiter
    except csv.Error:
        return ","


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_column(
    columns: list[str],
    candidates: tuple[str, ...],
    label: str,
) -> str:
    normalized = {
        column: normalize_column_name(column)
        for column in columns
    }

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        for column, column_normalized in normalized.items():
            if column_normalized == candidate_normalized:
                return column

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        for column, column_normalized in normalized.items():
            if candidate_normalized in column_normalized:
                return column

    raise ValueError(
        f"Could not find {label} column. "
        f"Available columns: {columns}"
    )


def convert_time_to_seconds(
    series: pd.Series,
    column_name: str,
) -> np.ndarray:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    name = column_name.lower()

    if "µs" in name or "us" in name or "micro" in name:
        return values * 1e-6

    if "ms" in name or "milli" in name:
        return values * 1e-3

    if "ns" in name or "nano" in name:
        return values * 1e-9

    finite = values[np.isfinite(values)]

    if finite.size < 2:
        raise ValueError("Not enough timestamps to infer time unit.")

    median_step = float(np.median(np.diff(finite)))

    if median_step >= 0.01:
        return values * 1e-3

    return values


def convert_current_to_mA(
    series: pd.Series,
    column_name: str,
) -> np.ndarray:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    name = column_name.lower()

    if "µa" in name or "ua" in name or "microamp" in name:
        return values / 1000.0

    if "(a)" in name or name.strip().endswith("_a"):
        return values * 1000.0

    return values


def load_ppk_csv(path: Path) -> pd.DataFrame:
    delimiter = sniff_delimiter(path)

    dataframe = pd.read_csv(
        path,
        sep=delimiter,
        encoding="utf-8-sig",
    )

    time_column = find_column(
        list(dataframe.columns),
        (
            "timestamp_ms",
            "timestamp",
            "time_ms",
            "time",
            "elapsed_ms",
            "elapsed_time",
        ),
        "timestamp",
    )

    current_column = find_column(
        list(dataframe.columns),
        (
            "current_mA",
            "current",
            "average_current",
            "smoothed_current",
        ),
        "current",
    )

    loaded = pd.DataFrame(
        {
            "time_s": convert_time_to_seconds(
                dataframe[time_column],
                time_column,
            ),
            "current_mA": convert_current_to_mA(
                dataframe[current_column],
                current_column,
            ),
        }
    )

    loaded = (
        loaded
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("time_s")
        .drop_duplicates("time_s")
        .reset_index(drop=True)
    )

    if loaded.empty:
        raise ValueError(f"No valid data in {path}")

    loaded["time_s"] -= float(loaded["time_s"].iloc[0])

    return loaded


# ============================================================
# Sync detection and binning
# ============================================================

def bin_mean(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    bin_s: float,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(time_s)
        & np.isfinite(current_mA)
        & (time_s >= start_s)
        & (time_s <= end_s)
    )

    selected_time = time_s[mask]
    selected_current = current_mA[mask]

    if selected_time.size == 0:
        raise ValueError("No samples in requested binning interval.")

    indices = np.floor(
        (selected_time - start_s) / bin_s
    ).astype(np.int64)

    counts = np.bincount(indices)
    sums = np.bincount(indices, weights=selected_current)

    valid = counts > 0

    centres = (
        start_s
        + (np.flatnonzero(valid) + 0.5) * bin_s
    )
    means = sums[valid] / counts[valid]

    return centres, means


def detect_sync_midpoint(dataframe: pd.DataFrame) -> float:
    time_s = dataframe["time_s"].to_numpy()
    current_mA = dataframe["current_mA"].to_numpy()

    search_end_s = min(
        SYNC_SEARCH_END_S,
        float(time_s[-1]),
    )

    centres, means = bin_mean(
        time_s,
        current_mA,
        SYNC_DETECTION_BIN_S,
        SYNC_SEARCH_START_S,
        search_end_s,
    )

    window_bins = max(
        1,
        int(round(
            SYNC_DURATION_S / SYNC_DETECTION_BIN_S
        )),
    )

    if means.size < window_bins:
        raise ValueError(
            "Recording is too short for sync-pulse detection."
        )

    rolling_mean = np.convolve(
        means,
        np.ones(window_bins, dtype=np.float64) / window_bins,
        mode="valid",
    )

    best_index = int(np.argmax(rolling_mean))

    pulse_start_s = (
        float(centres[best_index])
        - SYNC_DETECTION_BIN_S / 2.0
    )

    return pulse_start_s + SYNC_DURATION_S / 2.0


# ============================================================
# Phase statistics
# ============================================================

def phase_mean(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    window_s: tuple[float, float],
) -> float:
    start_s = window_s[0] + PHASE_GUARD_S
    end_s = window_s[1] - PHASE_GUARD_S

    mask = (
        (time_s >= start_s)
        & (time_s < end_s)
        & np.isfinite(current_mA)
    )

    if not np.any(mask):
        raise ValueError(
            f"No samples in phase window {window_s}"
        )

    return float(np.mean(current_mA[mask]))


# ============================================================
# Run processing
# ============================================================

def find_csv_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*.csv")
        if path.is_file()
    )


def process_run(path: Path) -> dict[str, object]:
    dataframe = load_ppk_csv(path)

    sync_midpoint_s = detect_sync_midpoint(dataframe)

    aligned_time_s = (
        dataframe["time_s"].to_numpy()
        - sync_midpoint_s
    )
    current_mA = dataframe["current_mA"].to_numpy()

    initial_mean_mA = phase_mean(
        aligned_time_s,
        current_mA,
        INITIAL_IDLE_WINDOW_S,
    )
    active_mean_mA = phase_mean(
        aligned_time_s,
        current_mA,
        ACTIVE_WINDOW_S,
    )
    final_mean_mA = phase_mean(
        aligned_time_s,
        current_mA,
        FINAL_IDLE_WINDOW_S,
    )

    plot_time_s, plot_current_mA = bin_mean(
        aligned_time_s,
        current_mA,
        PLOT_BIN_S,
        PLOT_START_S,
        PLOT_END_S,
    )

    return {
        "file": path.name,
        "sync_midpoint_s": sync_midpoint_s,
        "time_s": plot_time_s,
        "current_mA": plot_current_mA,
        "initial_mean_mA": initial_mean_mA,
        "active_mean_mA": active_mean_mA,
        "final_mean_mA": final_mean_mA,
        "drift_mA": final_mean_mA - initial_mean_mA,
    }


def common_grid() -> np.ndarray:
    return np.arange(
        PLOT_START_S + PLOT_BIN_S / 2.0,
        PLOT_END_S,
        PLOT_BIN_S,
    )


def interpolate_to_common_grid(
    run: dict[str, object],
    grid_s: np.ndarray,
) -> np.ndarray:
    time_s = np.asarray(run["time_s"], dtype=np.float64)
    current_mA = np.asarray(
        run["current_mA"],
        dtype=np.float64,
    )

    return np.interp(
        grid_s,
        time_s,
        current_mA,
        left=np.nan,
        right=np.nan,
    )


# ============================================================
# Plotting
# ============================================================

def add_phase_boundaries(ax: plt.Axes) -> None:
    boundaries = [
        -0.5,
        0.5,
        INITIAL_IDLE_WINDOW_S[0],
        INITIAL_IDLE_WINDOW_S[1],
        ACTIVE_WINDOW_S[1],
        FINAL_IDLE_WINDOW_S[1],
    ]

    for boundary in boundaries:
        ax.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )


def add_phase_labels(ax: plt.Axes) -> None:
    y_min, y_max = ax.get_ylim()
    label_y = y_max - 0.03 * (y_max - y_min)

    labels = [
        (0.0, "Sync"),
        (
            sum(INITIAL_IDLE_WINDOW_S) / 2.0,
            "Initial idle",
        ),
        (
            sum(ACTIVE_WINDOW_S) / 2.0,
            "Active",
        ),
        (
            sum(FINAL_IDLE_WINDOW_S) / 2.0,
            "Final idle",
        ),
    ]

    for x_position, label in labels:
        ax.text(
            x_position,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=9,
        )


def add_phase_mean_lines(
    ax: plt.Axes,
    initial_mean_mA: float,
    active_mean_mA: float,
    final_mean_mA: float,
) -> None:
    ax.hlines(
        initial_mean_mA,
        INITIAL_IDLE_WINDOW_S[0] + PHASE_GUARD_S,
        INITIAL_IDLE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Initial mean = {initial_mean_mA:.3f} mA",
    )

    ax.hlines(
        active_mean_mA,
        ACTIVE_WINDOW_S[0] + PHASE_GUARD_S,
        ACTIVE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Active mean = {active_mean_mA:.3f} mA",
    )

    ax.hlines(
        final_mean_mA,
        FINAL_IDLE_WINDOW_S[0] + PHASE_GUARD_S,
        FINAL_IDLE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Final mean = {final_mean_mA:.3f} mA",
    )


def plot_condition(
    condition: str,
    runs: list[dict[str, object]],
) -> None:
    grid_s = common_grid()

    interpolated_runs = np.vstack(
        [
            interpolate_to_common_grid(run, grid_s)
            for run in runs
        ]
    )

    mean_waveform_mA = np.nanmean(
        interpolated_runs,
        axis=0,
    )

    initial_mean_mA = float(
        np.mean([
            run["initial_mean_mA"]
            for run in runs
        ])
    )
    active_mean_mA = float(
        np.mean([
            run["active_mean_mA"]
            for run in runs
        ])
    )
    final_mean_mA = float(
        np.mean([
            run["final_mean_mA"]
            for run in runs
        ])
    )

    drift_mean_mA = final_mean_mA - initial_mean_mA

    fig, ax = plt.subplots(figsize=(14, 6))

    for run_waveform in interpolated_runs:
        ax.plot(
            grid_s,
            run_waveform,
            linewidth=0.6,
            alpha=0.25,
        )

    ax.plot(
        grid_s,
        mean_waveform_mA,
        linewidth=1.8,
        label="10-run mean waveform",
    )

    add_phase_boundaries(ax)

    add_phase_mean_lines(
        ax,
        initial_mean_mA,
        active_mean_mA,
        final_mean_mA,
    )

    ax.set_xlim(PLOT_START_S, PLOT_END_S)
    ax.set_xlabel("Aligned time from sync-pulse midpoint (s)")
    ax.set_ylabel("Current (mA)")
    ax.set_title(
        f"{condition}\n"
        f"Initial-to-final drift = {drift_mean_mA:+.3f} mA"
    )
    ax.grid(True, alpha=0.25)

    add_phase_labels(ax)

    ax.legend(
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{condition}_aligned_drift.png"

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)

    print(f"[SAVE] {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for condition, folder in RAW_PATHS.items():
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        csv_files = find_csv_files(folder)

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in: {folder}"
            )

        print(
            f"[INFO] {condition}: "
            f"{len(csv_files)} file(s)"
        )

        runs = [
            process_run(csv_path)
            for csv_path in csv_files
        ]

        plot_condition(
            condition,
            runs,
        )

    print("\n[OK] ADC baseline drift plots completed.")


if __name__ == "__main__":
    main()