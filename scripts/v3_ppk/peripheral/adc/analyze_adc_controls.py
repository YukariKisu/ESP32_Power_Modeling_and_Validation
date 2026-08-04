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
    "adc_wait_only": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_wait_only"
    ),
    "adc_one_read": Path(
        "data/raw/v3_ppk/peripheral/adc/adc_one_read"
    ),
}

OUTPUT_DIR = Path(
    "data/processed/v3_ppk/peripheral/adc/adc_controls"
)

PLOT_DIR = OUTPUT_DIR / "plots"


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

def select_phase_values(
    aligned_time_s: np.ndarray,
    current_mA: np.ndarray,
    window_s: tuple[float, float],
) -> np.ndarray:
    start_s = window_s[0] + PHASE_GUARD_S
    end_s = window_s[1] - PHASE_GUARD_S

    mask = (
        (aligned_time_s >= start_s)
        & (aligned_time_s < end_s)
        & np.isfinite(current_mA)
    )

    values = current_mA[mask]

    if values.size < 2:
        raise ValueError(
            f"Not enough samples in phase window {window_s}"
        )

    return values


def phase_stats(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
    )


def extract_run_number(path: Path) -> int | str:
    match = re.search(
        r"run[_-]?0*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return path.stem


# ============================================================
# Run analysis
# ============================================================

def analyse_run(
    condition: str,
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    dataframe = load_ppk_csv(path)

    sync_midpoint_s = detect_sync_midpoint(dataframe)

    aligned_time_s = (
        dataframe["time_s"].to_numpy()
        - sync_midpoint_s
    )
    current_mA = dataframe["current_mA"].to_numpy()

    initial_values = select_phase_values(
        aligned_time_s,
        current_mA,
        INITIAL_IDLE_WINDOW_S,
    )
    active_values = select_phase_values(
        aligned_time_s,
        current_mA,
        ACTIVE_WINDOW_S,
    )
    final_values = select_phase_values(
        aligned_time_s,
        current_mA,
        FINAL_IDLE_WINDOW_S,
    )

    initial_mean_mA, initial_std_mA = phase_stats(
        initial_values
    )
    active_mean_mA, active_std_mA = phase_stats(
        active_values
    )
    final_mean_mA, final_std_mA = phase_stats(
        final_values
    )

    adc_idle_mean_mA = (
        initial_mean_mA + final_mean_mA
    ) / 2.0

    delta_I_mA = active_mean_mA - adc_idle_mean_mA
    drift_mA = final_mean_mA - initial_mean_mA

    plot_time_s, plot_current_mA = bin_mean(
        aligned_time_s,
        current_mA,
        PLOT_BIN_S,
        PLOT_START_S,
        PLOT_END_S,
    )

    result = {
        "condition": condition,
        "run": extract_run_number(path),
        "file": path.name,
        "sync_midpoint_s": sync_midpoint_s,

        "initial_idle_mean_mA": initial_mean_mA,
        "initial_idle_std_mA": initial_std_mA,

        "active_mean_mA": active_mean_mA,
        "active_std_mA": active_std_mA,

        "final_idle_mean_mA": final_mean_mA,
        "final_idle_std_mA": final_std_mA,

        "adc_idle_mean_mA": adc_idle_mean_mA,
        "delta_I_mA": delta_I_mA,
        "initial_to_final_drift_mA": drift_mA,
    }

    plot_data = {
        "time_s": plot_time_s,
        "current_mA": plot_current_mA,
    }

    return result, plot_data


# ============================================================
# Summary
# ============================================================

def sample_sd(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return float("nan")

    return float(values.std(ddof=1))


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition, group in per_run.groupby(
        "condition",
        sort=False,
    ):
        rows.append(
            {
                "condition": condition,
                "n_runs": int(len(group)),

                "initial_idle_mean_mA":
                    float(group["initial_idle_mean_mA"].mean()),
                "initial_idle_std_mean_mA":
                    float(group["initial_idle_std_mA"].mean()),

                "active_mean_mA":
                    float(group["active_mean_mA"].mean()),
                "active_std_mean_mA":
                    float(group["active_std_mA"].mean()),

                "final_idle_mean_mA":
                    float(group["final_idle_mean_mA"].mean()),
                "final_idle_std_mean_mA":
                    float(group["final_idle_std_mA"].mean()),

                "delta_I_mean_mA":
                    float(group["delta_I_mA"].mean()),
                "delta_I_sd_mA":
                    sample_sd(group["delta_I_mA"]),

                "initial_to_final_drift_mean_mA":
                    float(
                        group["initial_to_final_drift_mA"].mean()
                    ),
                "initial_to_final_drift_sd_mA":
                    sample_sd(
                        group["initial_to_final_drift_mA"]
                    ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["delta_I_mean_plus_minus_sd"] = summary.apply(
        lambda row: (
            f"{row['delta_I_mean_mA']:.6f} ± "
            f"{row['delta_I_sd_mA']:.6f} mA"
        ),
        axis=1,
    )

    summary["drift_mean_plus_minus_sd"] = summary.apply(
        lambda row: (
            f"{row['initial_to_final_drift_mean_mA']:.6f} ± "
            f"{row['initial_to_final_drift_sd_mA']:.6f} mA"
        ),
        axis=1,
    )

    return summary


# ============================================================
# Plotting
# ============================================================

def common_grid() -> np.ndarray:
    return np.arange(
        PLOT_START_S + PLOT_BIN_S / 2.0,
        PLOT_END_S,
        PLOT_BIN_S,
    )


def interpolate_to_grid(
    plot_data: dict[str, np.ndarray],
    grid_s: np.ndarray,
) -> np.ndarray:
    return np.interp(
        grid_s,
        plot_data["time_s"],
        plot_data["current_mA"],
        left=np.nan,
        right=np.nan,
    )


def plot_condition(
    condition: str,
    run_results: list[dict[str, object]],
    plot_runs: list[dict[str, np.ndarray]],
) -> None:
    grid_s = common_grid()

    waveforms = np.vstack(
        [
            interpolate_to_grid(plot_data, grid_s)
            for plot_data in plot_runs
        ]
    )

    mean_waveform = np.nanmean(
        waveforms,
        axis=0,
    )

    initial_mean = float(np.mean(
        [row["initial_idle_mean_mA"] for row in run_results]
    ))
    active_mean = float(np.mean(
        [row["active_mean_mA"] for row in run_results]
    ))
    final_mean = float(np.mean(
        [row["final_idle_mean_mA"] for row in run_results]
    ))
    drift_mean = final_mean - initial_mean

    fig, ax = plt.subplots(figsize=(14, 6))

    for waveform in waveforms:
        ax.plot(
            grid_s,
            waveform,
            linewidth=0.7,
            alpha=0.35,
        )

    ax.plot(
        grid_s,
        mean_waveform,
        linewidth=2.0,
        label="Mean waveform",
    )

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

    ax.hlines(
        initial_mean,
        INITIAL_IDLE_WINDOW_S[0] + PHASE_GUARD_S,
        INITIAL_IDLE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Initial mean = {initial_mean:.3f} mA",
    )

    ax.hlines(
        active_mean,
        ACTIVE_WINDOW_S[0] + PHASE_GUARD_S,
        ACTIVE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Active mean = {active_mean:.3f} mA",
    )

    ax.hlines(
        final_mean,
        FINAL_IDLE_WINDOW_S[0] + PHASE_GUARD_S,
        FINAL_IDLE_WINDOW_S[1] - PHASE_GUARD_S,
        linewidth=2.0,
        label=f"Final mean = {final_mean:.3f} mA",
    )

    ax.set_xlim(PLOT_START_S, PLOT_END_S)
    ax.set_xlabel("Aligned time from sync-pulse midpoint (s)")
    ax.set_ylabel("Current (mA)")
    ax.set_title(
        f"{condition}\n"
        f"Initial-to-final drift = {drift_mean:+.3f} mA"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    output_path = PLOT_DIR / f"{condition}_aligned.png"

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)

    print(f"[SAVE] {output_path}")


# ============================================================
# Main
# ============================================================

def find_csv_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*.csv")
        if path.is_file()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: list[dict[str, object]] = []

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

        condition_rows: list[dict[str, object]] = []
        condition_plots: list[dict[str, np.ndarray]] = []

        for csv_path in csv_files:
            result, plot_data = analyse_run(
                condition,
                csv_path,
            )

            all_rows.append(result)
            condition_rows.append(result)
            condition_plots.append(plot_data)

            print(
                f"  {csv_path.name}: "
                f"initial={result['initial_idle_mean_mA']:.6f} mA, "
                f"active={result['active_mean_mA']:.6f} mA, "
                f"final={result['final_idle_mean_mA']:.6f} mA, "
                f"delta={result['delta_I_mA']:.6f} mA, "
                f"drift={result['initial_to_final_drift_mA']:.6f} mA"
            )

        plot_condition(
            condition,
            condition_rows,
            condition_plots,
        )

    per_run = pd.DataFrame(all_rows)

    per_run = per_run.sort_values(
        by=["condition", "run"],
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)

    summary = build_summary(per_run)

    per_run_path = OUTPUT_DIR / "adc_controls_per_run.csv"
    summary_path = OUTPUT_DIR / "adc_controls_summary.csv"

    per_run.to_csv(
        per_run_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n[OK] ADC control analysis completed.")
    print(f"[SAVE] {per_run_path}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()