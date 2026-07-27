from __future__ import annotations

import csv
import re
from pathlib import Path

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
    "data/processed/v3_ppk/peripheral/adc/baseline_stability"
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

# Exclude phase edges from mean and SD calculations.
PHASE_GUARD_S = 0.5

SYNC_DURATION_S = 1.0
SYNC_SEARCH_START_S = 1.0
SYNC_SEARCH_END_S = 10.0
SYNC_DETECTION_BIN_S = 0.001


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
# Sync pulse detection
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

def select_phase_current(
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

    selected = current_mA[mask]

    if selected.size < 2:
        raise ValueError(
            f"Not enough samples in phase window {window_s}"
        )

    return selected


def phase_statistics(values: np.ndarray) -> tuple[float, float]:
    mean_mA = float(np.mean(values))

    # Sample SD, ddof=1.
    std_mA = float(np.std(values, ddof=1))

    return mean_mA, std_mA


def extract_run_number(path: Path) -> int | str:
    match = re.search(
        r"run[_-]?0*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return path.stem


def analyse_run(
    condition: str,
    path: Path,
) -> dict[str, object]:
    dataframe = load_ppk_csv(path)

    sync_midpoint_s = detect_sync_midpoint(dataframe)

    aligned_time_s = (
        dataframe["time_s"].to_numpy()
        - sync_midpoint_s
    )
    current_mA = dataframe["current_mA"].to_numpy()

    initial_values = select_phase_current(
        aligned_time_s,
        current_mA,
        INITIAL_IDLE_WINDOW_S,
    )
    active_values = select_phase_current(
        aligned_time_s,
        current_mA,
        ACTIVE_WINDOW_S,
    )
    final_values = select_phase_current(
        aligned_time_s,
        current_mA,
        FINAL_IDLE_WINDOW_S,
    )

    initial_mean_mA, initial_std_mA = phase_statistics(
        initial_values
    )
    active_mean_mA, active_std_mA = phase_statistics(
        active_values
    )
    final_mean_mA, final_std_mA = phase_statistics(
        final_values
    )

    adc_idle_mean_mA = (
        initial_mean_mA + final_mean_mA
    ) / 2.0

    delta_I_mA = active_mean_mA - adc_idle_mean_mA

    drift_mA = final_mean_mA - initial_mean_mA

    return {
        "condition": condition,
        "run": extract_run_number(path),
        "file": path.name,
        "sync_midpoint_s": sync_midpoint_s,

        "initial_idle_mean_mA": initial_mean_mA,
        "initial_idle_std_mA": initial_std_mA,
        "initial_idle_n_samples": int(initial_values.size),

        "active_mean_mA": active_mean_mA,
        "active_std_mA": active_std_mA,
        "active_n_samples": int(active_values.size),

        "final_idle_mean_mA": final_mean_mA,
        "final_idle_std_mA": final_std_mA,
        "final_idle_n_samples": int(final_values.size),

        "adc_idle_mean_mA": adc_idle_mean_mA,
        "delta_I_mA": delta_I_mA,
        "initial_to_final_drift_mA": drift_mA,
    }


# ============================================================
# Condition-level summary
# ============================================================

def sample_sd(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return float("nan")

    return float(values.std(ddof=1))


def build_condition_summary(
    per_run: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition, group in per_run.groupby(
        "condition",
        sort=False,
    ):
        rows.append(
            {
                "condition": condition,
                "n_runs": int(len(group)),

                # Mean phase fluctuation across runs.
                "initial_idle_std_mean_mA":
                    float(group["initial_idle_std_mA"].mean()),
                "initial_idle_std_between_runs_mA":
                    sample_sd(group["initial_idle_std_mA"]),

                "active_std_mean_mA":
                    float(group["active_std_mA"].mean()),
                "active_std_between_runs_mA":
                    sample_sd(group["active_std_mA"]),

                "final_idle_std_mean_mA":
                    float(group["final_idle_std_mA"].mean()),
                "final_idle_std_between_runs_mA":
                    sample_sd(group["final_idle_std_mA"]),

                # Requested 10-run ΔI mean ± SD.
                "delta_I_mean_mA":
                    float(group["delta_I_mA"].mean()),
                "delta_I_sd_mA":
                    sample_sd(group["delta_I_mA"]),

                # Requested initial-to-final drift mean ± SD.
                "initial_to_final_drift_mean_mA":
                    float(
                        group["initial_to_final_drift_mA"].mean()
                    ),
                "initial_to_final_drift_sd_mA":
                    sample_sd(
                        group["initial_to_final_drift_mA"]
                    ),

                # Useful phase-level 10-run means.
                "initial_idle_mean_across_runs_mA":
                    float(group["initial_idle_mean_mA"].mean()),
                "active_mean_across_runs_mA":
                    float(group["active_mean_mA"].mean()),
                "final_idle_mean_across_runs_mA":
                    float(group["final_idle_mean_mA"].mean()),
            }
        )

    return pd.DataFrame(rows)


def add_readable_columns(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    readable = summary.copy()

    readable["delta_I_mean_plus_minus_sd"] = readable.apply(
        lambda row: (
            f"{row['delta_I_mean_mA']:.6f} ± "
            f"{row['delta_I_sd_mA']:.6f} mA"
        ),
        axis=1,
    )

    readable["drift_mean_plus_minus_sd"] = readable.apply(
        lambda row: (
            f"{row['initial_to_final_drift_mean_mA']:.6f} ± "
            f"{row['initial_to_final_drift_sd_mA']:.6f} mA"
        ),
        axis=1,
    )

    columns = [
        "condition",
        "n_runs",
        "initial_idle_std_mean_mA",
        "active_std_mean_mA",
        "final_idle_std_mean_mA",
        "delta_I_mean_mA",
        "delta_I_sd_mA",
        "delta_I_mean_plus_minus_sd",
        "initial_to_final_drift_mean_mA",
        "initial_to_final_drift_sd_mA",
        "drift_mean_plus_minus_sd",
        "initial_idle_mean_across_runs_mA",
        "active_mean_across_runs_mA",
        "final_idle_mean_across_runs_mA",
    ]

    return readable[columns]


# ============================================================
# Main
# ============================================================

def find_csv_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*.csv")
        if path.is_file()
    )


def process_all_runs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

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

        for csv_path in csv_files:
            result = analyse_run(
                condition,
                csv_path,
            )
            rows.append(result)

            print(
                f"  {csv_path.name}: "
                f"initial_std={result['initial_idle_std_mA']:.6f} mA, "
                f"active_std={result['active_std_mA']:.6f} mA, "
                f"final_std={result['final_idle_std_mA']:.6f} mA, "
                f"delta={result['delta_I_mA']:.6f} mA, "
                f"drift={result['initial_to_final_drift_mA']:.6f} mA"
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_run = process_all_runs()

    per_run = per_run.sort_values(
        by=["condition", "run"],
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)

    summary = build_condition_summary(per_run)
    readable_summary = add_readable_columns(summary)

    per_run_path = (
        OUTPUT_DIR / "adc_baseline_stability_per_run.csv"
    )
    summary_path = (
        OUTPUT_DIR / "adc_baseline_stability_summary.csv"
    )

    per_run.to_csv(
        per_run_path,
        index=False,
    )
    readable_summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n[OK] ADC baseline stability analysis completed.")
    print(f"[SAVE] {per_run_path}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()