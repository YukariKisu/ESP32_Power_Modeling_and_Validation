from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Fixed paths
# ============================================================

RAW_ROOT = Path(
    "data/raw/v3_ppk/peripheral/adc/final_predictioned"
)

RAW_PATHS = {
    "idle_baseline": Path(
        "data/raw/v3_ppk/idle_baseline"
    ),
    "adc_init_only": (
        RAW_ROOT / "adc_init_only"
    ),
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
    "data/processed/v3_ppk/peripheral/adc/"
    "final_predictioned/baseline_decomposition"
)

# ============================================================
# Experiment timing relative to sync-pulse midpoint
# ============================================================

# Firmware timing:
# sync pulse:        -0.5 to +0.5 s
# recovery idle:     +0.5 to +5.5 s
# initial idle:      +5.5 to +15.5 s
# active/test:       +15.5 to +35.5 s
# final idle:        +35.5 to +45.5 s

INITIAL_IDLE_WINDOW_S = (5.5, 15.5)
ACTIVE_WINDOW_S = (15.5, 35.5)
FINAL_IDLE_WINDOW_S = (35.5, 45.5)

# Trim phase edges to avoid transition contamination.
PHASE_GUARD_S = 0.5

# Sync pulse settings.
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
    normalized_columns = {
        column: normalize_column_name(column)
        for column in columns
    }

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        for column, normalized in normalized_columns.items():
            if normalized == candidate_normalized:
                return column

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        for column, normalized in normalized_columns.items():
            if candidate_normalized in normalized:
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

    # PPK2 exports commonly use milliseconds with a step near 0.01–0.1.
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

    # Default expected PPK2 export unit.
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

    time_selected = time_s[mask]
    current_selected = current_mA[mask]

    if time_selected.size == 0:
        raise ValueError("No data in requested binning range.")

    indices = np.floor(
        (time_selected - start_s) / bin_s
    ).astype(np.int64)

    counts = np.bincount(indices)
    sums = np.bincount(
        indices,
        weights=current_selected,
    )

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

    search_end = min(
        SYNC_SEARCH_END_S,
        float(time_s[-1]),
    )

    centres, means = bin_mean(
        time_s,
        current_mA,
        SYNC_DETECTION_BIN_S,
        SYNC_SEARCH_START_S,
        search_end,
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
        np.ones(window_bins) / window_bins,
        mode="valid",
    )

    best_index = int(np.argmax(rolling_mean))

    pulse_start_s = (
        float(centres[best_index])
        - SYNC_DETECTION_BIN_S / 2.0
    )

    return pulse_start_s + SYNC_DURATION_S / 2.0


# ============================================================
# Phase cutting and current calculation
# ============================================================

def calculate_phase_mean(
    aligned_time_s: np.ndarray,
    current_mA: np.ndarray,
    window_s: tuple[float, float],
) -> float:
    start_s = window_s[0] + PHASE_GUARD_S
    end_s = window_s[1] - PHASE_GUARD_S

    mask = (
        (aligned_time_s >= start_s)
        & (aligned_time_s < end_s)
        & np.isfinite(current_mA)
    )

    if not np.any(mask):
        raise ValueError(
            f"No samples in phase window {window_s}"
        )

    return float(np.mean(current_mA[mask]))


def extract_run_number(path: Path) -> int | str:
    match = re.search(
        r"run[_-]?0*(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return path.stem


def analyze_run(
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

    initial_idle_mean_mA = calculate_phase_mean(
        aligned_time_s,
        current_mA,
        INITIAL_IDLE_WINDOW_S,
    )

    active_mean_mA = calculate_phase_mean(
        aligned_time_s,
        current_mA,
        ACTIVE_WINDOW_S,
    )

    final_idle_mean_mA = calculate_phase_mean(
        aligned_time_s,
        current_mA,
        FINAL_IDLE_WINDOW_S,
    )

    adc_idle_mean_mA = (
        initial_idle_mean_mA
        + final_idle_mean_mA
    ) / 2.0

    delta_I_mA = (
        active_mean_mA
        - adc_idle_mean_mA
    )

    return {
        "condition": condition,
        "run": extract_run_number(path),
        "file": path.name,
        "sync_midpoint_s": sync_midpoint_s,
        "initial_idle_mean_mA": initial_idle_mean_mA,
        "active_mean_mA": active_mean_mA,
        "final_idle_mean_mA": final_idle_mean_mA,
        "adc_idle_mean_mA": adc_idle_mean_mA,
        "delta_I_mA": delta_I_mA,
    }


# ============================================================
# Folder processing
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
            result = analyze_run(
                condition,
                csv_path,
            )
            rows.append(result)

            print(
                f"  {csv_path.name}: "
                f"sync={result['sync_midpoint_s']:.6f}s, "
                f"idle={result['adc_idle_mean_mA']:.6f}mA, "
                f"active={result['active_mean_mA']:.6f}mA, "
                f"delta={result['delta_I_mA']:.6f}mA"
            )

    return pd.DataFrame(rows)


# ============================================================
# Baseline decomposition
# ============================================================

def condition_mean(
    phase_means: pd.DataFrame,
    condition: str,
    column: str,
) -> float:
    values = phase_means.loc[
        phase_means["condition"] == condition,
        column,
    ]

    if values.empty:
        raise ValueError(
            f"No values found for condition: {condition}"
        )

    return float(values.mean())


def build_delta_current_table(
    phase_means: pd.DataFrame,
) -> pd.DataFrame:
    return phase_means[
        [
            "condition",
            "run",
            "file",
            "delta_I_mA",
        ]
    ].copy()


def build_decomposition_table(
    phase_means: pd.DataFrame,
) -> pd.DataFrame:
    # I1 and I2 are condition-level baselines.
    I1_esp_idle_baseline_mA = condition_mean(
        phase_means,
        "idle_baseline",
        "adc_idle_mean_mA",
    )

    I2_adc_init_only_mA = condition_mean(
        phase_means,
        "adc_init_only",
        "adc_idle_mean_mA",
    )

    workload_conditions = [
        condition
        for condition in RAW_PATHS
        if condition not in {
            "idle_baseline",
            "adc_init_only",
        }
    ]

    rows: list[dict[str, float | str]] = []

    for condition in workload_conditions:
        I3_adc_idle_mA = condition_mean(
            phase_means,
            condition,
            "adc_idle_mean_mA",
        )

        I4_adc_active_mA = condition_mean(
            phase_means,
            condition,
            "active_mean_mA",
        )

        adc_init_overhead_mA = (
            I2_adc_init_only_mA
            - I1_esp_idle_baseline_mA
        )

        adc_idle_overhead_mA = (
            I3_adc_idle_mA
            - I2_adc_init_only_mA
        )

        adc_active_overhead_mA = (
            I4_adc_active_mA
            - I3_adc_idle_mA
        )

        total_overhead_mA = (
            I4_adc_active_mA
            - I1_esp_idle_baseline_mA
        )

        reconstructed_total_mA = (
            adc_init_overhead_mA
            + adc_idle_overhead_mA
            + adc_active_overhead_mA
        )

        rows.append(
            {
                "condition": condition,
                "I1_esp_idle_baseline_mA":
                    I1_esp_idle_baseline_mA,
                "I2_adc_init_only_mA":
                    I2_adc_init_only_mA,
                "I3_adc_idle_mA":
                    I3_adc_idle_mA,
                "I4_adc_active_mA":
                    I4_adc_active_mA,
                "adc_init_overhead_mA":
                    adc_init_overhead_mA,
                "adc_idle_overhead_mA":
                    adc_idle_overhead_mA,
                "adc_active_overhead_mA":
                    adc_active_overhead_mA,
                "total_overhead_mA":
                    total_overhead_mA,
                "reconstructed_total_mA":
                    reconstructed_total_mA,
                "closure_error_mA":
                    reconstructed_total_mA
                    - total_overhead_mA,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase_means = process_all_runs()

    phase_means = phase_means.sort_values(
        by=["condition", "run"],
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)

    delta_current = build_delta_current_table(
        phase_means
    )

    decomposition = build_decomposition_table(
        phase_means
    )

    phase_means_path = (
        OUTPUT_DIR / "adc_phase_means.csv"
    )
    delta_current_path = (
        OUTPUT_DIR / "adc_delta_current.csv"
    )
    decomposition_path = (
        OUTPUT_DIR / "adc_baseline_decomposition.csv"
    )

    phase_means.to_csv(
        phase_means_path,
        index=False,
    )
    delta_current.to_csv(
        delta_current_path,
        index=False,
    )
    decomposition.to_csv(
        decomposition_path,
        index=False,
    )

    print("\n[OK] ADC baseline decomposition completed.")
    print(f"[SAVE] {phase_means_path}")
    print(f"[SAVE] {delta_current_path}")
    print(f"[SAVE] {decomposition_path}")


if __name__ == "__main__":
    main()