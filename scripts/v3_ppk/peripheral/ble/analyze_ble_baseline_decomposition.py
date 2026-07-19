from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

INPUT_DIRS = [
    Path("data/raw/v3_ppk/idle_baseline"),
    Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_init_only"),
    Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_100ms"),
    Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_500ms"),
    Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_1000ms"),
]

OUTPUT_DIR = Path("results/v3_ppk/peripheral/ble_adv_only/baseline_decomposition")

CURRENT_UNIT = "auto"
INIT_CONDITION = "ble_init_only"
ESP_BASELINE_CONDITION = "idle_baseline"
TRIM_S = 0.2
SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_MARGIN_S = 1.5
SYNC_SMOOTH_WINDOW_S = 0.02


@dataclass(frozen=True)
class PhaseWindows:
    initial_start_s: float
    initial_end_s: float
    active_start_s: float
    active_end_s: float
    final_start_s: float
    final_end_s: float


@dataclass(frozen=True)
class RunStats:
    condition: str
    run_id: str
    source_file: str
    sync_mid_s: float
    initial_mean_ma: float
    active_mean_ma: float
    final_mean_ma: float
    delta_active_initial_ma: float
    delta_final_initial_ma: float
    initial_n: int
    active_n: int
    final_n: int


TIME_COLUMN_CANDIDATES = (
    "time",
    "timestamp",
    "timestamp_s",
    "time_s",
    "time (s)",
    "time[s]",
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


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for col in columns:
        col_norm = normalize_name(col)
        if kind == "time" and "time" in col_norm:
            return col
        if kind == "current" and "current" in col_norm:
            return col

    raise ValueError(f"Could not find {kind} column in columns: {list(columns)}")


def read_ppk_csv(path: Path, current_unit: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    time_col = find_column(df.columns, TIME_COLUMN_CANDIDATES, "time")
    current_col = find_column(df.columns, CURRENT_COLUMN_CANDIDATES, "current")

    time_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    current_raw = pd.to_numeric(df[current_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(time_raw) & np.isfinite(current_raw)
    time_raw = time_raw[valid]
    current_raw = current_raw[valid]

    if len(time_raw) < 10:
        raise ValueError(f"Not enough valid samples in {path}")

    time_s = convert_time_to_s(time_raw, time_col)
    time_s = time_s - time_s[0]
    current_ma = convert_current_to_ma(current_raw, current_col, current_unit)

    return pd.DataFrame({"time_s": time_s, "current_ma": current_ma})


def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)

    if "ms" in name:
        return values / 1000.0
    if "us" in name or "µs" in name:
        return values / 1000000.0

    return values


def convert_current_to_ma(values: np.ndarray, column_name: str, current_unit: str) -> np.ndarray:
    unit = current_unit.lower()

    if unit == "auto":
        name = normalize_name(column_name)
        if "ua" in name or "micro" in name or "µa" in name:
            unit = "ua"
        elif re.search(r"\bma\b", name) or "(ma" in name or "[ma" in name:
            unit = "ma"
        elif re.search(r"\ba\b", name) or "(a" in name or "[a" in name:
            unit = "a"
        else:
            median_abs = float(np.nanmedian(np.abs(values)))
            if median_abs < 1.0:
                unit = "a"
            elif median_abs > 1000.0:
                unit = "ua"
            else:
                unit = "ma"

    if unit == "a":
        return values * 1000.0
    if unit == "ua":
        return values / 1000.0
    if unit == "ma":
        return values

    raise ValueError(f"Unsupported current unit: {current_unit}")


def infer_condition(path: Path) -> str:
    text = str(path).lower()

    if "init" in text and "only" in text:
        return "ble_init_only"
    if "init_only" in text:
        return "ble_init_only"
    if "1000" in text:
        return "ble_adv_1000ms"
    if "500" in text:
        return "ble_adv_500ms"
    if "100" in text:
        return "ble_adv_100ms"

    return path.parent.name


def infer_run_id(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(run[_-]?\d+|\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return stem


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(values, kernel, mode="same")


def detect_sync_midpoint(
    df: pd.DataFrame,
    expected_start_s: float,
    expected_duration_s: float,
    search_margin_s: float,
    smooth_window_s: float,
) -> float:
    time_s = df["time_s"].to_numpy()
    current_ma = df["current_ma"].to_numpy()

    dt = float(np.median(np.diff(time_s)))
    smooth_samples = max(1, round(smooth_window_s / dt))
    smooth_current = rolling_mean(current_ma, smooth_samples)

    search_start = max(0.0, expected_start_s - search_margin_s)
    search_end = expected_start_s + expected_duration_s + search_margin_s
    search_mask = (time_s >= search_start) & (time_s <= search_end)
    if not np.any(search_mask):
        raise ValueError("Sync-pulse search window does not contain samples")

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline_mask = (time_s >= max(0.0, expected_start_s - 2.5)) & (time_s <= expected_start_s - 0.3)
    if np.any(baseline_mask):
        baseline = float(np.median(smooth_current[baseline_mask]))
    else:
        baseline = float(np.percentile(search_current, 20))

    peak = float(np.max(search_current))
    threshold = baseline + 0.45 * (peak - baseline)
    high = search_current >= threshold

    segments: list[tuple[int, int]] = []
    start = None
    for idx, is_high in enumerate(high):
        if is_high and start is None:
            start = idx
        elif not is_high and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(high) - 1))

    if not segments:
        peak_idx = int(np.argmax(search_current))
        return float(search_time[peak_idx])

    expected_mid = expected_start_s + expected_duration_s / 2.0

    def segment_score(segment: tuple[int, int]) -> tuple[float, float]:
        start_idx, end_idx = segment
        seg_start = search_time[start_idx]
        seg_end = search_time[end_idx]
        seg_mid = (seg_start + seg_end) / 2.0
        seg_duration = seg_end - seg_start
        duration_error = abs(seg_duration - expected_duration_s)
        timing_error = abs(seg_mid - expected_mid)
        return (duration_error, timing_error)

    best_start, best_end = min(segments, key=segment_score)
    return float((search_time[best_start] + search_time[best_end]) / 2.0)


def default_phase_windows(trim_s: float) -> PhaseWindows:
    initial_start = 0.5 + 5.0
    initial_end = initial_start + 10.0
    active_start = initial_end
    active_end = active_start + 20.0
    final_start = active_end
    final_end = final_start + 10.0

    return PhaseWindows(
        initial_start_s=initial_start + trim_s,
        initial_end_s=initial_end - trim_s,
        active_start_s=active_start + trim_s,
        active_end_s=active_end - trim_s,
        final_start_s=final_start + trim_s,
        final_end_s=final_end - trim_s,
    )


def window_mean(df: pd.DataFrame, start_s: float, end_s: float) -> tuple[float, int]:
    mask = (df["aligned_time_s"] >= start_s) & (df["aligned_time_s"] <= end_s)
    values = df.loc[mask, "current_ma"].to_numpy()
    if len(values) == 0:
        return (math.nan, 0)
    return (float(np.mean(values)), int(len(values)))


def analyze_run(path: Path, args: argparse.Namespace, windows: PhaseWindows) -> RunStats:
    df = read_ppk_csv(path, args.current_unit)
    sync_mid_s = detect_sync_midpoint(
        df,
        expected_start_s=args.sync_expected_start_s,
        expected_duration_s=args.sync_duration_s,
        search_margin_s=args.sync_search_margin_s,
        smooth_window_s=args.sync_smooth_window_s,
    )

    df["aligned_time_s"] = df["time_s"] - sync_mid_s

    initial_mean, initial_n = window_mean(df, windows.initial_start_s, windows.initial_end_s)
    active_mean, active_n = window_mean(df, windows.active_start_s, windows.active_end_s)
    final_mean, final_n = window_mean(df, windows.final_start_s, windows.final_end_s)

    return RunStats(
        condition=infer_condition(path),
        run_id=infer_run_id(path),
        source_file=str(path),
        sync_mid_s=sync_mid_s,
        initial_mean_ma=initial_mean,
        active_mean_ma=active_mean,
        final_mean_ma=final_mean,
        delta_active_initial_ma=active_mean - initial_mean,
        delta_final_initial_ma=final_mean - initial_mean,
        initial_n=initial_n,
        active_n=active_n,
        final_n=final_n,
    )

def collect_csv_files(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix.lower() == ".csv":
            files.append(input_path)
        elif input_path.is_dir():
            files.extend(sorted(input_path.rglob("*.csv")))
        else:
            raise FileNotFoundError(f"Input path not found or not CSV: {input_path}")
    return [path for path in files if path.is_file()]


def stats_to_dict(stats: RunStats) -> dict[str, object]:
    return {
        "condition": stats.condition,
        "run_id": stats.run_id,
        "source_file": stats.source_file,
        "sync_mid_s": stats.sync_mid_s,
        "initial_mean_ma": stats.initial_mean_ma,
        "active_mean_ma": stats.active_mean_ma,
        "final_mean_ma": stats.final_mean_ma,
        "delta_active_initial_ma": stats.delta_active_initial_ma,
        "delta_final_initial_ma": stats.delta_final_initial_ma,
        "initial_n": stats.initial_n,
        "active_n": stats.active_n,
        "final_n": stats.final_n,
    }


def summarize_conditions(per_run_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "initial_mean_ma",
        "active_mean_ma",
        "final_mean_ma",
        "delta_active_initial_ma",
        "delta_final_initial_ma",
    ]

    grouped = per_run_df.groupby("condition", sort=True)
    rows = []
    for condition, group in grouped:
        row: dict[str, object] = {"condition": condition, "n_runs": int(len(group))}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_baseline_decomposition(
    summary_df: pd.DataFrame,
    esp_baseline_ma: float,
    init_condition: str,
) -> pd.DataFrame:
    summary = summary_df.set_index("condition")
    if init_condition not in summary.index:
        raise ValueError(
            f"Init condition '{init_condition}' not found. "
            f"Available conditions: {list(summary.index)}"
        )

    i1 = esp_baseline_ma
    i2 = float(summary.loc[init_condition, "active_mean_ma_mean"])

    rows = [
        {
            "condition": init_condition,
            "I1_esp_baseline_ma": i1,
            "I2_ble_init_baseline_ma": i2,
            "I3_ble_adv_active_ma": math.nan,
            "ble_init_overhead_ma": i2 - i1,
            "ble_adv_overhead_ma": math.nan,
            "total_overhead_ma": i2 - i1,
        }
    ]

    for condition in sorted(summary.index):
        if condition == init_condition:
            continue
        if not condition.startswith("ble_adv"):
            continue

        i3 = float(summary.loc[condition, "active_mean_ma_mean"])
        rows.append(
            {
                "condition": condition,
                "I1_esp_baseline_ma": i1,
                "I2_ble_init_baseline_ma": i2,
                "I3_ble_adv_active_ma": i3,
                "ble_init_overhead_ma": i2 - i1,
                "ble_adv_overhead_ma": i3 - i2,
                "total_overhead_ma": i3 - i1,
            }
        )

    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, object]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return

    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze BLE baseline decomposition from PPK2 CSV files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing PPK2 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--esp-baseline-ma",
        type=float,
        required=True,
        help="Existing ESP baseline current in mA. This is I1.",
    )
    parser.add_argument(
        "--current-unit",
        choices=("auto", "a", "ma", "ua"),
        default="auto",
        help="Current unit in CSV. Default: auto.",
    )
    parser.add_argument(
        "--init-condition",
        default="ble_init_only",
        help="Condition name used as I2 BLE init baseline.",
    )
    parser.add_argument(
        "--trim-s",
        type=float,
        default=0.2,
        help="Seconds trimmed from both ends of each phase window.",
    )
    parser.add_argument(
        "--sync-expected-start-s",
        type=float,
        default=3.0,
        help="Expected sync pulse start time from recording start.",
    )
    parser.add_argument(
        "--sync-duration-s",
        type=float,
        default=1.0,
        help="Expected sync pulse duration.",
    )
    parser.add_argument(
        "--sync-search-margin-s",
        type=float,
        default=1.5,
        help="Search margin around expected sync pulse.",
    )
    parser.add_argument(
        "--sync-smooth-window-s",
        type=float,
        default=0.02,
        help="Smoothing window for sync pulse detection.",
    )
    return parser.parse_args()


def main() -> None:
    files = collect_csv_files(INPUT_DIRS)
    if not files:
        raise SystemExit(f"No CSV files found under: {INPUT_DIRS}")

    class Config:
        current_unit = CURRENT_UNIT
        sync_expected_start_s = SYNC_EXPECTED_START_S
        sync_duration_s = SYNC_DURATION_S
        sync_search_margin_s = SYNC_SEARCH_MARGIN_S
        sync_smooth_window_s = SYNC_SMOOTH_WINDOW_S

    config = Config()
    windows = default_phase_windows(TRIM_S)

    run_stats: list[RunStats] = []
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            run_stats.append(analyze_run(path, config, windows))
        except Exception as exc:
            errors.append({"source_file": str(path), "error": str(exc)})

    if not run_stats:
        write_csv(OUTPUT_DIR / "errors.csv", errors)
        raise SystemExit("No runs could be analyzed. See errors.csv.")

    per_run_rows = [stats_to_dict(stats) for stats in run_stats]
    per_run_df = pd.DataFrame(per_run_rows)
    summary_df = summarize_conditions(per_run_df)

    summary = summary_df.set_index("condition")

    if ESP_BASELINE_CONDITION not in summary.index:
        raise ValueError(
            f"ESP baseline condition '{ESP_BASELINE_CONDITION}' not found. "
            f"Available conditions: {list(summary.index)}"
        )

    if INIT_CONDITION not in summary.index:
        raise ValueError(
            f"Init condition '{INIT_CONDITION}' not found. "
            f"Available conditions: {list(summary.index)}"
        )

    i1 = float(summary.loc[ESP_BASELINE_CONDITION, "active_mean_ma_mean"])
    i2 = float(summary.loc[INIT_CONDITION, "active_mean_ma_mean"])

    rows = [
        {
            "condition": INIT_CONDITION,
            "I1_esp_baseline_ma": i1,
            "I2_ble_init_baseline_ma": i2,
            "I3_ble_adv_active_ma": math.nan,
            "ble_init_overhead_ma": i2 - i1,
            "ble_adv_overhead_ma": math.nan,
            "total_overhead_ma": i2 - i1,
        }
    ]

    for condition in ["ble_adv_100ms", "ble_adv_500ms", "ble_adv_1000ms"]:
        if condition not in summary.index:
            continue

        i3 = float(summary.loc[condition, "active_mean_ma_mean"])
        rows.append(
            {
                "condition": condition,
                "I1_esp_baseline_ma": i1,
                "I2_ble_init_baseline_ma": i2,
                "I3_ble_adv_active_ma": i3,
                "ble_init_overhead_ma": i2 - i1,
                "ble_adv_overhead_ma": i3 - i2,
                "total_overhead_ma": i3 - i1,
            }
        )

    decomposition_df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "per_run_stats.csv", per_run_df)
    write_csv(OUTPUT_DIR / "condition_summary.csv", summary_df)
    write_csv(OUTPUT_DIR / "baseline_decomposition.csv", decomposition_df)

    if errors:
        write_csv(OUTPUT_DIR / "errors.csv", errors)

    print(f"Analyzed runs: {len(run_stats)}")
    print(f"Skipped files: {len(errors)}")
    print(f"Wrote: {OUTPUT_DIR / 'per_run_stats.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'condition_summary.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'baseline_decomposition.csv'}")


if __name__ == "__main__":
    main()