from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------- User settings ----------------

INPUT_DIRS = {
    "ble_adv_100ms": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_100ms"),
    "ble_adv_500ms": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_500ms"),
    "ble_adv_1000ms": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_1000ms"),
}

OUTPUT_DIR = Path("results/v3_ppk/peripheral/ble_adv_only/mae_input_definitions")

# CPU-trained ODE parameters. Replace these with your final fitted values.
ODE_GAIN_MA = 21.310
ODE_TAU_RISE_S = 0.050
ODE_TAU_FALL_S = 0.050

# BLE input definition.
ACTIVE_DURATION_S = 20.0
BLE_EVENT_DURATION_S = 0.001128

OCCUPANCY_U = {
    "ble_adv_100ms": 0.01128,
    "ble_adv_500ms": 0.002256,
    "ble_adv_1000ms": 0.001128,
}

PULSE_INTERVAL_S = {
    "ble_adv_100ms": 0.100,
    "ble_adv_500ms": 0.500,
    "ble_adv_1000ms": 1.000,
}

# Analysis windows after sync-pulse midpoint.
INITIAL_START_S = 5.5
INITIAL_END_S = 15.5
ACTIVE_START_S = 15.5
ACTIVE_END_S = 35.5
WINDOW_TRIM_S = 0.2

# Resampling keeps MAE computation fast while preserving 1.128 ms BLE pulses.
UNIFORM_DT_S = 0.0001

# Sync pulse search settings.
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
class RunResult:
    condition: str
    input_definition: str
    run_id: str
    source_file: str
    sync_mid_s: float
    initial_mean_ma: float
    measured_active_delta_mean_ma: float
    predicted_active_delta_mean_ma: float
    mae_ma: float
    me_ma: float
    rmse_ma: float


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


def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if "ms" in name:
        return values / 1000.0
    if "us" in name or "µs" in name:
        return values / 1000000.0
    return values


def convert_current_to_ma(values: np.ndarray, column_name: str) -> np.ndarray:
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
    current_ma = convert_current_to_ma(current_raw, current_col)

    return pd.DataFrame({"time_s": time_s, "current_ma": current_ma})


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(values, kernel, mode="same")


def detect_sync_midpoint(df: pd.DataFrame) -> float:
    time_s = df["time_s"].to_numpy()
    current_ma = df["current_ma"].to_numpy()

    dt = float(np.median(np.diff(time_s)))
    smooth_samples = max(1, round(SYNC_SMOOTH_WINDOW_S / dt))
    smooth_current = rolling_mean(current_ma, smooth_samples)

    search_start = max(0.0, SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S)
    search_end = SYNC_EXPECTED_START_S + SYNC_DURATION_S + SYNC_SEARCH_MARGIN_S
    search_mask = (time_s >= search_start) & (time_s <= search_end)
    if not np.any(search_mask):
        raise ValueError("Sync-pulse search window does not contain samples")

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline_mask = (
        (time_s >= max(0.0, SYNC_EXPECTED_START_S - 2.5))
        & (time_s <= SYNC_EXPECTED_START_S - 0.3)
    )
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

    expected_mid = SYNC_EXPECTED_START_S + SYNC_DURATION_S / 2.0

    def score(segment: tuple[int, int]) -> tuple[float, float]:
        start_idx, end_idx = segment
        seg_start = search_time[start_idx]
        seg_end = search_time[end_idx]
        seg_mid = (seg_start + seg_end) / 2.0
        duration_error = abs((seg_end - seg_start) - SYNC_DURATION_S)
        timing_error = abs(seg_mid - expected_mid)
        return (duration_error, timing_error)

    best_start, best_end = min(segments, key=score)
    return float((search_time[best_start] + search_time[best_end]) / 2.0)


def infer_run_id(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(run[_-]?\d+|\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return stem


def make_occupancy_input(condition: str, t: np.ndarray) -> np.ndarray:
    return np.full_like(t, OCCUPANCY_U[condition], dtype=float)


def make_pulse_input(condition: str, t: np.ndarray) -> np.ndarray:
    interval_s = PULSE_INTERVAL_S[condition]
    phase = np.mod(t, interval_s)
    return (phase < BLE_EVENT_DURATION_S).astype(float)


def simulate_first_order_ode(u: np.ndarray, dt_s: float) -> np.ndarray:
    y = np.zeros_like(u, dtype=float)
    for i in range(1, len(u)):
        tau = ODE_TAU_RISE_S if (ODE_GAIN_MA * u[i]) >= y[i - 1] else ODE_TAU_FALL_S
        alpha = 1.0 - math.exp(-dt_s / tau)
        y[i] = y[i - 1] + alpha * ((ODE_GAIN_MA * u[i]) - y[i - 1])
    return y


def active_uniform_measurement(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    initial_mask = (
        (df["aligned_time_s"] >= INITIAL_START_S + WINDOW_TRIM_S)
        & (df["aligned_time_s"] <= INITIAL_END_S - WINDOW_TRIM_S)
    )
    initial_mean_ma = float(df.loc[initial_mask, "current_ma"].mean())

    active_start = ACTIVE_START_S
    active_end = ACTIVE_END_S
    active_mask = (
        (df["aligned_time_s"] >= active_start)
        & (df["aligned_time_s"] <= active_end)
    )

    active_time = df.loc[active_mask, "aligned_time_s"].to_numpy() - active_start
    active_current = df.loc[active_mask, "current_ma"].to_numpy()
    measured_delta = active_current - initial_mean_ma

    if len(active_time) < 10:
        raise ValueError("Active window does not contain enough samples")

    uniform_t = np.arange(0.0, ACTIVE_DURATION_S, UNIFORM_DT_S)
    uniform_measured_delta = np.interp(uniform_t, active_time, measured_delta)
    return uniform_t, uniform_measured_delta, initial_mean_ma


def analyze_one_file(condition: str, path: Path) -> list[RunResult]:
    df = read_ppk_csv(path)
    sync_mid_s = detect_sync_midpoint(df)
    df["aligned_time_s"] = df["time_s"] - sync_mid_s

    t, measured_delta, initial_mean_ma = active_uniform_measurement(df)
    run_id = infer_run_id(path)

    results: list[RunResult] = []
    for input_definition, input_builder in (
        ("occupancy", make_occupancy_input),
        ("pulse", make_pulse_input),
    ):
        u = input_builder(condition, t)
        prediction_delta = simulate_first_order_ode(u, UNIFORM_DT_S)
        error = prediction_delta - measured_delta

        results.append(
            RunResult(
                condition=condition,
                input_definition=input_definition,
                run_id=run_id,
                source_file=str(path),
                sync_mid_s=sync_mid_s,
                initial_mean_ma=initial_mean_ma,
                measured_active_delta_mean_ma=float(np.mean(measured_delta)),
                predicted_active_delta_mean_ma=float(np.mean(prediction_delta)),
                mae_ma=float(np.mean(np.abs(error))),
                me_ma=float(np.mean(error)),
                rmse_ma=float(np.sqrt(np.mean(error * error))),
            )
        )

    return results


def collect_csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def results_to_frame(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def summarize(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = per_run_df.groupby(["condition", "input_definition"], sort=True)

    for (condition, input_definition), group in grouped:
        rows.append(
            {
                "condition": condition,
                "input_definition": input_definition,
                "n_runs": int(len(group)),
                "mae_mean_ma": float(group["mae_ma"].mean()),
                "mae_std_ma": float(group["mae_ma"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mae_min_ma": float(group["mae_ma"].min()),
                "mae_max_ma": float(group["mae_ma"].max()),
                "me_mean_ma": float(group["me_ma"].mean()),
                "rmse_mean_ma": float(group["rmse_ma"].mean()),
                "measured_delta_mean_ma": float(group["measured_active_delta_mean_ma"].mean()),
                "predicted_delta_mean_ma": float(group["predicted_active_delta_mean_ma"].mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    all_results: list[RunResult] = []
    errors: list[dict[str, str]] = []

    for condition, input_dir in INPUT_DIRS.items():
        for path in collect_csv_files(input_dir):
            try:
                print(f"Processing {condition}: {path}", flush=True)
                all_results.extend(analyze_one_file(condition, path))
            except Exception as exc:
                errors.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(exc),
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not all_results:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / "errors.csv", index=False)
        raise SystemExit("No runs could be analyzed. See errors.csv.")

    per_run_df = results_to_frame(all_results)
    summary_df = summarize(per_run_df)

    per_run_df.to_csv(OUTPUT_DIR / "per_run_mae.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_mae.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / "errors.csv", index=False)

    print(f"Analyzed validation rows: {len(per_run_df)}")
    print(f"Skipped files: {len(errors)}")
    print(f"Wrote: {OUTPUT_DIR / 'per_run_mae.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'summary_mae.csv'}")


if __name__ == "__main__":
    main()