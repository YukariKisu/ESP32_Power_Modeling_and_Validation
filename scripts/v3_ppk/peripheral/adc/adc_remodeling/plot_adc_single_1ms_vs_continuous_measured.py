from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_ROOT = Path("data/raw/v3_ppk/peripheral/adc/final_predictioned")
SINGLE_1MS_DIR = RAW_ROOT / "adc_periodic_single" / "adc_single_1ms"
CONTINUOUS_DIR = RAW_ROOT / "adc_remodel_train"
OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/adc/final_predictioned/"
    "single_1ms_vs_continuous_measured"
)

SYNC_TO_ACTIVE_START_S = 16.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_START_S = 1.0
SYNC_SEARCH_END_S = 6.0
SYNC_SMOOTH_WINDOW_S = 0.020

BASELINE_START_S = 6.2
BASELINE_END_S = 15.8

START_WINDOW_S = (-0.020, 0.080)
ZOOM_WINDOW_S = (5.000, 5.010)
UNIFORM_DT_S = 0.00001

TIME_COLUMN_CANDIDATES = (
    "time", "timestamp", "timestamp_s", "time_s",
    "time (s)", "time[s]", "timestamp(ms)",
)
CURRENT_COLUMN_CANDIDATES = (
    "current", "current_ma", "current (ma)", "current[ma]",
    "current_ua", "current (ua)", "current[ua]",
    "current_a", "current (a)", "current[a]",
)

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())

def find_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    normalized = {normalize_name(c): c for c in columns}
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
    raise ValueError(f"Could not find {kind} column in {list(columns)}")

def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if "ms" in name:
        return values / 1000.0
    if "us" in name or "µs" in name:
        return values / 1_000_000.0
    return values

def convert_current_to_ma(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if "ua" in name or "µa" in name:
        return values / 1000.0
    if "ma" in name:
        return values
    if re.search(r"\ba\b", name) or "(a" in name or "[a" in name:
        return values * 1000.0
    med = float(np.nanmedian(np.abs(values)))
    if med < 1.0:
        return values * 1000.0
    if med > 1000.0:
        return values / 1000.0
    return values

def read_ppk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    time_col = find_column(df.columns, TIME_COLUMN_CANDIDATES, "time")
    current_col = find_column(df.columns, CURRENT_COLUMN_CANDIDATES, "current")

    t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    i_raw = pd.to_numeric(df[current_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(t_raw) & np.isfinite(i_raw)

    t = convert_time_to_s(t_raw[valid], time_col)
    t = t - t[0]
    i = convert_current_to_ma(i_raw[valid], current_col)

    return pd.DataFrame({"time_s": t, "current_ma": i})

def collect_csv_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder}")
    files = list(folder.glob("*.csv"))
    def sort_key(p: Path):
        m = re.search(r"run[_-]?(\d+)", p.stem, flags=re.IGNORECASE)
        return (int(m.group(1)) if m else 999999, p.name)
    files = sorted(files, key=sort_key)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    return files

def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values
    kernel = np.ones(samples) / samples
    return np.convolve(values, kernel, mode="same")

def detect_sync_start(df: pd.DataFrame) -> float:
    t = df["time_s"].to_numpy(float)
    i = df["current_ma"].to_numpy(float)

    dt = float(np.median(np.diff(t)))
    smooth_n = max(1, round(SYNC_SMOOTH_WINDOW_S / dt))
    smooth = rolling_mean(i, smooth_n)

    mask = (t >= SYNC_SEARCH_START_S) & (t <= SYNC_SEARCH_END_S)
    st = t[mask]
    sy = smooth[mask]

    baseline = float(np.percentile(sy, 20))
    peak = float(np.percentile(sy, 99))
    threshold = baseline + 0.45 * (peak - baseline)

    high = sy >= threshold
    segments = []
    start_idx = None
    for idx, is_high in enumerate(high):
        if is_high and start_idx is None:
            start_idx = idx
        elif not is_high and start_idx is not None:
            segments.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(high) - 1))
    if not segments:
        raise ValueError("No sync-like segment detected")

    def score(seg):
        i0, i1 = seg
        dur = float(st[i1] - st[i0])
        strength = float(np.mean(sy[i0:i1+1]) - baseline)
        return abs(dur - SYNC_DURATION_S), -strength

    i0, _ = min(segments, key=score)
    return float(st[i0])

def prepare_run(path: Path) -> dict[str, object]:
    df = read_ppk_csv(path)
    sync_start = detect_sync_start(df)

    aligned = df["time_s"].to_numpy(float) - sync_start
    current = df["current_ma"].to_numpy(float)

    baseline_mask = (
        (aligned >= BASELINE_START_S)
        & (aligned <= BASELINE_END_S)
    )
    baseline = float(np.mean(current[baseline_mask]))

    return {
        "path": path,
        "time_active_s": aligned - SYNC_TO_ACTIVE_START_S,
        "delta_ma": current - baseline,
        "baseline_ma": baseline,
    }

def build_mean_waveform(runs, start_s, end_s):
    grid = np.arange(start_s, end_s + UNIFORM_DT_S / 2, UNIFORM_DT_S)
    traces = []
    for run in runs:
        t = np.asarray(run["time_active_s"], float)
        y = np.asarray(run["delta_ma"], float)
        traces.append(np.interp(grid, t, y, left=np.nan, right=np.nan))
    matrix = np.vstack(traces)
    return grid, np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0, ddof=1)

def plot_active_start(single_runs, continuous_runs) -> Path:
    start_s, end_s = START_WINDOW_S
    ts, ys, _ = build_mean_waveform(single_runs, start_s, end_s)
    tc, yc, _ = build_mean_waveform(continuous_runs, start_s, end_s)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts * 1000, ys, linewidth=1.2, label="Single 1 ms — measured mean")
    ax.plot(tc * 1000, yc, linewidth=1.2, label="Continuous ADC — measured mean")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="Firmware active start")
    ax.set_xlabel("Time relative to firmware ADC active start [ms]")
    ax.set_ylabel("Current increase from idle baseline [mA]")
    ax.set_title("ADC Single 1 ms vs Continuous ADC — active-start transition")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "single_1ms_vs_continuous_active_start.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

def plot_active_zoom(single_runs, continuous_runs) -> Path:
    start_s, end_s = ZOOM_WINDOW_S
    ts, ys, _ = build_mean_waveform(single_runs, start_s, end_s)
    tc, yc, _ = build_mean_waveform(continuous_runs, start_s, end_s)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot((ts-start_s)*1000, ys, linewidth=1.0, label="Single 1 ms — measured mean")
    ax.plot((tc-start_s)*1000, yc, linewidth=1.0, label="Continuous ADC — measured mean")

    for k in range(11):
        ax.axvline(k, linestyle=":", linewidth=0.7, alpha=0.3)

    ax.set_xlabel(f"Time within {start_s:.3f}–{end_s:.3f} s active window [ms]")
    ax.set_ylabel("Current increase from idle baseline [mA]")
    ax.set_title("ADC Single 1 ms vs Continuous ADC — 10 ms steady-state zoom")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "single_1ms_vs_continuous_10ms_zoom.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

def plot_single_runs_zoom(single_runs) -> Path:
    start_s, end_s = ZOOM_WINDOW_S
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, run in enumerate(single_runs, start=1):
        t = np.asarray(run["time_active_s"], float)
        y = np.asarray(run["delta_ma"], float)
        mask = (t >= start_s) & (t <= end_s)
        ax.plot(
            (t[mask]-start_s)*1000,
            y[mask],
            linewidth=0.7,
            alpha=0.55,
            label=f"run{idx}",
        )

    for k in range(11):
        ax.axvline(k, linestyle=":", linewidth=0.7, alpha=0.3)

    ax.set_xlabel(f"Time within {start_s:.3f}–{end_s:.3f} s active window [ms]")
    ax.set_ylabel("Current increase from idle baseline [mA]")
    ax.set_title("ADC Single 1 ms — individual measured runs, 10 ms zoom")
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "single_1ms_individual_runs_10ms_zoom.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out

def summarize_active_level(runs, label):
    means = []
    for run in runs:
        t = np.asarray(run["time_active_s"], float)
        y = np.asarray(run["delta_ma"], float)
        mask = (t >= 1.0) & (t <= 19.0)
        means.append(float(np.mean(y[mask])))
    return {
        "condition": label,
        "n_runs": len(means),
        "active_delta_mean_ma": float(np.mean(means)),
        "active_delta_std_across_runs_ma": float(np.std(means, ddof=1)),
        "active_delta_min_run_ma": float(np.min(means)),
        "active_delta_max_run_ma": float(np.max(means)),
    }

def main() -> None:
    single_files = collect_csv_files(SINGLE_1MS_DIR)
    continuous_files = collect_csv_files(CONTINUOUS_DIR)

    print(f"Single 1 ms files: {len(single_files)}")
    print(f"Continuous ADC files: {len(continuous_files)}")

    single_runs = [prepare_run(p) for p in single_files]
    continuous_runs = [prepare_run(p) for p in continuous_files]

    p1 = plot_active_start(single_runs, continuous_runs)
    p2 = plot_active_zoom(single_runs, continuous_runs)
    p3 = plot_single_runs_zoom(single_runs)

    summary = pd.DataFrame([
        summarize_active_level(single_runs, "Single 1 ms"),
        summarize_active_level(continuous_runs, "Continuous ADC"),
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "single_1ms_vs_continuous_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Active-level comparison ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")
    print(f"Wrote: {summary_path}")

if __name__ == "__main__":
    main()