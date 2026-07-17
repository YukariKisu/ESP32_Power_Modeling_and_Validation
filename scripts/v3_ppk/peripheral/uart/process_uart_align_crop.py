from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Settings
# =========================

RAW_ROOT = Path("data/raw/v3_ppk/peripheral/uart_tx_only")
PROCESSED_ROOT = Path("data/processed/v3_ppk/peripheral/uart_tx_only")
PLOT_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only/alignment_check")

# Timing relative to CPU sync pulse rising edge
# t_sync = 0
# 0-1 s   : CPU sync pulse
# 1-6 s   : recovery idle
# 6-16 s  : initial idle
# 16-36 s : UART active
# 36-46 s : final idle
CROP_START_AFTER_SYNC_S = 6.0
CROP_END_AFTER_SYNC_S = 46.0

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0

# Speed settings
MAX_DETECTION_POINTS = 20000
MAX_PLOT_POINTS = 50000

# Sync detection settings
SMOOTH_WINDOW_POINTS = 101
THRESHOLD_STD_MULTIPLIER = 6.0
IGNORE_EARLY_SECONDS = 2.0
MIN_SYNC_DURATION_S = 0.2


# =========================
# Column helpers
# =========================

def find_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "time_s", "Time(s)", "Time (s)", "time", "Time",
        "timestamp_s", "Timestamp"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "time" in col.lower():
            return col

    raise ValueError(f"Could not find time column. Columns: {list(df.columns)}")


def find_current_column(df: pd.DataFrame) -> str:
    candidates = [
        "current_mA", "Current(mA)", "Current (mA)",
        "current", "Current"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "current" in col.lower():
            return col

    raise ValueError(f"Could not find current column. Columns: {list(df.columns)}")


def normalize_time_to_seconds(series: pd.Series) -> pd.Series:
    t = pd.to_numeric(series, errors="coerce")
    max_t = t.max()

    if max_t > 1e6:
        return t / 1_000_000.0   # likely microseconds
    elif max_t > 10_000:
        return t / 1000.0        # likely milliseconds
    else:
        return t                 # likely seconds


# =========================
# Speed helpers
# =========================

def downsample_arrays(time_s: np.ndarray, current_mA: np.ndarray, max_points: int):
    if len(time_s) <= max_points:
        return time_s, current_mA

    step = int(np.ceil(len(time_s) / max_points))
    return time_s[::step], current_mA[::step]


def downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


# =========================
# Sync detection
# =========================

def detect_sync_pulse(time_s: np.ndarray, current_mA: np.ndarray) -> float:
    """
    Detect the rising edge of the CPU sync pulse.
    Detection uses downsampled data for speed.
    Original-resolution data is still used for final crop/save.
    """

    time_ds, current_ds = downsample_arrays(
        time_s,
        current_mA,
        MAX_DETECTION_POINTS
    )

    s = pd.Series(current_ds)

    smooth = s.rolling(
        window=SMOOTH_WINDOW_POINTS,
        center=True,
        min_periods=max(5, SMOOTH_WINDOW_POINTS // 10)
    ).median()

    smooth = smooth.bfill().ffill().to_numpy()

    valid_mask = time_ds > (time_ds.min() + IGNORE_EARLY_SECONDS)

    t_valid = time_ds[valid_mask]
    y_valid = smooth[valid_mask]

    if len(t_valid) < 100:
        raise ValueError("Not enough data after ignoring early startup.")

    # Estimate idle baseline from the lower-current region
    q30 = np.quantile(y_valid, 0.30)
    baseline_values = y_valid[y_valid <= q30]

    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)

    threshold = baseline_mean + THRESHOLD_STD_MULTIPLIER * baseline_std

    above = y_valid > threshold

    if not np.any(above):
        raise ValueError(
            f"Could not detect sync pulse. "
            f"baseline_mean={baseline_mean:.3f}, "
            f"baseline_std={baseline_std:.3f}, "
            f"threshold={threshold:.3f}"
        )

    dt = np.median(np.diff(t_valid))
    min_samples = max(3, int(MIN_SYNC_DURATION_S / dt))

    above_int = above.astype(int)
    conv = np.convolve(
        above_int,
        np.ones(min_samples, dtype=int),
        mode="same"
    )

    sustained = conv >= min_samples

    if not np.any(sustained):
        raise ValueError("Detected high-current points, but no sustained sync pulse.")

    idx = np.argmax(sustained)

    # Walk backward to approximate rising edge
    while idx > 0 and above[idx - 1]:
        idx -= 1

    return float(t_valid[idx])


# =========================
# Processing helpers
# =========================

def add_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time_s"].to_numpy()

    phase = np.full(len(df), "unknown", dtype=object)

    phase[(t >= 0.0) & (t < INITIAL_IDLE_END_S)] = "initial_idle"
    phase[(t >= INITIAL_IDLE_END_S) & (t < UART_ACTIVE_END_S)] = "uart_active"
    phase[(t >= UART_ACTIVE_END_S) & (t <= FINAL_END_S)] = "final_idle"

    df["phase"] = phase
    return df


def parse_condition_from_path(path: Path) -> str:
    # Expected folder name example: uart_64B_100ms
    for part in path.parts:
        if re.match(r"uart_\d+B_\d+ms", part):
            return part

    return path.parent.name


def process_one_file(raw_file: Path) -> None:
    condition = parse_condition_from_path(raw_file)
    run_name = raw_file.stem

    print(f"\nProcessing: {raw_file}")
    print(f"Condition: {condition}")

    df = pd.read_csv(raw_file, comment="#")

    time_col = find_time_column(df)
    current_col = find_current_column(df)

    time_s = normalize_time_to_seconds(df[time_col])
    current_raw = pd.to_numeric(df[current_col], errors="coerce")

    current_col_lower = current_col.lower()

    if "ua" in current_col_lower or "µa" in current_col_lower or "μa" in current_col_lower:
        current_mA = current_raw / 1000.0
    else:
        current_mA = current_raw

    clean = pd.DataFrame({
        "time_raw_s": time_s,
        "current_mA": current_mA,
    }).dropna()

    clean["time_raw_s"] = clean["time_raw_s"] - clean["time_raw_s"].iloc[0]

    t = clean["time_raw_s"].to_numpy()
    y = clean["current_mA"].to_numpy()

    t_sync = detect_sync_pulse(t, y)

    crop_start = t_sync + CROP_START_AFTER_SYNC_S
    crop_end = t_sync + CROP_END_AFTER_SYNC_S

    cropped = clean[
        (clean["time_raw_s"] >= crop_start) &
        (clean["time_raw_s"] <= crop_end)
    ].copy()

    if cropped.empty:
        raise ValueError("Cropped dataframe is empty. Check sync detection and timing.")

    cropped["time_s"] = cropped["time_raw_s"] - crop_start
    cropped = cropped[["time_s", "current_mA"]]
    cropped = add_phase_column(cropped)

    out_dir = PROCESSED_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"aligned_{condition}_{run_name}.csv"
    cropped.to_csv(out_csv, index=False)

    print(f"Detected t_sync = {t_sync:.6f} s")
    print(f"Crop: {crop_start:.6f} s to {crop_end:.6f} s")
    print(f"Saved CSV: {out_csv}")

    # Plot check using downsampled data only
    plot_dir = PLOT_ROOT / condition
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_df = downsample_df(clean, MAX_PLOT_POINTS)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        plot_df["time_raw_s"],
        plot_df["current_mA"],
        linewidth=0.8,
        label="raw current"
    )

    ax.axvline(
        t_sync,
        linestyle="--",
        label="sync pulse rising edge"
    )

    ax.axvspan(
        crop_start,
        crop_end,
        alpha=0.15,
        label="cropped validation interval"
    )

    ax.set_title(f"Alignment check: {condition} / {raw_file.name}")
    ax.set_xlabel("Raw time [s]")
    ax.set_ylabel("Current [mA]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_plot = plot_dir / f"alignment_check_{condition}_{run_name}.png"
    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {out_plot}")


def main():
    raw_files = sorted(RAW_ROOT.glob("uart_*_100ms/*.csv"))

    if not raw_files:
        raise FileNotFoundError(f"No CSV files found under {RAW_ROOT}")

    print(f"Found {len(raw_files)} raw CSV files.")

    for raw_file in raw_files:
        process_one_file(raw_file)

    print("\nDone.")


if __name__ == "__main__":
    main()