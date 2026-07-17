from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROCESSED_ROOT = Path("data/processed/v3_ppk/peripheral/uart_tx_only")
RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")


# ============================================================
# Model reference values from CPU100-trained ODE model
# ============================================================

CPU_I_IDLE_MA = 47.2668
CPU_I_ACTIVE_MA = 67.4556

CPU_ACTIVE_MINUS_IDLE_MA = CPU_I_ACTIVE_MA - CPU_I_IDLE_MA


# ============================================================
# UART / validation timing
# ============================================================

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_START_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0

UART_BAUD_RATE = 115200
UART_BITS_PER_BYTE_8N1 = 10

# Idle window used as measured baseline
IDLE_BASELINE_START_S = 2.0
IDLE_BASELINE_END_S = 9.0

# Analyze each UART period inside active phase
TOP_PERCENT = 5.0

# Plot
MAX_PLOT_POINTS = 50000


# ============================================================
# Helpers
# ============================================================

def parse_condition(condition_name: str):
    """
    Expected:
        uart_64B_100ms
        uart_256B_100ms
        uart_512B_100ms
    """
    m = re.match(r"uart_(\d+)B_(\d+)ms", condition_name)

    if not m:
        raise ValueError(f"Could not parse condition name: {condition_name}")

    tx_bytes = int(m.group(1))
    period_ms = int(m.group(2))

    return tx_bytes, period_ms


def parse_run_name(file_path: Path) -> str:
    """
    Example:
        aligned_uart_64B_100ms_uart_64B_100ms_run1.csv
    """
    m = re.search(r"run(\d+)", file_path.stem)

    if m:
        return f"run{m.group(1)}"

    return file_path.stem


def compute_tx_duration_s(tx_bytes: int) -> float:
    return tx_bytes * UART_BITS_PER_BYTE_8N1 / UART_BAUD_RATE


def compute_uart_occupancy(tx_bytes: int, period_ms: int) -> float:
    period_s = period_ms / 1000.0
    tx_duration_s = compute_tx_duration_s(tx_bytes)
    return tx_duration_s / period_s


def top_percent_mean(values: np.ndarray, top_percent: float) -> float:
    """
    Mean of the top X percent values.
    More stable than max, but still represents the high-current part.
    """
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    percentile = 100.0 - top_percent
    threshold = np.percentile(values, percentile)
    top_values = values[values >= threshold]

    if len(top_values) == 0:
        return np.nan

    return float(np.mean(top_values))


def analyze_period_levels(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    period_s: float,
) -> pd.DataFrame:
    """
    Analyze every 100 ms UART period in the active phase.

    For each period:
        - mean current
        - max current
        - top 5% mean current
    """

    rows = []

    n_periods = int(round((UART_ACTIVE_END_S - UART_ACTIVE_START_S) / period_s))

    for i in range(n_periods):
        period_start_s = UART_ACTIVE_START_S + i * period_s
        period_end_s = period_start_s + period_s

        mask = (time_s >= period_start_s) & (time_s < period_end_s)

        if not np.any(mask):
            continue

        values = current_mA[mask]

        rows.append({
            "period_index": i,
            "period_start_s": period_start_s,
            "period_end_s": period_end_s,
            "period_mean_mA": float(np.mean(values)),
            "period_max_mA": float(np.max(values)),
            "period_top5_mean_mA": top_percent_mean(values, TOP_PERCENT),
        })

    return pd.DataFrame(rows)


def make_plot(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    idle_mean_mA: float,
    active_period_top5_mean_mA: float,
    active_period_max_mean_mA: float,
    out_plot: Path,
    title: str,
):
    n = len(time_s)

    if n > MAX_PLOT_POINTS:
        step = int(np.ceil(n / MAX_PLOT_POINTS))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    t_plot = time_s[idx]
    i_plot = current_mA[idx]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_plot, i_plot, linewidth=0.8, label="measured current")

    ax.axvspan(0, 10, alpha=0.08, label="initial idle")
    ax.axvspan(10, 30, alpha=0.08, label="UART active")
    ax.axvspan(30, 40, alpha=0.08, label="final idle")

    ax.axhline(idle_mean_mA, linestyle="--", linewidth=1.0, label="measured idle mean")
    ax.axhline(CPU_I_ACTIVE_MA, linestyle="--", linewidth=1.0, label="CPU I_active")
    ax.axhline(active_period_top5_mean_mA, linestyle="--", linewidth=1.0, label="UART period top5 mean")
    ax.axhline(active_period_max_mean_mA, linestyle=":", linewidth=1.0, label="UART period max mean")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [mA]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)


def process_one_file(processed_file: Path) -> dict:
    condition = processed_file.parent.name
    run = parse_run_name(processed_file)

    tx_bytes, period_ms = parse_condition(condition)
    period_s = period_ms / 1000.0

    tx_duration_s = compute_tx_duration_s(tx_bytes)
    uart_occupancy = compute_uart_occupancy(tx_bytes, period_ms)

    print(f"\nProcessing: {processed_file}")
    print(f"condition = {condition}, run = {run}")

    df = pd.read_csv(
        processed_file,
        usecols=["time_s", "current_mA"]
    )

    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    current_mA = pd.to_numeric(df["current_mA"], errors="coerce").to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(current_mA)
    time_s = time_s[valid]
    current_mA = current_mA[valid]

    idle_mask = (time_s >= IDLE_BASELINE_START_S) & (time_s < IDLE_BASELINE_END_S)
    active_mask = (time_s >= UART_ACTIVE_START_S) & (time_s < UART_ACTIVE_END_S)

    if not np.any(idle_mask):
        raise ValueError(f"No idle samples found in {processed_file}")

    if not np.any(active_mask):
        raise ValueError(f"No active samples found in {processed_file}")

    idle_mean_mA = float(np.mean(current_mA[idle_mask]))
    active_mean_mA = float(np.mean(current_mA[active_mask]))

    period_df = analyze_period_levels(
        time_s=time_s,
        current_mA=current_mA,
        period_s=period_s,
    )

    active_period_mean_mA = float(period_df["period_mean_mA"].mean())
    active_period_max_mean_mA = float(period_df["period_max_mA"].mean())
    active_period_top5_mean_mA = float(period_df["period_top5_mean_mA"].mean())

    uart_mean_minus_idle_mA = active_period_mean_mA - idle_mean_mA
    uart_max_minus_idle_mA = active_period_max_mean_mA - idle_mean_mA
    uart_top5_minus_idle_mA = active_period_top5_mean_mA - idle_mean_mA

    relative_uart_gain_mean = uart_mean_minus_idle_mA / CPU_ACTIVE_MINUS_IDLE_MA
    relative_uart_gain_max = uart_max_minus_idle_mA / CPU_ACTIVE_MINUS_IDLE_MA
    relative_uart_gain_top5 = uart_top5_minus_idle_mA / CPU_ACTIVE_MINUS_IDLE_MA

    out_dir = RESULTS_ROOT / condition / "active_peak_levels"
    out_dir.mkdir(parents=True, exist_ok=True)

    period_out = out_dir / f"period_levels_{processed_file.stem}.csv"
    period_df.to_csv(period_out, index=False)

    out_plot = out_dir / f"plot_active_peak_levels_{processed_file.stem}.png"
    make_plot(
        time_s=time_s,
        current_mA=current_mA,
        idle_mean_mA=idle_mean_mA,
        active_period_top5_mean_mA=active_period_top5_mean_mA,
        active_period_max_mean_mA=active_period_max_mean_mA,
        out_plot=out_plot,
        title=f"UART active peak levels: {condition} / {run}",
    )

    summary = {
        "condition": condition,
        "run": run,
        "source_file": str(processed_file),
        "tx_bytes": tx_bytes,
        "period_ms": period_ms,
        "tx_duration_s": tx_duration_s,
        "tx_duration_ms": tx_duration_s * 1000.0,
        "uart_occupancy": uart_occupancy,

        "idle_mean_mA": idle_mean_mA,
        "active_phase_mean_mA": active_mean_mA,

        "active_period_mean_mA": active_period_mean_mA,
        "active_period_max_mean_mA": active_period_max_mean_mA,
        "active_period_top5_mean_mA": active_period_top5_mean_mA,

        "cpu_I_active_mA": CPU_I_ACTIVE_MA,
        "cpu_I_idle_mA": CPU_I_IDLE_MA,

        "uart_mean_minus_idle_mA": uart_mean_minus_idle_mA,
        "uart_max_minus_idle_mA": uart_max_minus_idle_mA,
        "uart_top5_minus_idle_mA": uart_top5_minus_idle_mA,

        "cpu_active_minus_idle_mA": CPU_ACTIVE_MINUS_IDLE_MA,

        "relative_uart_gain_mean": relative_uart_gain_mean,
        "relative_uart_gain_max": relative_uart_gain_max,
        "relative_uart_gain_top5": relative_uart_gain_top5,

        "period_level_csv": str(period_out),
        "output_plot": str(out_plot),
    }

    print(f"Saved period-level CSV: {period_out}")
    print(f"Saved plot: {out_plot}")

    return summary


def main():
    processed_files = sorted(PROCESSED_ROOT.glob("uart_*_100ms/aligned_*.csv"))

    if not processed_files:
        raise FileNotFoundError(f"No aligned CSV files found under {PROCESSED_ROOT}")

    print(f"Found {len(processed_files)} processed CSV files.")

    summaries = []

    for processed_file in processed_files:
        summaries.append(process_one_file(processed_file))

    summary_df = pd.DataFrame(summaries)

    summary_out = RESULTS_ROOT / "summary_uart_active_peak_levels.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved summary: {summary_out}")

    grouped = summary_df.groupby("condition").agg(
        runs=("run", "count"),
        idle_mean_mA=("idle_mean_mA", "mean"),
        active_phase_mean_mA=("active_phase_mean_mA", "mean"),
        active_period_mean_mA=("active_period_mean_mA", "mean"),
        active_period_max_mean_mA=("active_period_max_mean_mA", "mean"),
        active_period_top5_mean_mA=("active_period_top5_mean_mA", "mean"),
        uart_mean_minus_idle_mA=("uart_mean_minus_idle_mA", "mean"),
        uart_max_minus_idle_mA=("uart_max_minus_idle_mA", "mean"),
        uart_top5_minus_idle_mA=("uart_top5_minus_idle_mA", "mean"),
        relative_uart_gain_mean=("relative_uart_gain_mean", "mean"),
        relative_uart_gain_max=("relative_uart_gain_max", "mean"),
        relative_uart_gain_top5=("relative_uart_gain_top5", "mean"),
    ).reset_index()

    grouped_out = RESULTS_ROOT / "summary_uart_active_peak_levels_grouped.csv"
    grouped.to_csv(grouped_out, index=False)

    print(f"Saved grouped summary: {grouped_out}")
    print("\nGrouped summary:")
    print(grouped.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()