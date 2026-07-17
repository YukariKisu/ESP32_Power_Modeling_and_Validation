from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROCESSED_ROOT = Path("data/processed/v3_ppk/peripheral/uart_tx_only")
RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")


# ============================================================
# Reference values from CPU100-trained ODE model
# ============================================================

CPU_I_IDLE_MA = 47.2668
CPU_I_ACTIVE_MA = 67.4556
CPU_ACTIVE_MINUS_IDLE_MA = CPU_I_ACTIVE_MA - CPU_I_IDLE_MA


# ============================================================
# Analysis windows
# ============================================================

# Use the stable part of the UART active phase.
ACTIVE_START_S = 20.0
ACTIVE_END_S = 30.0

# Use final idle to avoid the sync pulse / recovery part.
FINAL_IDLE_START_S = 32.0
FINAL_IDLE_END_S = 39.0

TOP_PERCENT = 5.0


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
    m = re.search(r"run(\d+)", file_path.stem)

    if m:
        return f"run{m.group(1)}"

    return file_path.stem


def top_percent_mean(values: np.ndarray, top_percent: float) -> float:
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    threshold = np.percentile(values, 100.0 - top_percent)
    top_values = values[values >= threshold]

    if len(top_values) == 0:
        return np.nan

    return float(np.mean(top_values))


def analyze_window(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    start_s: float,
    end_s: float,
    top_percent: float,
    label: str,
) -> dict:
    mask = (time_s >= start_s) & (time_s < end_s)

    if not np.any(mask):
        raise ValueError(f"No samples found for {label} window: {start_s}-{end_s}s")

    values = current_mA[mask]

    mean_mA = float(np.mean(values))
    std_mA = float(np.std(values, ddof=1))
    min_mA = float(np.min(values))
    max_mA = float(np.max(values))
    top_mA = top_percent_mean(values, top_percent)

    return {
        f"{label}_start_s": start_s,
        f"{label}_end_s": end_s,
        f"{label}_mean_mA": mean_mA,
        f"{label}_std_mA": std_mA,
        f"{label}_min_mA": min_mA,
        f"{label}_max_mA": max_mA,
        f"{label}_top5_mean_mA": top_mA,
        f"{label}_top5_minus_mean_mA": top_mA - mean_mA,
    }


def process_one_file(processed_file: Path) -> dict:
    condition = processed_file.parent.name
    run = parse_run_name(processed_file)

    tx_bytes, period_ms = parse_condition(condition)

    print(f"\nProcessing: {processed_file}")
    print(f"condition = {condition}, run = {run}")

    df = pd.read_csv(processed_file, usecols=["time_s", "current_mA"])

    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    current_mA = pd.to_numeric(df["current_mA"], errors="coerce").to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(current_mA)
    time_s = time_s[valid]
    current_mA = current_mA[valid]

    active = analyze_window(
        time_s=time_s,
        current_mA=current_mA,
        start_s=ACTIVE_START_S,
        end_s=ACTIVE_END_S,
        top_percent=TOP_PERCENT,
        label="active",
    )

    final_idle = analyze_window(
        time_s=time_s,
        current_mA=current_mA,
        start_s=FINAL_IDLE_START_S,
        end_s=FINAL_IDLE_END_S,
        top_percent=TOP_PERCENT,
        label="final_idle",
    )

    active_top5_minus_mean = active["active_top5_minus_mean_mA"]
    final_idle_top5_minus_mean = final_idle["final_idle_top5_minus_mean_mA"]

    excess_top5_mA = active_top5_minus_mean - final_idle_top5_minus_mean

    active_top5_minus_final_idle_top5_mA = (
        active["active_top5_mean_mA"] - final_idle["final_idle_top5_mean_mA"]
    )

    active_mean_minus_final_idle_mean_mA = (
        active["active_mean_mA"] - final_idle["final_idle_mean_mA"]
    )

    relative_excess_top5_gain = excess_top5_mA / CPU_ACTIVE_MINUS_IDLE_MA
    relative_active_top5_vs_idle_top5_gain = (
        active_top5_minus_final_idle_top5_mA / CPU_ACTIVE_MINUS_IDLE_MA
    )

    summary = {
        "condition": condition,
        "run": run,
        "source_file": str(processed_file),
        "tx_bytes": tx_bytes,
        "period_ms": period_ms,
        "top_percent": TOP_PERCENT,

        "cpu_I_idle_mA": CPU_I_IDLE_MA,
        "cpu_I_active_mA": CPU_I_ACTIVE_MA,
        "cpu_active_minus_idle_mA": CPU_ACTIVE_MINUS_IDLE_MA,
    }

    summary.update(active)
    summary.update(final_idle)

    summary.update({
        "active_mean_minus_final_idle_mean_mA": active_mean_minus_final_idle_mean_mA,
        "active_top5_minus_final_idle_top5_mA": active_top5_minus_final_idle_top5_mA,
        "excess_top5_mA": excess_top5_mA,
        "relative_excess_top5_gain": relative_excess_top5_gain,
        "relative_active_top5_vs_idle_top5_gain": relative_active_top5_vs_idle_top5_gain,
    })

    print(f"active top5-mean = {active_top5_minus_mean:.6f} mA")
    print(f"final idle top5-mean = {final_idle_top5_minus_mean:.6f} mA")
    print(f"excess top5 = {excess_top5_mA:.6f} mA")

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

    summary_out = RESULTS_ROOT / "summary_uart_active_vs_idle_top5.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved summary: {summary_out}")

    grouped = summary_df.groupby("condition").agg(
        runs=("run", "count"),

        active_mean_mA=("active_mean_mA", "mean"),
        active_top5_mean_mA=("active_top5_mean_mA", "mean"),
        active_top5_minus_mean_mA=("active_top5_minus_mean_mA", "mean"),

        final_idle_mean_mA=("final_idle_mean_mA", "mean"),
        final_idle_top5_mean_mA=("final_idle_top5_mean_mA", "mean"),
        final_idle_top5_minus_mean_mA=("final_idle_top5_minus_mean_mA", "mean"),

        active_mean_minus_final_idle_mean_mA=("active_mean_minus_final_idle_mean_mA", "mean"),
        active_top5_minus_final_idle_top5_mA=("active_top5_minus_final_idle_top5_mA", "mean"),
        excess_top5_mA=("excess_top5_mA", "mean"),

        relative_excess_top5_gain=("relative_excess_top5_gain", "mean"),
        relative_active_top5_vs_idle_top5_gain=("relative_active_top5_vs_idle_top5_gain", "mean"),
    ).reset_index()

    grouped_out = RESULTS_ROOT / "summary_uart_active_vs_idle_top5_grouped.csv"
    grouped.to_csv(grouped_out, index=False)

    print(f"Saved grouped summary: {grouped_out}")
    print("\nGrouped summary:")
    print(grouped.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()