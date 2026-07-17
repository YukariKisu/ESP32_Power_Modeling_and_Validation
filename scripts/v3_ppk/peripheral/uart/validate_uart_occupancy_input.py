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

D1_IDLE_BASELINE_DIR = Path("data/raw/v3_ppk/idle_baseline")

D3_UART_ENABLED_SUMMARY = Path(
    "results/v3_ppk/peripheral/uart_tx_only/uart_enabled/"
    "uart_enabled_idle_mean_20_30s.csv"
)


# ============================================================
# UART settings
# ============================================================

UART_BAUD_RATE = 115200
UART_BITS_PER_BYTE_8N1 = 10
UART_PERIOD_MS = 100

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0

DECOMPOSITION_WINDOW_START_S = 20.0
DECOMPOSITION_WINDOW_END_S = 30.0


# ============================================================
# ODE model parameters
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_TAU_S = 0.0004895

USE_IDLE_OFFSET_CORRECTION = True

EVAL_WINDOWS = {
    "initial_idle_center": (2.0, 9.0),
    "uart_active_center": (12.0, 29.0),
    "final_idle_center": (32.0, 39.0),
    "full_validation": (0.0, 40.0),
}


# ============================================================
# Helpers
# ============================================================

def natural_key(path: Path):
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else 0


def parse_run_index(text: str) -> int:
    m = re.search(r"run(\d+)", text)

    if not m:
        raise ValueError(f"Could not parse run index from: {text}")

    return int(m.group(1))


def parse_condition(condition_name: str):
    m = re.match(r"uart_(\d+)B_(\d+)ms", condition_name)

    if not m:
        raise ValueError(f"Could not parse condition name: {condition_name}")

    tx_bytes = int(m.group(1))
    period_ms = int(m.group(2))

    return tx_bytes, period_ms


def load_ppk2_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Timestamp(ms)" not in df.columns or "Current(uA)" not in df.columns:
        raise ValueError(f"Unexpected raw CSV columns in {path}: {df.columns.tolist()}")

    df["time_s"] = df["Timestamp(ms)"] / 1000.0
    df["current_mA"] = df["Current(uA)"] / 1000.0

    return df


def mean_current_window(df: pd.DataFrame, t0: float, t1: float) -> float:
    mask = (df["time_s"] >= t0) & (df["time_s"] < t1)

    if not mask.any():
        raise ValueError(f"No samples found in window {t0}-{t1}s")

    return df.loc[mask, "current_mA"].mean()


def load_d1_idle_baseline_means() -> dict:
    files = sorted(
        D1_IDLE_BASELINE_DIR.glob("ppk_idle_baseline_run*.csv"),
        key=natural_key
    )

    if not files:
        raise FileNotFoundError(f"No D1 idle baseline files found in {D1_IDLE_BASELINE_DIR}")

    result = {}

    for path in files:
        run_index = parse_run_index(path.name)
        df = load_ppk2_raw_csv(path)
        mean_mA = mean_current_window(
            df,
            DECOMPOSITION_WINDOW_START_S,
            DECOMPOSITION_WINDOW_END_S
        )
        result[run_index] = mean_mA

    return result


def load_d3_uart_enabled_means() -> dict:
    if not D3_UART_ENABLED_SUMMARY.exists():
        raise FileNotFoundError(f"D3 summary not found: {D3_UART_ENABLED_SUMMARY}")

    df = pd.read_csv(D3_UART_ENABLED_SUMMARY)

    if "mean_current_mA" not in df.columns:
        raise ValueError(
            f"D3 summary does not contain mean_current_mA: {D3_UART_ENABLED_SUMMARY}"
        )

    result = {}

    if "run_index" in df.columns:
        for _, row in df.iterrows():
            result[int(row["run_index"])] = float(row["mean_current_mA"])
    elif "run" in df.columns:
        for _, row in df.iterrows():
            run_index = parse_run_index(str(row["run"]))
            result[run_index] = float(row["mean_current_mA"])
    else:
        for idx, row in df.iterrows():
            result[idx + 1] = float(row["mean_current_mA"])

    return result


def compute_uart_occupancy(tx_bytes: int, period_ms: int) -> float:
    bytes_per_second = tx_bytes * 1000.0 / period_ms
    max_bytes_per_second = UART_BAUD_RATE / UART_BITS_PER_BYTE_8N1

    occupancy = bytes_per_second / max_bytes_per_second

    return occupancy


def add_occupancy_input(df: pd.DataFrame, occupancy: float) -> pd.DataFrame:
    t = df["time_s"].to_numpy()

    u = np.zeros(len(df), dtype=float)
    active_mask = (t >= INITIAL_IDLE_END_S) & (t < UART_ACTIVE_END_S)
    u[active_mask] = occupancy

    df["u_uart_occupancy"] = u

    return df


def simulate_first_order_ode(time_s: np.ndarray, u: np.ndarray, i0_mA: float) -> np.ndarray:
    if MODEL_TAU_S <= 0:
        raise ValueError("MODEL_TAU_S must be set to a positive value.")

    pred = np.zeros(len(time_s), dtype=float)
    pred[0] = i0_mA

    for k in range(1, len(time_s)):
        dt = time_s[k] - time_s[k - 1]

        if dt <= 0:
            pred[k] = pred[k - 1]
            continue

        i_target = MODEL_I_IDLE_MA + u[k - 1] * (MODEL_I_ACTIVE_MA - MODEL_I_IDLE_MA)

        alpha = math.exp(-dt / MODEL_TAU_S)
        pred[k] = i_target + (pred[k - 1] - i_target) * alpha

    return pred


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {}

    for name, (t0, t1) in EVAL_WINDOWS.items():
        mask = (df["time_s"] >= t0) & (df["time_s"] < t1)

        if not mask.any():
            continue

        err = df.loc[mask, "error_mA"].to_numpy()
        measured = df.loc[mask, "current_mA"].to_numpy()
        predicted = df.loc[mask, "current_pred_mA"].to_numpy()

        metrics[f"{name}_mae_mA"] = np.mean(np.abs(err))
        metrics[f"{name}_rmse_mA"] = math.sqrt(np.mean(err ** 2))
        metrics[f"{name}_mean_measured_mA"] = np.mean(measured)
        metrics[f"{name}_mean_predicted_mA"] = np.mean(predicted)
        metrics[f"{name}_mean_error_mA"] = np.mean(err)

    initial_mask = (df["time_s"] >= 2.0) & (df["time_s"] < 9.0)
    active_mask = (df["time_s"] >= 12.0) & (df["time_s"] < 29.0)
    final_mask = (df["time_s"] >= 32.0) & (df["time_s"] < 39.0)

    if initial_mask.any() and active_mask.any():
        initial_mean = df.loc[initial_mask, "current_mA"].mean()
        active_mean = df.loc[active_mask, "current_mA"].mean()
        metrics["measured_active_minus_initial_idle_mA"] = active_mean - initial_mean

    if final_mask.any() and active_mask.any():
        final_mean = df.loc[final_mask, "current_mA"].mean()
        active_mean = df.loc[active_mask, "current_mA"].mean()
        metrics["measured_active_minus_final_idle_mA"] = active_mean - final_mean

    return metrics


def make_plot(df: pd.DataFrame, out_plot: Path, title: str):
    plot_df = df

    max_plot_points = 50000
    if len(plot_df) > max_plot_points:
        step = int(np.ceil(len(plot_df) / max_plot_points))
        plot_df = plot_df.iloc[::step].copy()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        plot_df["time_s"],
        plot_df["current_mA"],
        linewidth=0.8,
        label="measured current"
    )

    ax1.plot(
        plot_df["time_s"],
        plot_df["current_pred_mA"],
        linewidth=1.0,
        label="ODE prediction"
    )

    ax1.axvspan(0, 10, alpha=0.08, label="initial idle")
    ax1.axvspan(10, 30, alpha=0.08, label="UART active")
    ax1.axvspan(30, 40, alpha=0.08, label="final idle")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Current [mA]")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["time_s"],
        plot_df["u_uart_occupancy"],
        linewidth=0.8,
        linestyle="--",
        label="u_uart_occupancy"
    )
    ax2.set_ylabel("UART occupancy input")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)


def process_one_file(
    processed_file: Path,
    d1_means_by_run: dict,
    d3_means_by_run: dict
) -> dict:
    condition = processed_file.parent.name
    tx_bytes, period_ms = parse_condition(condition)
    run_index = parse_run_index(processed_file.name)

    occupancy = compute_uart_occupancy(tx_bytes, period_ms)

    print(f"\nProcessing: {processed_file}")
    print(f"Condition: {condition}")
    print(f"run_index = {run_index}")
    print(f"tx_bytes = {tx_bytes}")
    print(f"period_ms = {period_ms}")
    print(f"UART occupancy = {occupancy:.6f}")

    df = pd.read_csv(processed_file)

    required_cols = {"time_s", "current_mA", "phase"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {processed_file}: {missing}")

    df = add_occupancy_input(df, occupancy)

    time_s = df["time_s"].to_numpy()
    measured = df["current_mA"].to_numpy()
    u = df["u_uart_occupancy"].to_numpy()

    i0_mA = measured[0]

    pred = simulate_first_order_ode(time_s, u, i0_mA)

    df["current_pred_mA"] = pred

    if USE_IDLE_OFFSET_CORRECTION:
        idle_mask = (df["time_s"] >= 2.0) & (df["time_s"] < 9.0)

        measured_idle_mean = df.loc[idle_mask, "current_mA"].mean()
        predicted_idle_mean = df.loc[idle_mask, "current_pred_mA"].mean()

        offset = measured_idle_mean - predicted_idle_mean
        df["current_pred_mA"] = df["current_pred_mA"] + offset
        df["prediction_offset_mA"] = offset
    else:
        df["prediction_offset_mA"] = 0.0

    df["error_mA"] = df["current_pred_mA"] - df["current_mA"]

    d4_active_mean_mA = mean_current_window(
        df,
        DECOMPOSITION_WINDOW_START_S,
        DECOMPOSITION_WINDOW_END_S
    )

    if run_index not in d1_means_by_run:
        raise ValueError(f"D1 idle baseline missing for run {run_index}")

    if run_index not in d3_means_by_run:
        raise ValueError(f"D3 UART enabled idle missing for run {run_index}")

    d1_idle_mA = d1_means_by_run[run_index]
    d3_enabled_mA = d3_means_by_run[run_index]

    uart_enable_overhead_mA = d3_enabled_mA - d1_idle_mA
    tx_activity_overhead_mA = d4_active_mean_mA - d3_enabled_mA
    total_uart_difference_mA = d4_active_mean_mA - d1_idle_mA

    out_dir = RESULTS_ROOT / condition / "occupancy_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"prediction_occupancy_{processed_file.stem}.csv"
    df.to_csv(out_csv, index=False)

    out_plot = out_dir / f"plot_occupancy_{processed_file.stem}.png"
    make_plot(
        df,
        out_plot,
        title=f"UART occupancy input validation: {condition} / {processed_file.name}"
    )

    metrics = compute_metrics(df)

    summary = {
        "condition": condition,
        "run_index": run_index,
        "source_file": str(processed_file),
        "tx_bytes": tx_bytes,
        "period_ms": period_ms,
        "uart_baud_rate": UART_BAUD_RATE,
        "uart_format": "8N1",
        "uart_occupancy": occupancy,
        "model_i_idle_mA": MODEL_I_IDLE_MA,
        "model_i_active_mA": MODEL_I_ACTIVE_MA,
        "model_tau_s": MODEL_TAU_S,
        "use_idle_offset_correction": USE_IDLE_OFFSET_CORRECTION,

        "decomposition_window_start_s": DECOMPOSITION_WINDOW_START_S,
        "decomposition_window_end_s": DECOMPOSITION_WINDOW_END_S,
        "D1_idle_baseline_mA": d1_idle_mA,
        "D3_uart_enabled_idle_mA": d3_enabled_mA,
        "D4_uart_tx_active_mA": d4_active_mean_mA,
        "D3_minus_D1_uart_enable_overhead_mA": uart_enable_overhead_mA,
        "D4_minus_D3_tx_activity_overhead_mA": tx_activity_overhead_mA,
        "D4_minus_D1_total_uart_difference_mA": total_uart_difference_mA,

        "output_prediction_csv": str(out_csv),
        "output_plot": str(out_plot),
    }

    summary.update(metrics)

    print(f"D1 idle baseline = {d1_idle_mA:.6f} mA")
    print(f"D3 UART enabled idle = {d3_enabled_mA:.6f} mA")
    print(f"D4 UART TX active = {d4_active_mean_mA:.6f} mA")
    print(f"D3-D1 enable overhead = {uart_enable_overhead_mA:.6f} mA")
    print(f"D4-D3 TX activity overhead = {tx_activity_overhead_mA:.6f} mA")
    print(f"D4-D1 total UART difference = {total_uart_difference_mA:.6f} mA")

    print(f"Saved prediction CSV: {out_csv}")
    print(f"Saved plot: {out_plot}")

    return summary


def save_condition_decomposition_summary(summary_df: pd.DataFrame):
    cols = [
        "D1_idle_baseline_mA",
        "D3_uart_enabled_idle_mA",
        "D4_uart_tx_active_mA",
        "D3_minus_D1_uart_enable_overhead_mA",
        "D4_minus_D3_tx_activity_overhead_mA",
        "D4_minus_D1_total_uart_difference_mA",
    ]

    decomposition_summary = (
        summary_df
        .groupby("condition")
        .agg(
            runs=("run_index", "count"),
            D1_mean_mA=("D1_idle_baseline_mA", "mean"),
            D3_mean_mA=("D3_uart_enabled_idle_mA", "mean"),
            D4_mean_mA=("D4_uart_tx_active_mA", "mean"),
            uart_enable_overhead_mean_mA=("D3_minus_D1_uart_enable_overhead_mA", "mean"),
            uart_enable_overhead_std_mA=("D3_minus_D1_uart_enable_overhead_mA", "std"),
            tx_activity_overhead_mean_mA=("D4_minus_D3_tx_activity_overhead_mA", "mean"),
            tx_activity_overhead_std_mA=("D4_minus_D3_tx_activity_overhead_mA", "std"),
            total_uart_difference_mean_mA=("D4_minus_D1_total_uart_difference_mA", "mean"),
            total_uart_difference_std_mA=("D4_minus_D1_total_uart_difference_mA", "std"),
        )
        .reset_index()
    )

    out = RESULTS_ROOT / "summary_uart_occupancy_baseline_decomposition.csv"
    decomposition_summary.to_csv(out, index=False)

    print(f"Saved decomposition summary: {out}")


def main():
    processed_files = sorted(PROCESSED_ROOT.glob("uart_*_100ms/aligned_*.csv"))

    if not processed_files:
        raise FileNotFoundError(f"No aligned CSV files found under {PROCESSED_ROOT}")

    print(f"Found {len(processed_files)} processed CSV files.")

    d1_means_by_run = load_d1_idle_baseline_means()
    d3_means_by_run = load_d3_uart_enabled_means()

    print(f"Loaded D1 idle baseline runs: {sorted(d1_means_by_run.keys())}")
    print(f"Loaded D3 UART enabled runs: {sorted(d3_means_by_run.keys())}")

    summaries = []

    for processed_file in processed_files:
        summary = process_one_file(
            processed_file,
            d1_means_by_run,
            d3_means_by_run
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_out = RESULTS_ROOT / "summary_uart_occupancy_input_validation.csv"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    save_condition_decomposition_summary(summary_df)

    print(f"\nSaved summary: {summary_out}")
    print("Done.")


if __name__ == "__main__":
    main()