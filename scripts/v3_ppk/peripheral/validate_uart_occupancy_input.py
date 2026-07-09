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
# UART settings
# ============================================================

UART_BAUD_RATE = 115200
UART_BITS_PER_BYTE_8N1 = 10
UART_PERIOD_MS = 100

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0


# ============================================================
# ODE model parameters
# IMPORTANT:
# Replace these values with the parameters from your CPU100-trained ODE model.
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_TAU_S = 0.0004895

# If True, prediction is vertically shifted so that the initial-idle mean
# matches the measured initial-idle mean.
# Use the same policy as your previous CPU validation scripts.
USE_IDLE_OFFSET_CORRECTION = True
SAVE_PREDICTION_CSV = False

# Evaluation windows.
# These avoid phase boundaries where timing/transition uncertainty is larger.
EVAL_WINDOWS = {
    "initial_idle_center": (2.0, 9.0),
    "uart_active_center": (12.0, 29.0),
    "final_idle_center": (32.0, 39.0),
    "full_validation": (0.0, 40.0),
}


# ============================================================
# Helpers
# ============================================================

def parse_condition(condition_name: str):
    """
    Expected condition name:
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


def compute_uart_occupancy(tx_bytes: int, period_ms: int) -> float:
    """
    Normalized UART TX occupancy.

    bytes_per_second = tx_bytes * 1000 / period_ms
    max_bytes_per_second = baud_rate / 10 for 8N1 UART framing
    """

    bytes_per_second = tx_bytes * 1000.0 / period_ms
    max_bytes_per_second = UART_BAUD_RATE / UART_BITS_PER_BYTE_8N1

    occupancy = bytes_per_second / max_bytes_per_second

    return occupancy


def add_occupancy_input(df: pd.DataFrame, occupancy: float) -> pd.DataFrame:
    """
    Phase-level occupancy input:

        0-10 s   : u = 0
        10-30 s  : u = occupancy
        30-40 s  : u = 0
    """

    t = df["time_s"].to_numpy()

    u = np.zeros(len(df), dtype=float)
    active_mask = (t >= INITIAL_IDLE_END_S) & (t < UART_ACTIVE_END_S)
    u[active_mask] = occupancy

    df["u_uart_occupancy"] = u

    return df


def simulate_occupancy_ode_vectorized(time_s: np.ndarray, occupancy: float, i0_mA: float) -> np.ndarray:
    """
    Fast vectorized prediction for phase-level UART occupancy input.

    Input:
        0-10 s   : u = 0
        10-30 s  : u = occupancy
        30-40 s  : u = 0
    """

    if MODEL_TAU_S <= 0:
        raise ValueError("MODEL_TAU_S must be set to a positive value.")

    i_idle_target = MODEL_I_IDLE_MA
    i_uart_target = MODEL_I_IDLE_MA + occupancy * (MODEL_I_ACTIVE_MA - MODEL_I_IDLE_MA)

    pred = np.zeros(len(time_s), dtype=float)

    # Segment 1: initial idle, 0-10s
    mask1 = time_s < INITIAL_IDLE_END_S
    t1 = time_s[mask1]
    pred[mask1] = i_idle_target + (i0_mA - i_idle_target) * np.exp(-(t1 - t1[0]) / MODEL_TAU_S)

    # State at 10s
    i_at_10 = i_idle_target + (i0_mA - i_idle_target) * np.exp(-(INITIAL_IDLE_END_S - t1[0]) / MODEL_TAU_S)

    # Segment 2: UART active, 10-30s
    mask2 = (time_s >= INITIAL_IDLE_END_S) & (time_s < UART_ACTIVE_END_S)
    t2 = time_s[mask2]
    pred[mask2] = i_uart_target + (i_at_10 - i_uart_target) * np.exp(-(t2 - INITIAL_IDLE_END_S) / MODEL_TAU_S)

    # State at 30s
    i_at_30 = i_uart_target + (i_at_10 - i_uart_target) * np.exp(-(UART_ACTIVE_END_S - INITIAL_IDLE_END_S) / MODEL_TAU_S)

    # Segment 3: final idle, 30-40s
    mask3 = time_s >= UART_ACTIVE_END_S
    t3 = time_s[mask3]
    pred[mask3] = i_idle_target + (i_at_30 - i_idle_target) * np.exp(-(t3 - UART_ACTIVE_END_S) / MODEL_TAU_S)

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

    # Useful current-level comparison
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
    """
    Save measured vs predicted current plot.
    Downsample only for plotting speed.
    """

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


def process_one_file(processed_file: Path) -> dict:
    condition = processed_file.parent.name
    tx_bytes, period_ms = parse_condition(condition)

    occupancy = compute_uart_occupancy(tx_bytes, period_ms)

    print(f"\nProcessing: {processed_file}")
    print(f"Condition: {condition}")
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

    # Initial condition from first measured sample
    i0_mA = measured[0]

    pred = simulate_occupancy_ode_vectorized(time_s, occupancy, i0_mA)

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

    out_dir = RESULTS_ROOT / condition / "occupancy_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"prediction_occupancy_{processed_file.stem}.csv"

    if SAVE_PREDICTION_CSV:
        df.to_csv(out_csv, index=False)
        output_csv_str = str(out_csv)
        print(f"Saved prediction CSV: {out_csv}")
    else:
        output_csv_str = "not_saved"

    out_plot = out_dir / f"plot_occupancy_{processed_file.stem}.png"
    make_plot(
        df,
        out_plot,
        title=f"UART occupancy input validation: {condition} / {processed_file.name}"
    )

    metrics = compute_metrics(df)

    summary = {
        "condition": condition,
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
        "output_prediction_csv": output_csv_str,
        "output_plot": str(out_plot),
    }

    summary.update(metrics)

    print(f"Saved plot: {out_plot}")

    return summary


def main():
    processed_files = sorted(PROCESSED_ROOT.glob("uart_*_100ms/aligned_*.csv"))

    if not processed_files:
        raise FileNotFoundError(f"No aligned CSV files found under {PROCESSED_ROOT}")

    print(f"Found {len(processed_files)} processed CSV files.")

    summaries = []

    for processed_file in processed_files:
        summary = process_one_file(processed_file)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_out = RESULTS_ROOT / "summary_uart_occupancy_input_validation.csv"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved summary: {summary_out}")
    print("Done.")


if __name__ == "__main__":
    main()