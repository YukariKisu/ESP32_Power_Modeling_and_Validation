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

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0


# ============================================================
# CPU100-trained ODE model parameters
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_TAU_S = 0.0004895

USE_IDLE_OFFSET_CORRECTION = True

# Keep this False unless you really need huge prediction CSV files.
SAVE_PREDICTION_CSV = False

# Plot downsampling
MAX_PLOT_POINTS = 50000


# ============================================================
# Evaluation windows
# ============================================================

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


def compute_tx_duration_s(tx_bytes: int) -> float:
    """
    UART 8N1:
        1 byte = 10 transmitted bits

    tx_duration_s = tx_bytes * 10 / baud_rate
    """
    return tx_bytes * UART_BITS_PER_BYTE_8N1 / UART_BAUD_RATE


def compute_uart_occupancy(tx_bytes: int, period_ms: int) -> float:
    bytes_per_second = tx_bytes * 1000.0 / period_ms
    max_bytes_per_second = UART_BAUD_RATE / UART_BITS_PER_BYTE_8N1
    return bytes_per_second / max_bytes_per_second


def compute_pulse_input(time_s: np.ndarray, tx_duration_s: float, period_s: float) -> np.ndarray:
    """
    100 ms pulse approximation input:

        0-10 s      : u = 0
        10-30 s     : u = 1 during estimated UART TX duration
                       u = 0 for the rest of each period
        30-40 s     : u = 0
    """

    u = np.zeros(len(time_s), dtype=float)

    active_mask = (time_s >= INITIAL_IDLE_END_S) & (time_s < UART_ACTIVE_END_S)
    t_active = time_s[active_mask] - INITIAL_IDLE_END_S

    phase_in_period = np.mod(t_active, period_s)
    pulse_mask = phase_in_period < tx_duration_s

    u_active = np.zeros(len(t_active), dtype=float)
    u_active[pulse_mask] = 1.0

    u[active_mask] = u_active

    return u


def fill_ode_segment(
    pred: np.ndarray,
    time_s: np.ndarray,
    start_s: float,
    end_s: float,
    state_start_mA: float,
    target_mA: float,
) -> float:
    """
    Fill prediction for [start_s, end_s) using exact first-order ODE update:

        I(t) = I_target + (I_start - I_target) * exp(-(t - start) / tau)

    Returns the state at end_s.
    """

    i0 = np.searchsorted(time_s, start_s, side="left")
    i1 = np.searchsorted(time_s, end_s, side="left")

    if i1 > i0:
        t_seg = time_s[i0:i1]
        pred[i0:i1] = target_mA + (state_start_mA - target_mA) * np.exp(
            -(t_seg - start_s) / MODEL_TAU_S
        )

    state_end_mA = target_mA + (state_start_mA - target_mA) * math.exp(
        -(end_s - start_s) / MODEL_TAU_S
    )

    return state_end_mA


def simulate_pulse_ode_fast(
    time_s: np.ndarray,
    tx_duration_s: float,
    period_s: float,
    i0_mA: float,
) -> np.ndarray:
    """
    Fast prediction for UART 100 ms pulse approximation input.

    Input definition:
        0-10 s      : u = 0
        10-30 s     : repeated UART TX pulse
        30-40 s     : u = 0

    During pulse:
        u = 1 -> target = I_active

    Outside pulse:
        u = 0 -> target = I_idle
    """

    if MODEL_TAU_S <= 0:
        raise ValueError("MODEL_TAU_S must be positive.")

    pred = np.zeros(len(time_s), dtype=float)

    i_idle = MODEL_I_IDLE_MA
    i_active = MODEL_I_ACTIVE_MA

    # Segment 1: initial idle, 0-10 s
    state = fill_ode_segment(
        pred=pred,
        time_s=time_s,
        start_s=0.0,
        end_s=INITIAL_IDLE_END_S,
        state_start_mA=i0_mA,
        target_mA=i_idle,
    )

    # Segment 2: UART active interval, 10-30 s
    n_periods = int(round((UART_ACTIVE_END_S - INITIAL_IDLE_END_S) / period_s))

    for n in range(n_periods):
        period_start = INITIAL_IDLE_END_S + n * period_s
        pulse_start = period_start
        pulse_end = min(period_start + tx_duration_s, UART_ACTIVE_END_S)
        period_end = min(period_start + period_s, UART_ACTIVE_END_S)

        # UART TX pulse: u = 1
        if pulse_end > pulse_start:
            state = fill_ode_segment(
                pred=pred,
                time_s=time_s,
                start_s=pulse_start,
                end_s=pulse_end,
                state_start_mA=state,
                target_mA=i_active,
            )

        # Rest of period: u = 0
        if period_end > pulse_end:
            state = fill_ode_segment(
                pred=pred,
                time_s=time_s,
                start_s=pulse_end,
                end_s=period_end,
                state_start_mA=state,
                target_mA=i_idle,
            )

    # Segment 3: final idle, 30-40 s
    state = fill_ode_segment(
        pred=pred,
        time_s=time_s,
        start_s=UART_ACTIVE_END_S,
        end_s=FINAL_END_S + 1e-9,
        state_start_mA=state,
        target_mA=i_idle,
    )

    return pred


def compute_metrics(time_s: np.ndarray, measured: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - measured
    metrics = {}

    for name, (t0, t1) in EVAL_WINDOWS.items():
        mask = (time_s >= t0) & (time_s < t1)

        if not np.any(mask):
            continue

        err = error[mask]
        meas = measured[mask]
        pred = predicted[mask]

        metrics[f"{name}_mae_mA"] = float(np.mean(np.abs(err)))
        metrics[f"{name}_rmse_mA"] = float(math.sqrt(np.mean(err ** 2)))
        metrics[f"{name}_mean_measured_mA"] = float(np.mean(meas))
        metrics[f"{name}_mean_predicted_mA"] = float(np.mean(pred))
        metrics[f"{name}_mean_error_mA"] = float(np.mean(err))

    initial_mask = (time_s >= 2.0) & (time_s < 9.0)
    active_mask = (time_s >= 12.0) & (time_s < 29.0)
    final_mask = (time_s >= 32.0) & (time_s < 39.0)

    if np.any(initial_mask) and np.any(active_mask):
        initial_mean = float(np.mean(measured[initial_mask]))
        active_mean = float(np.mean(measured[active_mask]))
        pred_initial_mean = float(np.mean(predicted[initial_mask]))
        pred_active_mean = float(np.mean(predicted[active_mask]))

        metrics["measured_active_minus_initial_idle_mA"] = active_mean - initial_mean
        metrics["predicted_active_minus_initial_idle_mA"] = pred_active_mean - pred_initial_mean

    if np.any(final_mask) and np.any(active_mask):
        final_mean = float(np.mean(measured[final_mask]))
        active_mean = float(np.mean(measured[active_mask]))
        pred_final_mean = float(np.mean(predicted[final_mask]))
        pred_active_mean = float(np.mean(predicted[active_mask]))

        metrics["measured_active_minus_final_idle_mA"] = active_mean - final_mean
        metrics["predicted_active_minus_final_idle_mA"] = pred_active_mean - pred_final_mean

    return metrics


def make_plot(
    time_s: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
    tx_duration_s: float,
    period_s: float,
    out_plot: Path,
    title: str,
):
    """
    Save measured vs predicted current plot.
    Downsample only for plotting speed.
    """

    n = len(time_s)

    if n > MAX_PLOT_POINTS:
        step = int(np.ceil(n / MAX_PLOT_POINTS))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    t_plot = time_s[idx]
    measured_plot = measured[idx]
    predicted_plot = predicted[idx]
    u_plot = compute_pulse_input(t_plot, tx_duration_s, period_s)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        t_plot,
        measured_plot,
        linewidth=0.8,
        label="measured current"
    )

    ax1.plot(
        t_plot,
        predicted_plot,
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
        t_plot,
        u_plot,
        linewidth=0.8,
        linestyle="--",
        label="u_uart_pulse"
    )
    ax2.set_ylabel("UART pulse input")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)


def maybe_save_prediction_csv(
    time_s: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
    tx_duration_s: float,
    period_s: float,
    out_csv: Path,
) -> str:
    if not SAVE_PREDICTION_CSV:
        return "not_saved"

    u = compute_pulse_input(time_s, tx_duration_s, period_s)
    error = predicted - measured

    out_df = pd.DataFrame({
        "time_s": time_s,
        "current_mA": measured,
        "u_uart_pulse": u,
        "current_pred_mA": predicted,
        "error_mA": error,
    })

    out_df.to_csv(out_csv, index=False)
    return str(out_csv)


def process_one_file(processed_file: Path) -> dict:
    condition = processed_file.parent.name
    tx_bytes, period_ms = parse_condition(condition)

    period_s = period_ms / 1000.0
    tx_duration_s = compute_tx_duration_s(tx_bytes)
    occupancy = compute_uart_occupancy(tx_bytes, period_ms)

    print(f"\nProcessing: {processed_file}")
    print(f"Condition: {condition}")
    print(f"tx_bytes = {tx_bytes}")
    print(f"period_ms = {period_ms}")
    print(f"tx_duration_s = {tx_duration_s:.9f}")
    print(f"UART occupancy = {occupancy:.6f}")

    df = pd.read_csv(
        processed_file,
        usecols=["time_s", "current_mA"]
    )

    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    measured = pd.to_numeric(df["current_mA"], errors="coerce").to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(measured)
    time_s = time_s[valid]
    measured = measured[valid]

    # Initial condition from first measured sample
    i0_mA = measured[0]

    predicted = simulate_pulse_ode_fast(
        time_s=time_s,
        tx_duration_s=tx_duration_s,
        period_s=period_s,
        i0_mA=i0_mA,
    )

    if USE_IDLE_OFFSET_CORRECTION:
        idle_mask = (time_s >= 2.0) & (time_s < 9.0)

        measured_idle_mean = float(np.mean(measured[idle_mask]))
        predicted_idle_mean = float(np.mean(predicted[idle_mask]))

        offset = measured_idle_mean - predicted_idle_mean
        predicted = predicted + offset
    else:
        offset = 0.0

    out_dir = RESULTS_ROOT / condition / "pulse_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"prediction_pulse_{processed_file.stem}.csv"
    output_csv_str = maybe_save_prediction_csv(
        time_s=time_s,
        measured=measured,
        predicted=predicted,
        tx_duration_s=tx_duration_s,
        period_s=period_s,
        out_csv=out_csv,
    )

    out_plot = out_dir / f"plot_pulse_{processed_file.stem}.png"
    make_plot(
        time_s=time_s,
        measured=measured,
        predicted=predicted,
        tx_duration_s=tx_duration_s,
        period_s=period_s,
        out_plot=out_plot,
        title=f"UART 100 ms pulse input validation: {condition} / {processed_file.name}"
    )

    metrics = compute_metrics(time_s, measured, predicted)

    summary = {
        "condition": condition,
        "source_file": str(processed_file),
        "tx_bytes": tx_bytes,
        "period_ms": period_ms,
        "uart_baud_rate": UART_BAUD_RATE,
        "uart_format": "8N1",
        "tx_duration_s": tx_duration_s,
        "tx_duration_ms": tx_duration_s * 1000.0,
        "uart_occupancy": occupancy,
        "model_i_idle_mA": MODEL_I_IDLE_MA,
        "model_i_active_mA": MODEL_I_ACTIVE_MA,
        "model_tau_s": MODEL_TAU_S,
        "use_idle_offset_correction": USE_IDLE_OFFSET_CORRECTION,
        "prediction_offset_mA": offset,
        "output_prediction_csv": output_csv_str,
        "output_plot": str(out_plot),
    }

    summary.update(metrics)

    print(f"Saved plot: {out_plot}")

    if SAVE_PREDICTION_CSV:
        print(f"Saved prediction CSV: {out_csv}")

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

    summary_out = RESULTS_ROOT / "summary_uart_pulse_input_validation.csv"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved summary: {summary_out}")
    print("Done.")


if __name__ == "__main__":
    main()