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

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0

DECOMPOSITION_WINDOW_START_S = 20.0
DECOMPOSITION_WINDOW_END_S = 30.0


# ============================================================
# CPU100-trained ODE model parameters
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_TAU_S = 0.0004895

USE_IDLE_OFFSET_CORRECTION = True

# Keep this False unless you really need huge prediction CSV files.
SAVE_PREDICTION_CSV = False

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


def mean_current_window_from_df(df: pd.DataFrame, t0: float, t1: float) -> float:
    mask = (df["time_s"] >= t0) & (df["time_s"] < t1)

    if not mask.any():
        raise ValueError(f"No samples found in window {t0}-{t1}s")

    return df.loc[mask, "current_mA"].mean()


def mean_current_window_from_arrays(
    time_s: np.ndarray,
    current_mA: np.ndarray,
    t0: float,
    t1: float,
) -> float:
    mask = (time_s >= t0) & (time_s < t1)

    if not np.any(mask):
        raise ValueError(f"No samples found in window {t0}-{t1}s")

    return float(np.mean(current_mA[mask]))


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
        mean_mA = mean_current_window_from_df(
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


def compute_tx_duration_s(tx_bytes: int) -> float:
    return tx_bytes * UART_BITS_PER_BYTE_8N1 / UART_BAUD_RATE


def compute_uart_occupancy(tx_bytes: int, period_ms: int) -> float:
    bytes_per_second = tx_bytes * 1000.0 / period_ms
    max_bytes_per_second = UART_BAUD_RATE / UART_BITS_PER_BYTE_8N1
    return bytes_per_second / max_bytes_per_second


def compute_pulse_input(time_s: np.ndarray, tx_duration_s: float, period_s: float) -> np.ndarray:
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
    if MODEL_TAU_S <= 0:
        raise ValueError("MODEL_TAU_S must be positive.")

    pred = np.zeros(len(time_s), dtype=float)

    i_idle = MODEL_I_IDLE_MA
    i_active = MODEL_I_ACTIVE_MA

    state = fill_ode_segment(
        pred=pred,
        time_s=time_s,
        start_s=0.0,
        end_s=INITIAL_IDLE_END_S,
        state_start_mA=i0_mA,
        target_mA=i_idle,
    )

    n_periods = int(round((UART_ACTIVE_END_S - INITIAL_IDLE_END_S) / period_s))

    for n in range(n_periods):
        period_start = INITIAL_IDLE_END_S + n * period_s
        pulse_start = period_start
        pulse_end = min(period_start + tx_duration_s, UART_ACTIVE_END_S)
        period_end = min(period_start + period_s, UART_ACTIVE_END_S)

        if pulse_end > pulse_start:
            state = fill_ode_segment(
                pred=pred,
                time_s=time_s,
                start_s=pulse_start,
                end_s=pulse_end,
                state_start_mA=state,
                target_mA=i_active,
            )

        if period_end > pulse_end:
            state = fill_ode_segment(
                pred=pred,
                time_s=time_s,
                start_s=pulse_end,
                end_s=period_end,
                state_start_mA=state,
                target_mA=i_idle,
            )

    fill_ode_segment(
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


def process_one_file(
    processed_file: Path,
    d1_means_by_run: dict,
    d3_means_by_run: dict,
) -> dict:
    condition = processed_file.parent.name
    tx_bytes, period_ms = parse_condition(condition)
    run_index = parse_run_index(processed_file.name)

    period_s = period_ms / 1000.0
    tx_duration_s = compute_tx_duration_s(tx_bytes)
    occupancy = compute_uart_occupancy(tx_bytes, period_ms)

    print(f"\nProcessing: {processed_file}")
    print(f"Condition: {condition}")
    print(f"run_index = {run_index}")
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

    d4_active_mean_mA = mean_current_window_from_arrays(
        time_s,
        measured,
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
        "run_index": run_index,
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

        "decomposition_window_start_s": DECOMPOSITION_WINDOW_START_S,
        "decomposition_window_end_s": DECOMPOSITION_WINDOW_END_S,
        "D1_idle_baseline_mA": d1_idle_mA,
        "D3_uart_enabled_idle_mA": d3_enabled_mA,
        "D4_uart_tx_active_mA": d4_active_mean_mA,
        "D3_minus_D1_uart_enable_overhead_mA": uart_enable_overhead_mA,
        "D4_minus_D3_tx_activity_overhead_mA": tx_activity_overhead_mA,
        "D4_minus_D1_total_uart_difference_mA": total_uart_difference_mA,

        "output_prediction_csv": output_csv_str,
        "output_plot": str(out_plot),
    }

    summary.update(metrics)

    print(f"D1 idle baseline = {d1_idle_mA:.6f} mA")
    print(f"D3 UART enabled idle = {d3_enabled_mA:.6f} mA")
    print(f"D4 UART TX active = {d4_active_mean_mA:.6f} mA")
    print(f"D3-D1 enable overhead = {uart_enable_overhead_mA:.6f} mA")
    print(f"D4-D3 TX activity overhead = {tx_activity_overhead_mA:.6f} mA")
    print(f"D4-D1 total UART difference = {total_uart_difference_mA:.6f} mA")

    print(f"Saved plot: {out_plot}")

    if SAVE_PREDICTION_CSV:
        print(f"Saved prediction CSV: {out_csv}")

    return summary


def save_condition_decomposition_summary(summary_df: pd.DataFrame):
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

    out = RESULTS_ROOT / "summary_uart_pulse_baseline_decomposition.csv"
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
            d3_means_by_run,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_out = RESULTS_ROOT / "summary_uart_pulse_input_validation.csv"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    save_condition_decomposition_summary(summary_df)

    print(f"\nSaved summary: {summary_out}")
    print("Done.")


if __name__ == "__main__":
    main()