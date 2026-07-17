from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")

# This file already contains the processed source_file paths
SOURCE_SUMMARY_FILE = RESULTS_ROOT / "summary_uart_active_peak_levels.csv"

OUTPUT_RUN_SUMMARY = RESULTS_ROOT / "summary_uart_segment_errors.csv"
OUTPUT_GROUPED_SUMMARY = RESULTS_ROOT / "summary_uart_segment_errors_grouped.csv"


# ============================================================
# CPU-trained first-order ODE model
# ============================================================

CPU_I_IDLE_MA = 47.2668
CPU_I_ACTIVE_MA = 67.4556
CPU_DELTA_I_MA = CPU_I_ACTIVE_MA - CPU_I_IDLE_MA

MODEL_TAU_S = 0.0004895


# ============================================================
# UART experiment definition
# ============================================================

ACTIVE_START_S = 10.0
ACTIVE_END_S = 30.0

UART_PERIOD_S = 0.100

# 115200 baud, 8N1:
# 10 bits per byte
UART_BAUD = 115200
BITS_PER_BYTE = 10

UART_BYTES = {
    "uart_64B_100ms": 64,
    "uart_256B_100ms": 256,
    "uart_512B_100ms": 512,
}


# ============================================================
# Helpers
# ============================================================

def parse_run_number(source_file: str) -> int:
    match = re.search(r"run(\d+)", source_file)

    if not match:
        raise ValueError(
            f"Could not parse run number from source file: {source_file}"
        )

    return int(match.group(1))


def calculate_tx_duration_s(data_bytes: int) -> float:
    return data_bytes * BITS_PER_BYTE / UART_BAUD


def create_occupancy_input(
    time_s: np.ndarray,
    occupancy: float,
) -> np.ndarray:
    """
    Phase-level occupancy input:

    idle phase  -> u = 0
    active phase -> u = occupancy
    """
    u = np.zeros_like(time_s, dtype=float)

    active_mask = (
        (time_s >= ACTIVE_START_S)
        & (time_s < ACTIVE_END_S)
    )

    u[active_mask] = occupancy

    return u


def create_pulse_input(
    time_s: np.ndarray,
    tx_duration_s: float,
) -> np.ndarray:
    """
    Pulse-based input:

    During the 10–30 s UART active phase,
    u = 1 during each UART TX burst,
    u = 0 during the inter-burst interval.
    """
    u = np.zeros_like(time_s, dtype=float)

    active_mask = (
        (time_s >= ACTIVE_START_S)
        & (time_s < ACTIVE_END_S)
    )

    relative_time = time_s[active_mask] - ACTIVE_START_S
    phase_in_period = np.mod(relative_time, UART_PERIOD_S)

    u[active_mask] = (
        phase_in_period < tx_duration_s
    ).astype(float)

    return u


def simulate_first_order_model(
    time_s: np.ndarray,
    input_u: np.ndarray,
) -> np.ndarray:
    """
    Exact discrete update for:

        dI/dt = (I_target - I) / tau

        I_target = I_idle + delta_I * u
    """
    prediction = np.empty_like(time_s, dtype=float)

    prediction[0] = CPU_I_IDLE_MA

    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]

        if dt <= 0:
            prediction[i] = prediction[i - 1]
            continue

        target_current = (
            CPU_I_IDLE_MA
            + CPU_DELTA_I_MA * input_u[i - 1]
        )

        alpha = np.exp(-dt / MODEL_TAU_S)

        prediction[i] = (
            target_current
            + (prediction[i - 1] - target_current) * alpha
        )

    return prediction


def create_segment_masks(
    time_s: np.ndarray,
    tx_duration_s: float,
) -> dict[str, np.ndarray]:
    """
    Create masks for:

    - initial_idle
    - burst
    - inter_burst
    - final_idle
    - idle_all
    """
    initial_idle = time_s < ACTIVE_START_S

    final_idle = time_s >= ACTIVE_END_S

    active_phase = (
        (time_s >= ACTIVE_START_S)
        & (time_s < ACTIVE_END_S)
    )

    relative_time = time_s - ACTIVE_START_S
    phase_in_period = np.mod(relative_time, UART_PERIOD_S)

    burst = (
        active_phase
        & (phase_in_period < tx_duration_s)
    )

    inter_burst = (
        active_phase
        & (phase_in_period >= tx_duration_s)
    )

    idle_all = initial_idle | final_idle

    return {
        "initial_idle": initial_idle,
        "burst": burst,
        "inter_burst": inter_burst,
        "final_idle": final_idle,
        "idle_all": idle_all,
    }


def calculate_segment_metrics(
    condition: str,
    run_number: int,
    source_file: str,
    input_definition: str,
    measured_mA: np.ndarray,
    predicted_mA: np.ndarray,
    segment_masks: dict[str, np.ndarray],
) -> list[dict]:
    """
    error = prediction - measurement

    positive mean error -> overestimation
    negative mean error -> underestimation
    """
    error_mA = predicted_mA - measured_mA

    rows = []

    for segment_name, mask in segment_masks.items():
        segment_error = error_mA[mask]

        if len(segment_error) == 0:
            continue

        rows.append({
            "condition": condition,
            "run": run_number,
            "source_file": source_file,
            "input_definition": input_definition,
            "segment": segment_name,
            "sample_count": len(segment_error),
            "MAE_mA": np.mean(np.abs(segment_error)),
            "mean_error_mA": np.mean(segment_error),
            "median_error_mA": np.median(segment_error),
            "positive_error_ratio": np.mean(segment_error > 0),
        })

    return rows


def load_processed_file(source_file: str) -> pd.DataFrame:
    path = Path(source_file)

    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    df = pd.read_csv(
        path,
        usecols=["time_s", "current_mA"],
    )

    df["time_s"] = pd.to_numeric(
        df["time_s"],
        errors="coerce",
    )

    df["current_mA"] = pd.to_numeric(
        df["current_mA"],
        errors="coerce",
    )

    df = (
        df.dropna()
        .sort_values("time_s")
        .drop_duplicates(subset="time_s")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError(f"Too few valid samples in: {path}")

    return df


# ============================================================
# Main processing
# ============================================================

def main() -> None:
    if not SOURCE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Source summary file not found: {SOURCE_SUMMARY_FILE}"
        )

    source_summary = pd.read_csv(SOURCE_SUMMARY_FILE)

    required_columns = {
        "condition",
        "source_file",
    }

    missing_columns = (
        required_columns - set(source_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in source summary: {missing_columns}"
        )

    # One source file per run
    source_rows = (
        source_summary[
            ["condition", "source_file"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    output_rows = []

    for _, source_row in source_rows.iterrows():
        condition = source_row["condition"]
        source_file = source_row["source_file"]

        if condition not in UART_BYTES:
            print(f"Skipping unknown condition: {condition}")
            continue

        data_bytes = UART_BYTES[condition]
        tx_duration_s = calculate_tx_duration_s(data_bytes)
        occupancy = tx_duration_s / UART_PERIOD_S

        run_number = parse_run_number(source_file)

        df = load_processed_file(source_file)

        time_s = df["time_s"].to_numpy(dtype=float)
        measured_mA = df["current_mA"].to_numpy(dtype=float)

        segment_masks = create_segment_masks(
            time_s=time_s,
            tx_duration_s=tx_duration_s,
        )

        # ----------------------------------------------------
        # Occupancy-based input
        # ----------------------------------------------------

        occupancy_u = create_occupancy_input(
            time_s=time_s,
            occupancy=occupancy,
        )

        occupancy_prediction = simulate_first_order_model(
            time_s=time_s,
            input_u=occupancy_u,
        )

        output_rows.extend(
            calculate_segment_metrics(
                condition=condition,
                run_number=run_number,
                source_file=source_file,
                input_definition="occupancy",
                measured_mA=measured_mA,
                predicted_mA=occupancy_prediction,
                segment_masks=segment_masks,
            )
        )

        # ----------------------------------------------------
        # Pulse-based input
        # ----------------------------------------------------

        pulse_u = create_pulse_input(
            time_s=time_s,
            tx_duration_s=tx_duration_s,
        )

        pulse_prediction = simulate_first_order_model(
            time_s=time_s,
            input_u=pulse_u,
        )

        output_rows.extend(
            calculate_segment_metrics(
                condition=condition,
                run_number=run_number,
                source_file=source_file,
                input_definition="pulse",
                measured_mA=measured_mA,
                predicted_mA=pulse_prediction,
                segment_masks=segment_masks,
            )
        )

        print(
            f"Processed {condition} run{run_number}: "
            f"tx={tx_duration_s * 1000:.3f} ms, "
            f"occupancy={occupancy:.6f}"
        )

    run_summary = pd.DataFrame(output_rows)

    run_summary = run_summary.sort_values(
        by=[
            "condition",
            "run",
            "input_definition",
            "segment",
        ]
    ).reset_index(drop=True)

    OUTPUT_RUN_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_summary.to_csv(
        OUTPUT_RUN_SUMMARY,
        index=False,
    )

    # ========================================================
    # Grouped summary across runs
    # ========================================================

    grouped_summary = (
        run_summary
        .groupby(
            [
                "condition",
                "input_definition",
                "segment",
            ],
            as_index=False,
        )
        .agg(
            runs=("run", "nunique"),

            MAE_mean_mA=("MAE_mA", "mean"),
            MAE_std_mA=("MAE_mA", "std"),

            mean_error_mean_mA=("mean_error_mA", "mean"),
            mean_error_std_mA=("mean_error_mA", "std"),

            median_error_mean_mA=("median_error_mA", "mean"),
            median_error_std_mA=("median_error_mA", "std"),

            positive_error_ratio_mean=(
                "positive_error_ratio",
                "mean",
            ),
            positive_error_ratio_std=(
                "positive_error_ratio",
                "std",
            ),
        )
    )

    grouped_summary.to_csv(
        OUTPUT_GROUPED_SUMMARY,
        index=False,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_RUN_SUMMARY}")
    print(f"  {OUTPUT_GROUPED_SUMMARY}")

    print("\nGrouped summary:")
    print(grouped_summary.to_string(index=False))


if __name__ == "__main__":
    main()