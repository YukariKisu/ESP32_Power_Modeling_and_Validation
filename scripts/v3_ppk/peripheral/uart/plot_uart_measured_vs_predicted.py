from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

# Choose one:
CONDITION = "uart_256B_100ms"
RUN_INDEX = 5

# ------------------------------------------------------------
# Raw-data directories
# ------------------------------------------------------------
#
# You wrote:
# data/raw/v3_ppk/peripheral/uart_tx_only_uart_64B_100ms
# data/raw/v3_ppk/peripheral/uart_tx_only_uart_256B_100ms
# data/raw/v3_ppk/peripheral/uart_tx_only_uart_512B_100ms
#
# If your actual folders instead have:
# uart_tx_only/uart_64B_100ms
# just change the dictionary below.

RAW_DIRS = {
    "uart_64B_100ms": Path(
        "data/raw/v3_ppk/peripheral/uart_tx_only/uart_64B_100ms"
    ),
    "uart_256B_100ms": Path(
        "data/raw/v3_ppk/peripheral/uart_tx_only/uart_256B_100ms"
    ),
    "uart_512B_100ms": Path(
        "data/raw/v3_ppk/peripheral/uart_tx_only/uart_512B_100ms"
    ),
}


OUT_DIR = Path(
    "results/v3_ppk/peripheral/uart_tx_only/"
    "thesis_measured_vs_predicted"
)

# Save the numerical plotting data too
SAVE_CSV = True


# ============================================================
# Experiment timing
# ============================================================

# Relative to detected CPU sync rising edge:
#
# 0–1 s   : CPU sync pulse
# 1–6 s   : recovery idle
# 6–16 s  : initial idle
# 16–36 s : UART active
# 36–46 s : final idle
#
# We crop 6–46 s after sync and redefine that interval as 0–40 s.

CROP_START_AFTER_SYNC_S = 6.0
CROP_END_AFTER_SYNC_S = 46.0

INITIAL_IDLE_END_S = 10.0
UART_ACTIVE_START_S = 10.0
UART_ACTIVE_END_S = 30.0
FINAL_END_S = 40.0


# ============================================================
# CPU-trained first-order model
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_DELTA_I_MA = MODEL_I_ACTIVE_MA - MODEL_I_IDLE_MA

MODEL_TAU_S = 0.000490


# ============================================================
# UART settings
# ============================================================

UART_BAUD_RATE = 115200
UART_BITS_PER_BYTE_8N1 = 10


# ============================================================
# Idle alignment
# ============================================================

USE_IDLE_OFFSET_CORRECTION = True

# Stable part of initial idle after cropping
IDLE_OFFSET_WINDOW = (2.0, 9.0)


# ============================================================
# Sync-pulse detection
# ============================================================

MAX_DETECTION_POINTS = 20000

SMOOTH_WINDOW_POINTS = 101
THRESHOLD_STD_MULTIPLIER = 6.0
IGNORE_EARLY_SECONDS = 2.0
MIN_SYNC_DURATION_S = 0.2


# ============================================================
# Plot settings
# ============================================================

MAX_PLOT_POINTS = 70000

PLOT_XLIM = (0.0, 40.0)

# Let matplotlib determine the y range automatically.
# If you later want a fixed range, e.g.:
# PLOT_YLIM = (42, 70)
PLOT_YLIM = None


# ============================================================
# Helpers
# ============================================================

def natural_key(path: Path):
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else 0


def parse_condition(condition: str):
    m = re.match(r"uart_(\d+)B_(\d+)ms", condition)

    if not m:
        raise ValueError(
            f"Could not parse condition: {condition}"
        )

    tx_bytes = int(m.group(1))
    period_ms = int(m.group(2))

    return tx_bytes, period_ms


def find_raw_file(raw_dir: Path, run_index: int) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw-data directory does not exist:\n{raw_dir}"
        )

    patterns = [
        f"*run{run_index}.csv",
        f"*run{run_index:02d}.csv",
        f"*Run{run_index}.csv",
        f"*Run{run_index:02d}.csv",
    ]

    for pattern in patterns:
        matches = sorted(
            raw_dir.glob(pattern),
            key=natural_key,
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find run{run_index} in:\n{raw_dir}"
    )


def find_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "Timestamp(ms)",
        "time_s",
        "Time(s)",
        "Time (s)",
        "time",
        "Time",
        "timestamp_s",
        "Timestamp",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "time" in col.lower():
            return col

    raise ValueError(
        f"Could not find time column. "
        f"Columns: {list(df.columns)}"
    )


def find_current_column(df: pd.DataFrame) -> str:
    candidates = [
        "Current(uA)",
        "current_mA",
        "Current(mA)",
        "Current (mA)",
        "current",
        "Current",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "current" in col.lower():
            return col

    raise ValueError(
        f"Could not find current column. "
        f"Columns: {list(df.columns)}"
    )


def convert_time_to_seconds(
    values: pd.Series,
    column_name: str,
) -> np.ndarray:

    t = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    name = column_name.lower()

    if "ms" in name:
        t = t / 1000.0

    elif "us" in name or "µs" in name:
        t = t / 1_000_000.0

    return t


def convert_current_to_mA(
    values: pd.Series,
    column_name: str,
) -> np.ndarray:

    current = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    name = column_name.lower()

    if "ua" in name or "µa" in name:
        current = current / 1000.0

    elif "ma" in name:
        pass

    else:
        median_abs = np.nanmedian(np.abs(current))

        # Fallback heuristic
        if median_abs < 1.0:
            current = current * 1000.0
        elif median_abs > 1000.0:
            current = current / 1000.0

    return current


def load_raw_csv(path: Path):
    df = pd.read_csv(
        path,
        comment="#",
    )

    time_col = find_time_column(df)
    current_col = find_current_column(df)

    time_s = convert_time_to_seconds(
        df[time_col],
        time_col,
    )

    current_mA = convert_current_to_mA(
        df[current_col],
        current_col,
    )

    valid = (
        np.isfinite(time_s)
        & np.isfinite(current_mA)
    )

    time_s = time_s[valid]
    current_mA = current_mA[valid]

    # Recording begins at t = 0
    time_s = time_s - time_s[0]

    return time_s, current_mA


# ============================================================
# Sync detection
# ============================================================

def downsample_arrays(
    time_s,
    current_mA,
    max_points,
):
    if len(time_s) <= max_points:
        return time_s, current_mA

    step = int(
        np.ceil(len(time_s) / max_points)
    )

    return (
        time_s[::step],
        current_mA[::step],
    )


def detect_sync_pulse(
    time_s,
    current_mA,
):

    time_ds, current_ds = downsample_arrays(
        time_s,
        current_mA,
        MAX_DETECTION_POINTS,
    )

    smooth = (
        pd.Series(current_ds)
        .rolling(
            window=SMOOTH_WINDOW_POINTS,
            center=True,
            min_periods=max(
                5,
                SMOOTH_WINDOW_POINTS // 10,
            ),
        )
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )

    valid_mask = (
        time_ds
        > time_ds.min() + IGNORE_EARLY_SECONDS
    )

    t_valid = time_ds[valid_mask]
    y_valid = smooth[valid_mask]

    q30 = np.quantile(
        y_valid,
        0.30,
    )

    baseline_values = y_valid[
        y_valid <= q30
    ]

    baseline_mean = np.mean(
        baseline_values
    )

    baseline_std = np.std(
        baseline_values
    )

    threshold = (
        baseline_mean
        + THRESHOLD_STD_MULTIPLIER
        * baseline_std
    )

    above = y_valid > threshold

    if not np.any(above):
        raise ValueError(
            "Could not detect CPU sync pulse."
        )

    dt = np.median(
        np.diff(t_valid)
    )

    min_samples = max(
        3,
        int(
            MIN_SYNC_DURATION_S
            / dt
        ),
    )

    sustained = (
        np.convolve(
            above.astype(int),
            np.ones(
                min_samples,
                dtype=int,
            ),
            mode="same",
        )
        >= min_samples
    )

    if not np.any(sustained):
        raise ValueError(
            "A high-current region was found, "
            "but no sustained sync pulse was detected."
        )

    idx = np.argmax(sustained)

    # Walk backward toward actual rising edge
    while (
        idx > 0
        and above[idx - 1]
    ):
        idx -= 1

    return float(
        t_valid[idx]
    )


# ============================================================
# Crop / alignment
# ============================================================

def crop_validation_window(
    time_s,
    current_mA,
    sync_time_s,
):

    crop_start = (
        sync_time_s
        + CROP_START_AFTER_SYNC_S
    )

    crop_end = (
        sync_time_s
        + CROP_END_AFTER_SYNC_S
    )

    mask = (
        (time_s >= crop_start)
        & (time_s <= crop_end)
    )

    if not np.any(mask):
        raise ValueError(
            "Validation crop is empty."
        )

    t = (
        time_s[mask]
        - crop_start
    )

    current = current_mA[mask]

    return t, current


# ============================================================
# UART model inputs
# ============================================================

def compute_tx_duration_s(
    tx_bytes,
):
    return (
        tx_bytes
        * UART_BITS_PER_BYTE_8N1
        / UART_BAUD_RATE
    )


def compute_occupancy(
    tx_bytes,
    period_ms,
):

    period_s = (
        period_ms / 1000.0
    )

    tx_duration_s = (
        compute_tx_duration_s(
            tx_bytes
        )
    )

    return (
        tx_duration_s
        / period_s
    )


def build_occupancy_input(
    time_s,
    occupancy,
):

    u = np.zeros(
        len(time_s),
        dtype=float,
    )

    active_mask = (
        (time_s >= UART_ACTIVE_START_S)
        & (time_s < UART_ACTIVE_END_S)
    )

    u[active_mask] = occupancy

    return u


def build_pulse_input(
    time_s,
    tx_duration_s,
    period_s,
):

    u = np.zeros(
        len(time_s),
        dtype=float,
    )

    active_mask = (
        (time_s >= UART_ACTIVE_START_S)
        & (time_s < UART_ACTIVE_END_S)
    )

    t_active = (
        time_s[active_mask]
        - UART_ACTIVE_START_S
    )

    phase_in_period = np.mod(
        t_active,
        period_s,
    )

    pulse_mask = (
        phase_in_period
        < tx_duration_s
    )

    u_active = np.zeros(
        len(t_active),
        dtype=float,
    )

    u_active[pulse_mask] = 1.0

    u[active_mask] = u_active

    return u


# ============================================================
# First-order ODE
# ============================================================

def simulate_first_order_ode(
    time_s,
    u,
    i0_mA,
):

    pred = np.zeros(
        len(time_s),
        dtype=float,
    )

    pred[0] = i0_mA

    for k in range(
        1,
        len(time_s),
    ):

        dt = (
            time_s[k]
            - time_s[k - 1]
        )

        if dt <= 0:
            pred[k] = pred[k - 1]
            continue

        target = (
            MODEL_I_IDLE_MA
            + MODEL_DELTA_I_MA
            * u[k - 1]
        )

        alpha = math.exp(
            -dt / MODEL_TAU_S
        )

        pred[k] = (
            target
            + (
                pred[k - 1]
                - target
            )
            * alpha
        )

    return pred


# ============================================================
# Idle-offset correction
# ============================================================

def apply_idle_offset(
    time_s,
    measured,
    predicted,
):

    if not USE_IDLE_OFFSET_CORRECTION:
        return predicted, 0.0

    t0, t1 = IDLE_OFFSET_WINDOW

    idle_mask = (
        (time_s >= t0)
        & (time_s < t1)
    )

    if not np.any(idle_mask):
        raise ValueError(
            "Idle-offset window contains no samples."
        )

    measured_idle = np.mean(
        measured[idle_mask]
    )

    predicted_idle = np.mean(
        predicted[idle_mask]
    )

    offset = (
        measured_idle
        - predicted_idle
    )

    return (
        predicted + offset,
        offset,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_mae(
    time_s,
    measured,
    predicted,
    start_s,
    end_s,
):

    mask = (
        (time_s >= start_s)
        & (time_s < end_s)
    )

    return float(
        np.mean(
            np.abs(
                predicted[mask]
                - measured[mask]
            )
        )
    )


# ============================================================
# Plot
# ============================================================

def plot_thesis_figure(
    time_s,
    measured,
    occupancy_pred,
    pulse_pred,
    condition,
    run_index,
    tx_bytes,
    period_ms,
    out_png,
    out_pdf,
):

    if len(time_s) > MAX_PLOT_POINTS:
        step = int(
            np.ceil(
                len(time_s)
                / MAX_PLOT_POINTS
            )
        )

        idx = np.arange(
            0,
            len(time_s),
            step,
        )

    else:
        idx = np.arange(
            len(time_s)
        )

    t = time_s[idx]
    measured_plot = measured[idx]
    occupancy_plot = occupancy_pred[idx]
    pulse_plot = pulse_pred[idx]

    fig, ax = plt.subplots(
        figsize=(13, 6),
    )

    # Measured first, predictions on top
    ax.plot(
        t,
        measured_plot,
        linewidth=0.7,
        alpha=0.70,
        label="Measured current",
    )

    ax.plot(
        t,
        occupancy_plot,
        linewidth=1.8,
        label="CPU-derived model: occupancy input",
    )

    ax.plot(
        t,
        pulse_plot,
        linewidth=1.2,
        linestyle="--",
        label="CPU-derived model: pulse input",
    )

    # Active-phase boundaries
    ax.axvline(
        UART_ACTIVE_START_S,
        linestyle=":",
        linewidth=1.0,
    )

    ax.axvline(
        UART_ACTIVE_END_S,
        linestyle=":",
        linewidth=1.0,
    )

    # Very light active-region indication
    ax.axvspan(
        UART_ACTIVE_START_S,
        UART_ACTIVE_END_S,
        alpha=0.05,
    )

    ax.set_xlim(
        *PLOT_XLIM
    )

    if PLOT_YLIM is not None:
        ax.set_ylim(
            *PLOT_YLIM
        )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Current [mA]"
    )

    ax.set_title(
        "UART TX-only workload: measured current vs CPU-derived model"
        f"\nUART {tx_bytes} B / {period_ms} ms, representative run"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    if CONDITION not in RAW_DIRS:
        raise ValueError(
            f"Unknown condition: {CONDITION}"
        )

    tx_bytes, period_ms = (
        parse_condition(
            CONDITION
        )
    )

    period_s = (
        period_ms / 1000.0
    )

    tx_duration_s = (
        compute_tx_duration_s(
            tx_bytes
        )
    )

    occupancy = (
        compute_occupancy(
            tx_bytes,
            period_ms,
        )
    )

    raw_dir = RAW_DIRS[
        CONDITION
    ]

    raw_file = find_raw_file(
        raw_dir,
        RUN_INDEX,
    )

    print(
        f"Raw file: {raw_file}"
    )

    time_raw_s, current_raw_mA = (
        load_raw_csv(
            raw_file
        )
    )

    sync_time_s = detect_sync_pulse(
        time_raw_s,
        current_raw_mA,
    )

    print(
        f"Detected sync pulse: "
        f"{sync_time_s:.6f} s"
    )

    time_s, measured = (
        crop_validation_window(
            time_raw_s,
            current_raw_mA,
            sync_time_s,
        )
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    u_occupancy = (
        build_occupancy_input(
            time_s,
            occupancy,
        )
    )

    u_pulse = (
        build_pulse_input(
            time_s,
            tx_duration_s,
            period_s,
        )
    )

    # --------------------------------------------------------
    # Simulations
    # --------------------------------------------------------

    occupancy_pred = (
        simulate_first_order_ode(
            time_s,
            u_occupancy,
            measured[0],
        )
    )

    pulse_pred = (
        simulate_first_order_ode(
            time_s,
            u_pulse,
            measured[0],
        )
    )

    # Same measured idle reference for both
    occupancy_pred, occupancy_offset = (
        apply_idle_offset(
            time_s,
            measured,
            occupancy_pred,
        )
    )

    pulse_pred, pulse_offset = (
        apply_idle_offset(
            time_s,
            measured,
            pulse_pred,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    occupancy_active_mae = (
        calculate_mae(
            time_s,
            measured,
            occupancy_pred,
            UART_ACTIVE_START_S,
            UART_ACTIVE_END_S,
        )
    )

    pulse_active_mae = (
        calculate_mae(
            time_s,
            measured,
            pulse_pred,
            UART_ACTIVE_START_S,
            UART_ACTIVE_END_S,
        )
    )

    measured_active_mean = float(
        np.mean(
            measured[
                (time_s >= UART_ACTIVE_START_S)
                & (time_s < UART_ACTIVE_END_S)
            ]
        )
    )

    occupancy_active_mean = float(
        np.mean(
            occupancy_pred[
                (time_s >= UART_ACTIVE_START_S)
                & (time_s < UART_ACTIVE_END_S)
            ]
        )
    )

    pulse_active_mean = float(
        np.mean(
            pulse_pred[
                (time_s >= UART_ACTIVE_START_S)
                & (time_s < UART_ACTIVE_END_S)
            ]
        )
    )

    print()
    print("=== UART settings ===")
    print(
        f"TX bytes       = {tx_bytes}"
    )
    print(
        f"Period         = {period_ms} ms"
    )
    print(
        f"TX duration    = "
        f"{tx_duration_s * 1000:.4f} ms"
    )
    print(
        f"Occupancy      = "
        f"{occupancy:.6f}"
    )

    print()
    print("=== Active-phase results ===")
    print(
        f"Measured mean  = "
        f"{measured_active_mean:.4f} mA"
    )
    print(
        f"Occupancy mean = "
        f"{occupancy_active_mean:.4f} mA"
    )
    print(
        f"Pulse mean     = "
        f"{pulse_active_mean:.4f} mA"
    )
    print(
        f"Occupancy MAE  = "
        f"{occupancy_active_mae:.4f} mA"
    )
    print(
        f"Pulse MAE      = "
        f"{pulse_active_mae:.4f} mA"
    )

    print()
    print("=== Idle-offset correction ===")
    print(
        f"Occupancy offset = "
        f"{occupancy_offset:+.4f} mA"
    )
    print(
        f"Pulse offset     = "
        f"{pulse_offset:+.4f} mA"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        f"{CONDITION}_run{RUN_INDEX}_"
        "measured_vs_cpu_model_inputs"
    )

    out_png = (
        OUT_DIR
        / f"{base_name}.png"
    )

    out_pdf = (
        OUT_DIR
        / f"{base_name}.pdf"
    )

    if SAVE_CSV:
        out_csv = (
            OUT_DIR
            / f"{base_name}.csv"
        )

        output_df = pd.DataFrame({
            "time_s": time_s,
            "measured_current_mA": measured,
            "occupancy_input": u_occupancy,
            "pulse_input": u_pulse,
            "occupancy_prediction_mA": occupancy_pred,
            "pulse_prediction_mA": pulse_pred,
        })

        output_df.to_csv(
            out_csv,
            index=False,
        )

        print(
            f"\nSaved CSV: {out_csv}"
        )

    plot_thesis_figure(
        time_s=time_s,
        measured=measured,
        occupancy_pred=occupancy_pred,
        pulse_pred=pulse_pred,
        condition=CONDITION,
        run_index=RUN_INDEX,
        out_png=out_png,
        out_pdf=out_pdf,
        tx_bytes=tx_bytes,
        period_ms=period_ms,
    )

    print(
        f"Saved PNG: {out_png}"
    )
    print(
        f"Saved PDF: {out_pdf}"
    )


if __name__ == "__main__":
    main()