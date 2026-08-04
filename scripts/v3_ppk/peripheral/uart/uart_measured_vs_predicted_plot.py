from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

RAW_ROOT = Path("data/raw/v3_ppk/peripheral/uart_tx_only")
OUT_DIR = Path("results/v3_ppk/peripheral/uart_tx_only/measured_vs_predicted_plots")

CONDITION = "uart_64B_100ms"   # uart_64B_100ms, uart_256B_100ms, uart_512B_100ms
RUN_INDEX = 5

# "occupancy" or "pulse"
INPUT_DEFINITION = ["occupancy", "pulse"]

# Raw firmware timing:
# 0-3 s   settling idle
# 3-4 s   CPU sync pulse
# 4-9 s   recovery idle
# 9-19 s  initial idle
# 19-39 s UART active
# 39-49 s final idle
#
# Therefore, subtract 9 s so the validation plot becomes:
# 0-10 s  initial idle
# 10-30 s UART active
# 30-40 s final idle
VALIDATION_START_OFFSET_S = 9.0

PLOT_WINDOW_S = (0.0, 40.0)


# ============================================================
# CPU-trained ODE model parameters
# ============================================================

MODEL_I_IDLE_MA = 47.2668
MODEL_I_ACTIVE_MA = 67.4556
MODEL_TAU_S = 0.00049


# ============================================================
# UART / experiment settings
# ============================================================

UART_BAUD_RATE = 115200
UART_BITS_PER_BYTE_8N1 = 10

ACTIVE_START_S = 10.0
ACTIVE_END_S = 30.0
FINAL_END_S = 40.0

USE_IDLE_OFFSET_CORRECTION = True
IDLE_OFFSET_WINDOW_S = (2.0, 9.0)

AUTO_FIX_NEGATIVE_CURRENT = True


# ============================================================
# Helpers
# ============================================================

def parse_condition(condition: str) -> tuple[int, int]:
    m = re.match(r"uart_(\d+)B_(\d+)ms", condition)
    if not m:
        raise ValueError(f"Could not parse condition: {condition}")
    return int(m.group(1)), int(m.group(2))


def find_raw_file() -> Path:
    condition_dir = RAW_ROOT / CONDITION
    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition directory not found: {condition_dir}")

    patterns = [
        f"*run{RUN_INDEX}.csv",
        f"*run{RUN_INDEX:02d}.csv",
        f"*Run{RUN_INDEX}.csv",
        f"*Run{RUN_INDEX:02d}.csv",
    ]

    for pattern in patterns:
        candidates = sorted(condition_dir.glob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No run{RUN_INDEX} CSV found in {condition_dir}. "
        "Check CONDITION and RUN_INDEX."
    )


def find_time_column(df: pd.DataFrame) -> str:
    preferred = [
        "time_s",
        "time",
        "timestamp_s",
        "timestamp",
        "Time [s]",
        "Time",
    ]
    lower_map = {c.lower().strip(): c for c in df.columns}

    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        if "time" in col.lower():
            return col

    raise ValueError(f"Could not find time column. Available columns: {list(df.columns)}")


def find_current_column(df: pd.DataFrame) -> str:
    preferred = [
        "current_mA",
        "current_ma",
        "Current [mA]",
        "Current(mA)",
        "current_uA",
        "current_ua",
        "Current [uA]",
        "Current(uA)",
        "current_A",
        "current_a",
        "Current [A]",
        "Current(A)",
        "current",
        "Current",
    ]
    lower_map = {c.lower().strip(): c for c in df.columns}

    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        if "current" in col.lower():
            return col

    raise ValueError(f"Could not find current column. Available columns: {list(df.columns)}")


def convert_current_to_mA(series: pd.Series, column_name: str) -> pd.Series:
    current = pd.to_numeric(series, errors="coerce")
    name = column_name.lower()
    median_abs = current.abs().median()

    if "ua" in name or "micro" in name:
        current = current / 1000.0
    elif re.search(r"(^|[^a-z])a([^a-z]|$)", name) and "ma" not in name:
        current = current * 1000.0
    elif "ma" in name:
        current = current
    else:
        if median_abs < 1.0:
            current = current * 1000.0       # A -> mA
        elif median_abs > 1000.0:
            current = current / 1000.0       # uA -> mA

    if AUTO_FIX_NEGATIVE_CURRENT and current.median() < -1.0:
        current = -current

    return current


def load_raw_current_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = find_time_column(df)
    current_col = find_current_column(df)

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[time_col], errors="coerce"),
        "current_mA": convert_current_to_mA(df[current_col], current_col),
    }).dropna()

    out["time_s"] = out["time_s"] - float(VALIDATION_START_OFFSET_S)
    out = out[(out["time_s"] >= 0.0) & (out["time_s"] <= FINAL_END_S)].copy()

    if out.empty:
        raise ValueError("No samples found in the 0-40 s validation window.")

    print(f"Using time column: {time_col}")
    print(f"Using current column: {current_col}")
    print(f"Measured current median after conversion: {out['current_mA'].median():.3f} mA")

    return out


def build_uart_input(
    time_s: np.ndarray,
    tx_bytes: int,
    period_ms: int,
    input_definition: str,
) -> np.ndarray:
    period_s = period_ms / 1000.0
    tx_duration_s = tx_bytes * UART_BITS_PER_BYTE_8N1 / UART_BAUD_RATE
    occupancy = tx_duration_s / period_s

    u = np.zeros(len(time_s), dtype=float)
    active_mask = (time_s >= ACTIVE_START_S) & (time_s < ACTIVE_END_S)

    if input_definition == "occupancy":
        u[active_mask] = occupancy
        return u

    if input_definition == "pulse":
        phase_t = (time_s - ACTIVE_START_S) % period_s
        tx_mask = active_mask & (phase_t < tx_duration_s)
        u[tx_mask] = 1.0
        return u

    raise ValueError("input_definition must be 'occupancy' or 'pulse'.")


def simulate_first_order_ode(time_s: np.ndarray, u: np.ndarray, i0_mA: float) -> np.ndarray:
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


def downsample_df(df: pd.DataFrame, max_points: int = 70000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def make_plot(df: pd.DataFrame, raw_file: Path, out_png: Path, out_pdf: Path) -> None:
    t0, t1 = PLOT_WINDOW_S
    plot_df = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].copy()
    plot_df = downsample_df(plot_df)

    mae_occ = float(np.mean(np.abs(plot_df["current_pred_occupancy_mA"] - plot_df["current_mA"])))
    mae_pulse = float(np.mean(np.abs(plot_df["current_pred_pulse_mA"] - plot_df["current_mA"])))

    fig, ax = plt.subplots(figsize=(13.5, 7.0))

    ax.plot(
        plot_df["time_s"],
        plot_df["current_mA"],
        color="#111111",
        linewidth=1.0,
        label="Measured current",
    )

    ax.plot(
        plot_df["time_s"],
        plot_df["current_pred_occupancy_mA"],
        color="#2563eb",
        linewidth=1.8,
        label=f"Predicted current, occupancy input, MAE = {mae_occ:.2f} mA",
    )

    ax.plot(
        plot_df["time_s"],
        plot_df["current_pred_pulse_mA"],
        color="#dc2626",
        linewidth=1.4,
        linestyle="--",
        label=f"Predicted current, pulse input, MAE = {mae_pulse:.2f} mA",
    )

    ax.axvspan(ACTIVE_START_S, ACTIVE_END_S, color="#f97316", alpha=0.08)
    ax.axvline(ACTIVE_START_S, color="#f97316", linewidth=1.0, alpha=0.45)
    ax.axvline(ACTIVE_END_S, color="#f97316", linewidth=1.0, alpha=0.45)

    ax.text(
        0.02,
        0.96,
        f"{CONDITION}, run{RUN_INDEX}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.9,
        },
    )

    ax.set_title(
        "Measured vs Predicted Current Response\n"
        "CPU-trained first-order ODE model on UART TX-only workload",
        fontsize=18,
        pad=16,
    )
    ax.set_xlabel("Time [s]", fontsize=13)
    ax.set_ylabel("Current [mA]", fontsize=13)
    ax.grid(True, alpha=0.25)

    y_values = np.concatenate([
        plot_df["current_mA"].to_numpy(),
        plot_df["current_pred_occupancy_mA"].to_numpy(),
        plot_df["current_pred_pulse_mA"].to_numpy(),
    ])
    y_values = y_values[np.isfinite(y_values)]

    y_low, y_high = np.percentile(y_values, [1, 99])
    y_margin = max((y_high - y_low) * 0.15, 0.5)
    ax.set_ylim(y_low - y_margin, y_high + y_margin)

    ax.legend(loc="best", frameon=True, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Source: {raw_file}")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


def main() -> None:
    tx_bytes, period_ms = parse_condition(CONDITION)
    raw_file = find_raw_file()
    df = load_raw_current_csv(raw_file)

    time_s = df["time_s"].to_numpy()

    for input_definition in INPUT_DEFINITION:
        u = build_uart_input(time_s, tx_bytes, period_ms, input_definition)
        pred = simulate_first_order_ode(time_s, u, MODEL_I_IDLE_MA)

        df[f"u_{input_definition}"] = u
        df[f"current_pred_{input_definition}_mA"] = pred

        if USE_IDLE_OFFSET_CORRECTION:
            idle_t0, idle_t1 = IDLE_OFFSET_WINDOW_S
            idle_mask = (df["time_s"] >= idle_t0) & (df["time_s"] < idle_t1)

            if idle_mask.any():
                offset = (
                    df.loc[idle_mask, "current_mA"].mean()
                    - df.loc[idle_mask, f"current_pred_{input_definition}_mA"].mean()
                )
                df[f"current_pred_{input_definition}_mA"] += offset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{CONDITION}_run{RUN_INDEX}_measured_vs_predicted_inputs"
    out_csv = OUT_DIR / f"{base_name}.csv"
    out_png = OUT_DIR / f"{base_name}.png"
    out_pdf = OUT_DIR / f"{base_name}.pdf"

    df.to_csv(out_csv, index=False)
    make_plot(df, raw_file, out_png, out_pdf)
    print(f"Saved plot data: {out_csv}")


if __name__ == "__main__":
    main()