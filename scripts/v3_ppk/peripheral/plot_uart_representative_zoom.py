from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROCESSED_ROOT = Path("data/processed/v3_ppk/peripheral/uart_tx_only")
RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")
SUMMARY_FILE = RESULTS_ROOT / "summary_uart_active_peak_levels.csv"


# ============================================================
# Reference values from CPU-trained model
# ============================================================

CPU_I_IDLE_MA = 47.2668
CPU_I_ACTIVE_MA = 67.4556


# ============================================================
# Plot settings
# ============================================================

# Zoom window inside UART active phase
ZOOM_START_S = 12.0
ZOOM_END_S = 12.5

# Optional full plot
SAVE_FULL_PLOT = False

# Plot downsampling
MAX_PLOT_POINTS = 50000


# ============================================================
# Helpers
# ============================================================

def parse_run_number_from_source(source_file: str) -> int:
    m = re.search(r"run(\d+)", source_file)
    if not m:
        raise ValueError(f"Could not parse run number from source_file: {source_file}")
    return int(m.group(1))


def downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def choose_representative_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each condition, choose the run whose active_period_top5_mean_mA
    is closest to the condition mean.
    """
    selected_rows = []

    for condition, group in summary_df.groupby("condition"):
        target = group["active_period_top5_mean_mA"].mean()
        idx = (group["active_period_top5_mean_mA"] - target).abs().idxmin()
        selected_rows.append(summary_df.loc[idx])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def make_zoom_plot(
    df: pd.DataFrame,
    condition: str,
    run_label: str,
    idle_mean_mA: float,
    active_top5_mean_mA: float,
    active_max_mean_mA: float,
    out_path: Path,
):
    zoom_df = df[(df["time_s"] >= ZOOM_START_S) & (df["time_s"] <= ZOOM_END_S)].copy()

    if zoom_df.empty:
        raise ValueError(f"No data in zoom window for {condition} / {run_label}")

    zoom_df = downsample_df(zoom_df, MAX_PLOT_POINTS)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        zoom_df["time_s"],
        zoom_df["current_mA"],
        linewidth=0.8,
        color="black",
        label="measured current"
    )

    ax.axhline(CPU_I_IDLE_MA, color="gray", linestyle="--", linewidth=1.0, label="CPU I_idle")
    ax.axhline(CPU_I_ACTIVE_MA, color="red", linestyle="--", linewidth=1.0, label="CPU I_active")
    ax.axhline(idle_mean_mA, color="green", linestyle=":", linewidth=1.2, label="UART idle mean")
    ax.axhline(active_top5_mean_mA, color="orange", linestyle="-.", linewidth=1.2, label="UART top5 mean")

    ax.set_title(f"Representative UART zoom plot: {condition} / {run_label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [mA]")
    ax.set_xlim(ZOOM_START_S, ZOOM_END_S)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_full_plot(
    df: pd.DataFrame,
    condition: str,
    run_label: str,
    idle_mean_mA: float,
    active_top5_mean_mA: float,
    active_max_mean_mA: float,
    out_path: Path,
):
    plot_df = downsample_df(df, MAX_PLOT_POINTS)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        plot_df["time_s"],
        plot_df["current_mA"],
        linewidth=0.8,
        label="measured current"
    )

    ax.axvspan(0, 10, alpha=0.08, label="initial idle")
    ax.axvspan(10, 30, alpha=0.08, label="UART active")
    ax.axvspan(30, 40, alpha=0.08, label="final idle")

    ax.axhline(CPU_I_IDLE_MA, linestyle="--", linewidth=1.0, label="CPU I_idle")
    ax.axhline(CPU_I_ACTIVE_MA, linestyle="--", linewidth=1.0, label="CPU I_active")
    ax.axhline(idle_mean_mA, linestyle=":", linewidth=1.2, label="UART idle mean")
    ax.axhline(active_top5_mean_mA, linestyle="-.", linewidth=1.2, label="UART top5 mean")
    ax.axhline(active_max_mean_mA, linestyle="-.", linewidth=1.2, label="UART max mean")

    ax.set_title(f"Representative UART full plot: {condition} / {run_label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [mA]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_one_representative(row: pd.Series) -> None:
    condition = row["condition"]
    source_file = row["source_file"]
    run_number = parse_run_number_from_source(source_file)
    run_label = f"run{run_number}"

    processed_file = Path(source_file)

    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")

    df = pd.read_csv(processed_file, usecols=["time_s", "current_mA"])

    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["current_mA"] = pd.to_numeric(df["current_mA"], errors="coerce")
    df = df.dropna().copy()

    out_dir = RESULTS_ROOT / "representative_zoom_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom_out = out_dir / f"{condition}_{run_label}_zoom.png"
    make_zoom_plot(
        df=df,
        condition=condition,
        run_label=run_label,
        idle_mean_mA=float(row["idle_mean_mA"]),
        active_top5_mean_mA=float(row["active_period_top5_mean_mA"]),
        active_max_mean_mA=float(row["active_period_max_mean_mA"]),
        out_path=zoom_out,
    )

    print(f"Saved zoom plot: {zoom_out}")

    if SAVE_FULL_PLOT:
        full_out = out_dir / f"{condition}_{run_label}_full.png"
        make_full_plot(
            df=df,
            condition=condition,
            run_label=run_label,
            idle_mean_mA=float(row["idle_mean_mA"]),
            active_top5_mean_mA=float(row["active_period_top5_mean_mA"]),
            active_max_mean_mA=float(row["active_period_max_mean_mA"]),
            out_path=full_out,
        )
        print(f"Saved full plot: {full_out}")


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_FILE}")

    summary_df = pd.read_csv(SUMMARY_FILE)

    required_cols = {
        "condition",
        "source_file",
        "idle_mean_mA",
        "active_period_max_mean_mA",
        "active_period_top5_mean_mA",
    }
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary file: {missing}")

    rep_df = choose_representative_runs(summary_df)

    print("Selected representative runs:")
    print(rep_df[["condition", "source_file", "active_period_top5_mean_mA"]].to_string(index=False))

    for _, row in rep_df.iterrows():
        process_one_representative(row)

    rep_csv_out = RESULTS_ROOT / "representative_zoom_plots" / "selected_representative_runs.csv"
    rep_df.to_csv(rep_csv_out, index=False)
    print(f"\nSaved representative run list: {rep_csv_out}")
    print("Done.")


if __name__ == "__main__":
    main()