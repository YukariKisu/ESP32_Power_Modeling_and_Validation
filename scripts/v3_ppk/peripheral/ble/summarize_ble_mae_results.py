from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "results/v3_ppk/peripheral/ble_adv_only/mae_input_definitions/per_run_mae.csv"
)

OUTPUT_DIR = Path(
    "results/v3_ppk/peripheral/ble_adv_only/mae_input_definitions/summary_plots"
)

INTERVAL_ORDER = ["ble_adv_100ms", "ble_adv_500ms", "ble_adv_1000ms"]
INTERVAL_LABELS = {
    "ble_adv_100ms": "100 ms",
    "ble_adv_500ms": "500 ms",
    "ble_adv_1000ms": "1000 ms",
}
INPUT_ORDER = ["occupancy", "pulse"]
INPUT_LABELS = {
    "occupancy": "Occupancy",
    "pulse": "Pulse",
}


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in per_run_mae.csv: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def load_per_run() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)

    rename_map = {
    "measured_active_delta_mean_ma": "measured_delta_mean_ma",
    "predicted_active_delta_mean_ma": "predicted_delta_mean_ma",
    }
    df = df.rename(columns=rename_map)
    
    required = [
        "condition",
        "input_definition",
        "mae_ma",
        "me_ma",
        "rmse_ma",
        "measured_delta_mean_ma",
        "predicted_delta_mean_ma",
    ]
    require_columns(df, required)

    df = df[df["condition"].isin(INTERVAL_ORDER)].copy()
    df = df[df["input_definition"].isin(INPUT_ORDER)].copy()

    df["condition"] = pd.Categorical(
        df["condition"], categories=INTERVAL_ORDER, ordered=True
    )
    df["input_definition"] = pd.Categorical(
        df["input_definition"], categories=INPUT_ORDER, ordered=True
    )
    return df.sort_values(["condition", "input_definition"])


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["condition", "input_definition"], observed=True)
        .agg(
            n_runs=("mae_ma", "count"),
            mae_mean_ma=("mae_ma", "mean"),
            mae_std_ma=("mae_ma", "std"),
            me_mean_ma=("me_ma", "mean"),
            me_std_ma=("me_ma", "std"),
            rmse_mean_ma=("rmse_ma", "mean"),
            rmse_std_ma=("rmse_ma", "std"),
            measured_delta_mean_ma=("measured_delta_mean_ma", "mean"),
            measured_delta_std_ma=("measured_delta_mean_ma", "std"),
            predicted_delta_mean_ma=("predicted_delta_mean_ma", "mean"),
            predicted_delta_std_ma=("predicted_delta_mean_ma", "std"),
        )
        .reset_index()
    )

    summary["interval_ms"] = (
        summary["condition"].astype(str).str.extract(r"(\d+)ms").astype(int)
    )
    summary.insert(0, "interval_ms", summary.pop("interval_ms"))
    return summary


def add_mean_std_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    def fmt(mean_col: str, std_col: str) -> pd.Series:
        return out.apply(
            lambda r: f"{r[mean_col]:.3f} ± {r[std_col]:.3f}", axis=1
        )

    out["MAE mean ± std (mA)"] = fmt("mae_mean_ma", "mae_std_ma")
    out["ME mean ± std (mA)"] = fmt("me_mean_ma", "me_std_ma")
    out["RMSE mean ± std (mA)"] = fmt("rmse_mean_ma", "rmse_std_ma")
    out["measured ΔI mean ± std (mA)"] = fmt(
        "measured_delta_mean_ma", "measured_delta_std_ma"
    )
    out["predicted ΔI mean ± std (mA)"] = fmt(
        "predicted_delta_mean_ma", "predicted_delta_std_ma"
    )

    compact_cols = [
        "interval_ms",
        "condition",
        "input_definition",
        "n_runs",
        "MAE mean ± std (mA)",
        "ME mean ± std (mA)",
        "RMSE mean ± std (mA)",
        "measured ΔI mean ± std (mA)",
        "predicted ΔI mean ± std (mA)",
    ]
    return out[compact_cols]


def setup_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_bars(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(INTERVAL_ORDER))
    width = 0.34

    for i, input_def in enumerate(INPUT_ORDER):
        subset = (
            summary[summary["input_definition"] == input_def]
            .set_index("condition")
            .loc[INTERVAL_ORDER]
        )
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            subset[metric_mean],
            width,
            yerr=subset[metric_std],
            capsize=4,
            label=INPUT_LABELS[input_def],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([INTERVAL_LABELS[c] for c in INTERVAL_ORDER])
    ax.set_xlabel("Advertising interval")
    setup_axis(ax, title, ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mae_me(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    x = np.arange(len(INTERVAL_ORDER))
    width = 0.34

    for i, input_def in enumerate(INPUT_ORDER):
        subset = (
            summary[summary["input_definition"] == input_def]
            .set_index("condition")
            .loc[INTERVAL_ORDER]
        )
        offset = (i - 0.5) * width

        axes[0].bar(
            x + offset,
            subset["mae_mean_ma"],
            width,
            yerr=subset["mae_std_ma"],
            capsize=4,
            label=INPUT_LABELS[input_def],
        )
        axes[1].bar(
            x + offset,
            subset["me_mean_ma"],
            width,
            yerr=subset["me_std_ma"],
            capsize=4,
            label=INPUT_LABELS[input_def],
        )

    setup_axis(axes[0], "MAE mean ± std", "MAE (mA)")
    setup_axis(axes[1], "ME mean ± std", "ME: predicted - measured (mA)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([INTERVAL_LABELS[c] for c in INTERVAL_ORDER])
    axes[1].set_xlabel("Advertising interval")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(INTERVAL_ORDER))
    width = 0.25

    measured = (
        summary[summary["input_definition"] == "occupancy"]
        .set_index("condition")
        .loc[INTERVAL_ORDER]
    )
    ax.bar(
        x - width,
        measured["measured_delta_mean_ma"],
        width,
        yerr=measured["measured_delta_std_ma"],
        capsize=4,
        label="Measured ΔI",
    )

    for i, input_def in enumerate(INPUT_ORDER):
        subset = (
            summary[summary["input_definition"] == input_def]
            .set_index("condition")
            .loc[INTERVAL_ORDER]
        )
        ax.bar(
            x + i * width,
            subset["predicted_delta_mean_ma"],
            width,
            yerr=subset["predicted_delta_std_ma"],
            capsize=4,
            label=f"Predicted ΔI ({INPUT_LABELS[input_def]})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([INTERVAL_LABELS[c] for c in INTERVAL_ORDER])
    ax.set_xlabel("Advertising interval")
    setup_axis(ax, "Measured ΔI vs predicted ΔI", "ΔI (mA)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_run_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_base = np.arange(len(INTERVAL_ORDER))
    offsets = {"occupancy": -0.08, "pulse": 0.08}

    for input_def in INPUT_ORDER:
        subset = df[df["input_definition"] == input_def]
        xs = []
        ys = []
        for j, condition in enumerate(INTERVAL_ORDER):
            values = subset[subset["condition"] == condition]["mae_ma"].to_numpy()
            jitter = np.linspace(-0.025, 0.025, len(values)) if len(values) else []
            xs.extend(x_base[j] + offsets[input_def] + jitter)
            ys.extend(values)
        ax.scatter(xs, ys, s=38, alpha=0.8, label=INPUT_LABELS[input_def])

    ax.set_xticks(x_base)
    ax.set_xticklabels([INTERVAL_LABELS[c] for c in INTERVAL_ORDER])
    ax.set_xlabel("Advertising interval")
    setup_axis(ax, "Per-run MAE scatter", "MAE (mA)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_per_run()
    summary = make_summary(df)
    compact = add_mean_std_columns(summary)

    summary_path = OUTPUT_DIR / "summary_mae_me_delta_with_std.csv"
    compact_path = OUTPUT_DIR / "summary_mae_me_delta_compact.csv"
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)

    plot_mae_me(summary, OUTPUT_DIR / "mae_me_mean_std.png")
    plot_delta_comparison(summary, OUTPUT_DIR / "measured_vs_predicted_delta.png")
    plot_metric_bars(
        summary,
        "rmse_mean_ma",
        "rmse_std_ma",
        "RMSE (mA)",
        "RMSE mean ± std",
        OUTPUT_DIR / "rmse_mean_std.png",
    )
    plot_per_run_scatter(df, OUTPUT_DIR / "per_run_mae_scatter.png")

    print("Wrote:")
    print(f"  {summary_path}")
    print(f"  {compact_path}")
    print(f"  {OUTPUT_DIR / 'mae_me_mean_std.png'}")
    print(f"  {OUTPUT_DIR / 'measured_vs_predicted_delta.png'}")
    print(f"  {OUTPUT_DIR / 'rmse_mean_std.png'}")
    print(f"  {OUTPUT_DIR / 'per_run_mae_scatter.png'}")
    print()
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()