from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")

INPUT_CSV = RESULTS_ROOT / "summary_uart_segment_errors_grouped.csv"

OUTPUT_DIR = RESULTS_ROOT / "input_definition_comparison"

OUTPUT_COMPARISON_CSV = (
    OUTPUT_DIR / "summary_uart_occupancy_vs_pulse_by_segment.csv"
)


# ============================================================
# Comparison settings
# ============================================================

CONDITION_ORDER = [
    "uart_64B_100ms",
    "uart_256B_100ms",
    "uart_512B_100ms",
]

# Main comparison segments
SEGMENT_ORDER = [
    "idle_all",
    "burst",
    "inter_burst",
]

SEGMENT_LABELS = {
    "idle_all": "Idle",
    "burst": "Burst",
    "inter_burst": "Inter-burst",
}

CONDITION_LABELS = {
    "uart_64B_100ms": "64B / 100 ms",
    "uart_256B_100ms": "256B / 100 ms",
    "uart_512B_100ms": "512B / 100 ms",
}


# ============================================================
# Helpers
# ============================================================

def validate_input_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "condition",
        "input_definition",
        "segment",
        "runs",
        "MAE_mean_mA",
        "MAE_std_mA",
        "mean_error_mean_mA",
        "mean_error_std_mA",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_CSV}: {sorted(missing)}"
        )


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the grouped long-format summary into a direct
    occupancy-versus-pulse comparison table.
    """
    filtered = df[
        df["condition"].isin(CONDITION_ORDER)
        & df["segment"].isin(SEGMENT_ORDER)
        & df["input_definition"].isin(["occupancy", "pulse"])
    ].copy()

    if filtered.empty:
        raise ValueError("No matching comparison rows were found.")

    index_columns = [
        "condition",
        "segment",
    ]

    value_columns = [
        "MAE_mean_mA",
        "MAE_std_mA",
        "mean_error_mean_mA",
        "mean_error_std_mA",
    ]

    pivot = filtered.pivot(
        index=index_columns,
        columns="input_definition",
        values=value_columns,
    )

    pivot.columns = [
        f"{input_definition}_{metric}"
        for metric, input_definition in pivot.columns
    ]

    comparison = pivot.reset_index()

    required_pivot_columns = {
        "occupancy_MAE_mean_mA",
        "pulse_MAE_mean_mA",
        "occupancy_MAE_std_mA",
        "pulse_MAE_std_mA",
        "occupancy_mean_error_mean_mA",
        "pulse_mean_error_mean_mA",
        "occupancy_mean_error_std_mA",
        "pulse_mean_error_std_mA",
    }

    missing = required_pivot_columns - set(comparison.columns)

    if missing:
        raise ValueError(
            "Some occupancy/pulse combinations are missing: "
            f"{sorted(missing)}"
        )

    # Positive difference means pulse has larger error.
    comparison["pulse_minus_occupancy_MAE_mA"] = (
        comparison["pulse_MAE_mean_mA"]
        - comparison["occupancy_MAE_mean_mA"]
    )

    comparison["pulse_minus_occupancy_mean_error_mA"] = (
        comparison["pulse_mean_error_mean_mA"]
        - comparison["occupancy_mean_error_mean_mA"]
    )

    MAE_EQUAL_THRESHOLD_MA = 0.01

    mae_difference = (
        comparison["pulse_MAE_mean_mA"]
        - comparison["occupancy_MAE_mean_mA"]
    )

    comparison["better_input_by_MAE"] = np.where(
        np.abs(mae_difference) < MAE_EQUAL_THRESHOLD_MA,
        "equal",
        np.where(
            mae_difference > 0,
            "occupancy",
            "pulse",
        ),
    )

    comparison["condition_order"] = comparison["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )

    comparison["segment_order"] = comparison["segment"].map(
        {name: index for index, name in enumerate(SEGMENT_ORDER)}
    )

    comparison = (
        comparison
        .sort_values(["condition_order", "segment_order"])
        .drop(columns=["condition_order", "segment_order"])
        .reset_index(drop=True)
    )

    return comparison


def add_bar_labels(
    ax: plt.Axes,
    bars,
    decimals: int = 2,
) -> None:
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_metric_for_condition(
    condition_df: pd.DataFrame,
    condition: str,
    metric_name: str,
    occupancy_column: str,
    pulse_column: str,
    occupancy_std_column: str,
    pulse_std_column: str,
    ylabel: str,
    output_name: str,
    draw_zero_line: bool = False,
) -> None:
    condition_df = condition_df.copy()

    condition_df["segment_order"] = condition_df["segment"].map(
        {name: index for index, name in enumerate(SEGMENT_ORDER)}
    )

    condition_df = (
        condition_df
        .sort_values("segment_order")
        .reset_index(drop=True)
    )

    segment_labels = [
        SEGMENT_LABELS.get(segment, segment)
        for segment in condition_df["segment"]
    ]

    occupancy_values = condition_df[occupancy_column].to_numpy()
    pulse_values = condition_df[pulse_column].to_numpy()

    occupancy_std = condition_df[occupancy_std_column].to_numpy()
    pulse_std = condition_df[pulse_std_column].to_numpy()

    x = np.arange(len(segment_labels))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    occupancy_bars = ax.bar(
        x - bar_width / 2,
        occupancy_values,
        bar_width,
        yerr=occupancy_std,
        capsize=4,
        label="Occupancy input",
        hatch="//",
    )

    pulse_bars = ax.bar(
        x + bar_width / 2,
        pulse_values,
        bar_width,
        yerr=pulse_std,
        capsize=4,
        label="Pulse input",
        hatch="..",
    )

    if draw_zero_line:
        ax.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
        )

    add_bar_labels(ax, occupancy_bars)
    add_bar_labels(ax, pulse_bars)

    condition_label = CONDITION_LABELS.get(condition, condition)

    ax.set_title(
        f"{metric_name}: Occupancy vs Pulse\n"
        f"UART {condition_label}"
    )

    ax.set_xlabel("Segment")
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {output_path}")


def print_compact_summary(comparison: pd.DataFrame) -> None:
    display_columns = [
        "condition",
        "segment",
        "occupancy_MAE_mean_mA",
        "pulse_MAE_mean_mA",
        "better_input_by_MAE",
        "occupancy_mean_error_mean_mA",
        "pulse_mean_error_mean_mA",
    ]

    print("\nOccupancy vs pulse comparison:")
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_df = pd.read_csv(INPUT_CSV)

    validate_input_columns(grouped_df)

    comparison = create_comparison_table(grouped_df)

    comparison.to_csv(
        OUTPUT_COMPARISON_CSV,
        index=False,
    )

    print(f"Saved comparison CSV: {OUTPUT_COMPARISON_CSV}")

    for condition in CONDITION_ORDER:
        condition_df = comparison[
            comparison["condition"] == condition
        ].copy()

        if condition_df.empty:
            print(f"Skipping missing condition: {condition}")
            continue

        short_label = condition.replace("uart_", "")

        # ----------------------------------------------------
        # MAE comparison
        # ----------------------------------------------------

        plot_metric_for_condition(
            condition_df=condition_df,
            condition=condition,
            metric_name="Segment MAE",
            occupancy_column="occupancy_MAE_mean_mA",
            pulse_column="pulse_MAE_mean_mA",
            occupancy_std_column="occupancy_MAE_std_mA",
            pulse_std_column="pulse_MAE_std_mA",
            ylabel="MAE [mA]",
            output_name=f"{short_label}_occupancy_vs_pulse_MAE.png",
        )

        # ----------------------------------------------------
        # Mean signed error comparison
        # ----------------------------------------------------

        plot_metric_for_condition(
            condition_df=condition_df,
            condition=condition,
            metric_name="Segment Mean Signed Error",
            occupancy_column="occupancy_mean_error_mean_mA",
            pulse_column="pulse_mean_error_mean_mA",
            occupancy_std_column="occupancy_mean_error_std_mA",
            pulse_std_column="pulse_mean_error_std_mA",
            ylabel="Mean error: prediction - measurement [mA]",
            output_name=(
                f"{short_label}_occupancy_vs_pulse_mean_error.png"
            ),
            draw_zero_line=True,
        )

    print_compact_summary(comparison)

    print("\nDone.")


if __name__ == "__main__":
    main()