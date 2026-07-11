from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_ROOT = Path("results/v3_ppk/peripheral/uart_tx_only")

INPUT_CSV = RESULTS_ROOT / "summary_uart_segment_errors.csv"

OUTPUT_DIR = RESULTS_ROOT / "error_distribution_boxplots"


# ============================================================
# Plot settings
# ============================================================

CONDITION_ORDER = [
    "uart_64B_100ms",
    "uart_256B_100ms",
    "uart_512B_100ms",
]

CONDITION_LABELS = {
    "uart_64B_100ms": "UART 64B / 100 ms",
    "uart_256B_100ms": "UART 256B / 100 ms",
    "uart_512B_100ms": "UART 512B / 100 ms",
}

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

INPUT_ORDER = [
    "occupancy",
    "pulse",
]

INPUT_LABELS = {
    "occupancy": "Occupancy",
    "pulse": "Pulse",
}


# ============================================================
# Helpers
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "condition",
        "run",
        "input_definition",
        "segment",
        "median_error_mA",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_CSV}: {sorted(missing)}"
        )


def create_boxplot_for_condition(
    df: pd.DataFrame,
    condition: str,
) -> None:
    condition_df = df[
        (df["condition"] == condition)
        & (df["segment"].isin(SEGMENT_ORDER))
        & (df["input_definition"].isin(INPUT_ORDER))
    ].copy()

    if condition_df.empty:
        print(f"Skipping missing condition: {condition}")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    positions = []
    boxplot_data = []
    tick_positions = []

    group_spacing = 3.0
    box_offset = 0.38

    for segment_index, segment in enumerate(SEGMENT_ORDER):
        group_center = segment_index * group_spacing

        tick_positions.append(group_center)

        for input_index, input_definition in enumerate(INPUT_ORDER):
            values = condition_df[
                (condition_df["segment"] == segment)
                & (
                    condition_df["input_definition"]
                    == input_definition
                )
            ]["median_error_mA"].dropna().to_numpy()

            if len(values) == 0:
                continue

            position = (
                group_center
                + (-box_offset if input_index == 0 else box_offset)
            )

            positions.append(position)
            boxplot_data.append(values)

    boxplot = ax.boxplot(
        boxplot_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
    )

    # Occupancy / Pulse
    for i, box in enumerate(boxplot["boxes"]):
        if i % 2 == 0:
            box.set_facecolor("lightgray")   # Occupancy
        else:
            box.set_facecolor("white")       # Pulse

    # Add labels manually above each input group
    for segment_index, segment in enumerate(SEGMENT_ORDER):
        group_center = segment_index * group_spacing

        ax.text(
            group_center - box_offset,
            ax.get_ylim()[0],
            "Occ.",
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.text(
            group_center + box_offset,
            ax.get_ylim()[0],
            "Pulse",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([
        SEGMENT_LABELS[segment]
        for segment in SEGMENT_ORDER
    ])

    ax.set_xlabel("Segment")
    ax.set_ylabel("Run-level median error [mA]")

    ax.set_title(
        f"{CONDITION_LABELS.get(condition, condition)}\n"
        "Segment Error Distribution: Occupancy vs Pulse"
    )

    legend_handles = [
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            label="Occupancy input",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Pulse input",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=1.5,
            linestyle="--",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            label="Outlier",
        ),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=1.5,
            linestyle="--",
            label="Zero error",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{condition}_segment_error_boxplot.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_CSV)

    validate_columns(df)

    for condition in CONDITION_ORDER:
        create_boxplot_for_condition(
            df=df,
            condition=condition,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()