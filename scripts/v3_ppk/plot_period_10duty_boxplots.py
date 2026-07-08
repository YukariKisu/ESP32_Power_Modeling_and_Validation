import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

CONDITIONS = [
    {
        "label": "100 ms / 10%",
        "name": "period_100ms_10duty",
        "processed_dir": "data/processed/v3_ppk/per_100ms_10duty",
    },
    {
        "label": "500 ms / 10%",
        "name": "period_500ms_10duty",
        "processed_dir": "data/processed/v3_ppk/per_500ms_10duty",
    },
    {
        "label": "1 s / 10%",
        "name": "period_1s_10duty",
        "processed_dir": "data/processed/v3_ppk/per_1s_10duty",
    },
]

BASE_OUT_DIR = "data/processed/v3_ppk/period_10duty_boxplots"

os.makedirs(BASE_OUT_DIR, exist_ok=True)

# Use the periodic steady-state window, same as validation summary.
EVAL_WINDOW_S = (0.010, 10.0)

# Current range filter to avoid accidental power-on spikes or corrupted samples.
# Normal ESP32 current is around 45-70 mA here.
CURRENT_MIN_MA = 30.0
CURRENT_MAX_MA = 100.0

# For plot visibility only.
ERROR_YLIM_MA = (-5, 5)
CURRENT_YLIM_MA = (40, 75)


# ============================================================
# Load distribution data
# ============================================================

def load_condition_data(condition):
    processed_dir = condition["processed_dir"]

    # Use firmware-defined debug files first.
    patterns = [
        os.path.join(processed_dir, "*_firmware_binary_input_debug_waveform.csv"),
        os.path.join(processed_dir, "*_ppk_estimated_binary_input_debug_waveform.csv"),
    ]

    files = []

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            print(f"Using debug waveform files: {pattern}")
            break

    if not files:
        raise FileNotFoundError(
            "No debug waveform files found. Tried:\n"
            + "\n".join(patterns)
        )

    rows = []

    for path in files:
        base = os.path.basename(path)

        run_name = base.replace(
            "_firmware_binary_input_debug_waveform.csv",
            "",
        ).replace(
            "_ppk_estimated_binary_input_debug_waveform.csv",
            "",
        )

        df = pd.read_csv(path)

        time_col = "time_from_first_active_start_s"
        if time_col not in df.columns:
            time_col = "time_from_active_start_s"

        required_cols = [
            time_col,
            "measured_current_mA",
            "predicted_current_mA",
            "prediction_error_mA",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {path}")

        mask = (
            (df[time_col] >= EVAL_WINDOW_S[0])
            & (df[time_col] <= EVAL_WINDOW_S[1])
            & np.isfinite(df["measured_current_mA"])
            & np.isfinite(df["predicted_current_mA"])
            & np.isfinite(df["prediction_error_mA"])
            & (df["measured_current_mA"] >= CURRENT_MIN_MA)
            & (df["measured_current_mA"] <= CURRENT_MAX_MA)
        )

        selected = df.loc[mask, [
            time_col,
            "measured_current_mA",
            "predicted_current_mA",
            "prediction_error_mA",
        ]].copy()

        selected = selected.rename(columns={time_col: "time_s"})
        selected["condition"] = condition["label"]
        selected["condition_name"] = condition["name"]
        selected["run"] = run_name

        # Add absolute error too.
        selected["abs_prediction_error_mA"] = selected["prediction_error_mA"].abs()

        rows.append(selected)

    return pd.concat(rows, ignore_index=True)


all_data = []

for condition in CONDITIONS:
    print(f"Loading {condition['label']}")
    condition_df = load_condition_data(condition)
    all_data.append(condition_df)

distribution_df = pd.concat(all_data, ignore_index=True)

out_distribution_csv = os.path.join(
    BASE_OUT_DIR,
    "period_10duty_firmware_binary_distribution_data.csv",
)
distribution_df.to_csv(out_distribution_csv, index=False)

print("Saved distribution data:")
print(out_distribution_csv)


# ============================================================
# Summary statistics
# ============================================================

summary_rows = []

for label, group in distribution_df.groupby("condition", sort=False):
    error = group["prediction_error_mA"]
    abs_error = group["abs_prediction_error_mA"]
    current = group["measured_current_mA"]

    summary_rows.append({
        "condition": label,
        "n_samples": len(group),

        "current_median_mA": current.median(),
        "current_q1_mA": current.quantile(0.25),
        "current_q3_mA": current.quantile(0.75),
        "current_iqr_mA": current.quantile(0.75) - current.quantile(0.25),

        "error_median_mA": error.median(),
        "error_q1_mA": error.quantile(0.25),
        "error_q3_mA": error.quantile(0.75),
        "error_iqr_mA": error.quantile(0.75) - error.quantile(0.25),

        "abs_error_median_mA": abs_error.median(),
        "abs_error_q1_mA": abs_error.quantile(0.25),
        "abs_error_q3_mA": abs_error.quantile(0.75),
        "abs_error_iqr_mA": abs_error.quantile(0.75) - abs_error.quantile(0.25),
    })

summary_df = pd.DataFrame(summary_rows)

out_summary_csv = os.path.join(
    BASE_OUT_DIR,
    "period_10duty_firmware_binary_distribution_summary.csv",
)
summary_df.to_csv(out_summary_csv, index=False)

print("\nDistribution summary:")
print(summary_df)

print("\nSaved summary:")
print(out_summary_csv)


# ============================================================
# Helper: boxplot
# ============================================================

def save_boxplot(data_by_condition, labels, ylabel, title, out_path, ylim=None):
    plt.figure(figsize=(9, 6))

    plt.boxplot(
        data_by_condition,
        labels=labels,
        showfliers=False,
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved plot:")
    print(out_path)


labels = [condition["label"] for condition in CONDITIONS]

# Keep order fixed
groups = [
    distribution_df.loc[
        distribution_df["condition"] == label
    ]
    for label in labels
]


# ============================================================
# Boxplot 1: prediction error distribution
# ============================================================

error_data = [
    group["prediction_error_mA"].dropna().to_numpy()
    for group in groups
]

save_boxplot(
    data_by_condition=error_data,
    labels=labels,
    ylabel="Prediction error [mA]",
    title="Prediction error distribution\nFirmware-defined binary input, 10% duty",
    out_path=os.path.join(
        BASE_OUT_DIR,
        "period_10duty_prediction_error_boxplot.png",
    ),
    ylim=ERROR_YLIM_MA,
)


# ============================================================
# Boxplot 2: absolute prediction error distribution
# ============================================================

abs_error_data = [
    group["abs_prediction_error_mA"].dropna().to_numpy()
    for group in groups
]

save_boxplot(
    data_by_condition=abs_error_data,
    labels=labels,
    ylabel="Absolute prediction error [mA]",
    title="Absolute prediction error distribution\nFirmware-defined binary input, 10% duty",
    out_path=os.path.join(
        BASE_OUT_DIR,
        "period_10duty_absolute_prediction_error_boxplot.png",
    ),
    ylim=(0, 5),
)


# ============================================================
# Boxplot 3: measured current distribution
# ============================================================

current_data = [
    group["measured_current_mA"].dropna().to_numpy()
    for group in groups
]

save_boxplot(
    data_by_condition=current_data,
    labels=labels,
    ylabel="Measured current [mA]",
    title="Measured current distribution\n10% duty periodic workload",
    out_path=os.path.join(
        BASE_OUT_DIR,
        "period_10duty_measured_current_boxplot.png",
    ),
    ylim=CURRENT_YLIM_MA,
)