import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



DEFAULT_P_IDLE_MW = 168.956222
DEFAULT_K_MW = 51.321679166018875


POWER_COL_CANDIDATES = [
    "power_mW",
    "power_mw",
    "Power_mW",
    "power",
]

TIME_COL_CANDIDATES = [
    "timestamp_ms",
    "time_ms",
    "elapsed_ms",
    "timestamp_s",
    "time_s",
    "elapsed_s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate static-limit first ODE model for CPU-only 100% step data."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing raw measurement CSV files."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to save validation results. Default: input_dir.parent/analysis/ode1_cpu100_10ms"
    )

    parser.add_argument(
        "--p-idle",
        type=float,
        default=DEFAULT_P_IDLE_MW,
        help="Stable idle baseline power in mW."
    )

    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K_MW,
        help="Estimated gain K in mW for u=1."
    )

    parser.add_argument(
        "--active-start-s",
        type=float,
        default=10.0,
        help="Active workload start time in seconds."
    )

    parser.add_argument(
        "--active-end-s",
        type=float,
        default=30.0,
        help="Active workload end time in seconds."
    )

    parser.add_argument(
        "--stable-margin-s",
        type=float,
        default=1.0,
        help="Seconds removed around switching points for plateau-only validation."
    )

    parser.add_argument(
        "--dt-s",
        type=float,
        default=0.01,
        help="Fallback sampling interval in seconds if no time column exists."
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="CSV filename pattern. Example: run*.csv"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample prediction CSV files."
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable measured-vs-model plot output."
    )

    return parser.parse_args()


def find_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_time_seconds(df, dt_s):
    time_col = find_col(df, TIME_COL_CANDIDATES)

    if time_col is None:
        df["t_s"] = np.arange(len(df)) * dt_s
        return df

    if time_col.endswith("_ms"):
        df["t_s"] = df[time_col] / 1000.0
    else:
        df["t_s"] = df[time_col]

    df["t_s"] = df["t_s"] - df["t_s"].iloc[0]
    return df


def is_raw_run_csv(file):
    name = file.name.lower()
    excluded_keywords = ["summary", "jitter", "baseline", "validation"]

    return not any(keyword in name for keyword in excluded_keywords)


def calculate_error_metrics(error):
    return {
        "mae_mW": np.mean(np.abs(error)),
        "rmse_mW": np.sqrt(np.mean(error ** 2)),
        "mean_error_mW": np.mean(error),
        "sd_error_mW": np.std(error, ddof=1),
        "max_abs_error_mW": np.max(np.abs(error)),
        "min_error_mW": np.min(error),
        "max_error_mW": np.max(error),
    }


def add_prefixed_metrics(row, prefix, error):
    metrics = calculate_error_metrics(error)

    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value

    return row


def make_model_prediction(df, p_idle_mw, k_mw, active_start_s, active_end_s):
    df["u"] = np.where(
        (df["t_s"] >= active_start_s) & (df["t_s"] < active_end_s),
        1.0,
        0.0
    )

    df["P_model_mW"] = p_idle_mw + k_mw * df["u"]
    return df


def make_plateau_mask(df, active_start_s, active_end_s, stable_margin_s):
    idle_pre = (
        (df["t_s"] >= stable_margin_s) &
        (df["t_s"] < active_start_s - stable_margin_s)
    )

    active_stable = (
        (df["t_s"] >= active_start_s + stable_margin_s) &
        (df["t_s"] < active_end_s - stable_margin_s)
    )

    idle_post = (
        (df["t_s"] >= active_end_s + stable_margin_s) &
        (df["t_s"] <= df["t_s"].max() - stable_margin_s)
    )

    return idle_pre | active_stable | idle_post


def save_plot(df, power_col, output_path, title):
    plt.figure(figsize=(10, 4))
    plt.plot(df["t_s"], df[power_col], label="Measured power")
    plt.plot(df["t_s"], df["P_model_mW"], label="Model power")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [mW]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    input_dir = args.input_dir

    if args.output_dir is None:
        output_dir = input_dir.parent / "analysis" / "ode1_cpu100_10ms"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_dir = output_dir / "predictions"
    plot_dir = output_dir / "plots"

    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        f for f in input_dir.glob(args.pattern)
        if is_raw_run_csv(f)
    )

    if not csv_files:
        raise FileNotFoundError(f"No raw run CSV files found in: {input_dir}")

    rows = []

    for file in csv_files:
        df = pd.read_csv(file)
        df = add_time_seconds(df, args.dt_s)

        power_col = find_col(df, POWER_COL_CANDIDATES)

        if power_col is None:
            raise ValueError(
                f"No power column found in {file.name}. "
                f"Available columns: {list(df.columns)}"
            )

        df = make_model_prediction(
            df,
            p_idle_mw=args.p_idle,
            k_mw=args.k,
            active_start_s=args.active_start_s,
            active_end_s=args.active_end_s,
        )

        df["error_mW"] = df[power_col] - df["P_model_mW"]

        full_error = df["error_mW"].to_numpy()

        plateau_mask = make_plateau_mask(
            df,
            active_start_s=args.active_start_s,
            active_end_s=args.active_end_s,
            stable_margin_s=args.stable_margin_s,
        )

        plateau_error = df.loc[plateau_mask, "error_mW"].to_numpy()

        active_mask = df["u"] == 1.0
        idle_mask = df["u"] == 0.0

        active_error = df.loc[active_mask, "error_mW"].to_numpy()
        idle_error = df.loc[idle_mask, "error_mW"].to_numpy()

        row = {
            "run": file.stem,
            "source_file": file.name,
            "n_total_samples": len(df),
            "n_plateau_samples": int(plateau_mask.sum()),
            "n_active_samples": int(active_mask.sum()),
            "n_idle_samples": int(idle_mask.sum()),
            "P_idle_mW": args.p_idle,
            "K_mW": args.k,
            "P_active_model_mW": args.p_idle + args.k,
        }

        row = add_prefixed_metrics(row, "full", full_error)
        row = add_prefixed_metrics(row, "plateau_only", plateau_error)
        row = add_prefixed_metrics(row, "active_window", active_error)
        row = add_prefixed_metrics(row, "idle_window", idle_error)

        rows.append(row)

        if args.save_predictions:
            save_cols = ["t_s", power_col, "u", "P_model_mW", "error_mW"]
            prediction_path = prediction_dir / f"{file.stem}_prediction.csv"
            df[save_cols].to_csv(prediction_path, index=False)

        if not args.no_plots:
            plot_path = plot_dir / f"{file.stem}_measured_vs_model.png"
            save_plot(
                df,
                power_col=power_col,
                output_path=plot_path,
                title=f"{file.stem}: measured vs static-limit model"
            )

    per_run = pd.DataFrame(rows)

    per_run_path = output_dir / "validation_per_run_summary.csv"
    per_run.to_csv(per_run_path, index=False)

    overall = pd.DataFrame([{
        "n_runs": len(per_run),
        "P_idle_mW": args.p_idle,
        "K_mW": args.k,
        "P_active_model_mW": args.p_idle + args.k,

        "full_MAE_mean_mW": per_run["full_mae_mW"].mean(),
        "full_RMSE_mean_mW": per_run["full_rmse_mW"].mean(),
        "full_mean_error_mean_mW": per_run["full_mean_error_mW"].mean(),

        "plateau_only_MAE_mean_mW": per_run["plateau_only_mae_mW"].mean(),
        "plateau_only_RMSE_mean_mW": per_run["plateau_only_rmse_mW"].mean(),
        "plateau_only_mean_error_mean_mW": per_run["plateau_only_mean_error_mW"].mean(),

        "active_window_MAE_mean_mW": per_run["active_window_mae_mW"].mean(),
        "idle_window_MAE_mean_mW": per_run["idle_window_mae_mW"].mean(),
    }])

    overall_path = output_dir / "validation_overall_summary.csv"
    overall.to_csv(overall_path, index=False)

    print("Input folder:")
    print(input_dir)
    print()

    print("Output folder:")
    print(output_dir)
    print()

    print("Processed files:")
    for file in csv_files:
        print(f"  - {file.name}")
    print()

    print("Saved:")
    print(f"  - {per_run_path}")
    print(f"  - {overall_path}")

    if args.save_predictions:
        print(f"  - prediction CSVs in {prediction_dir}")

    if not args.no_plots:
        print(f"  - plots in {plot_dir}")

    print()
    print("Overall validation summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()