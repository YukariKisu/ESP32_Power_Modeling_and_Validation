from pathlib import Path
import argparse
import pandas as pd
import numpy as np


POWER_COL_CANDIDATES = [
    "power_mW",
    "power_mw",
    "Power_mW",
    "power",
]

TIME_COL_CANDIDATES = [
    "time_s",
    "timestamp_s",
    "elapsed_s",
    "time_ms",
    "timestamp_ms",
    "elapsed_ms",
]

WORKLOAD_COL_CANDIDATES = [
    "workload",
    "workload_level",
    "cpu_busy",
    "busy",
    "u",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate K for CPU 100% workload from 10 ms measurement CSV files."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing raw measurement CSV files."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v2_timer_driven_polling/first_ode"),
        help="Folder to save summary CSV files."
    )

    parser.add_argument(
        "--p-idle",
        type=float,
        default=168.956222,
        help="Stable idle baseline power in mW."
    )

    parser.add_argument(
        "--stable-margin-s",
        type=float,
        default=1.0,
        help="Seconds to remove from the start and end of the active region."
    )

    parser.add_argument(
        "--active-start-s",
        type=float,
        default=None,
        help="Manual active start time in seconds, used if no workload column exists."
    )

    parser.add_argument(
        "--active-end-s",
        type=float,
        default=None,
        help="Manual active end time in seconds, used if no workload column exists."
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="CSV filename pattern. Example: run*.csv"
    )

    return parser.parse_args()


def find_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_time_seconds(df):
    time_col = find_col(df, TIME_COL_CANDIDATES)

    if time_col is None:
        # fallback: assume 10 ms sampling
        df["t_s"] = np.arange(len(df)) * 0.01
        return df

    if time_col.endswith("_ms"):
        df["t_s"] = df[time_col] / 1000.0
    else:
        df["t_s"] = df[time_col]

    df["t_s"] = df["t_s"] - df["t_s"].iloc[0]
    return df


def get_active_mask(df, active_start_s, active_end_s):
    workload_col = find_col(df, WORKLOAD_COL_CANDIDATES)

    if workload_col is not None:
        max_u = df[workload_col].max()

        if max_u > 2:
            # percentage style: 0, 25, 50, 100
            return df[workload_col] >= 99
        else:
            # normalized style: 0, 0.25, 0.5, 1.0
            return df[workload_col] >= 0.99

    if active_start_s is None or active_end_s is None:
        raise ValueError(
            "No workload column found. "
            "Please specify --active-start-s and --active-end-s."
        )

    return (df["t_s"] >= active_start_s) & (df["t_s"] <= active_end_s)


def get_stable_mask(df, active_mask, stable_margin_s):
    active_df = df[active_mask]

    if active_df.empty:
        raise ValueError("No active samples found.")

    active_start = active_df["t_s"].min()
    active_end = active_df["t_s"].max()

    stable_start = active_start + stable_margin_s
    stable_end = active_end - stable_margin_s

    stable_mask = (
        active_mask
        & (df["t_s"] >= stable_start)
        & (df["t_s"] <= stable_end)
    )

    # If active duration is too short, use the whole active region.
    if stable_mask.sum() == 0:
        stable_mask = active_mask

    return stable_mask


def is_raw_run_csv(file):
    name = file.name.lower()

    excluded_keywords = [
        "summary",
        "jitter",
        "baseline",
    ]

    return not any(keyword in name for keyword in excluded_keywords)


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    p_idle_mw = args.p_idle

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        f for f in input_dir.glob(args.pattern)
        if is_raw_run_csv(f)
    )

    if not csv_files:
        raise FileNotFoundError(f"No raw run CSV files found in: {input_dir}")

    rows = []

    for file in csv_files:
        df = pd.read_csv(file)
        df = add_time_seconds(df)

        power_col = find_col(df, POWER_COL_CANDIDATES)

        if power_col is None:
            raise ValueError(
                f"No power column found in {file.name}. "
                f"Available columns: {list(df.columns)}"
            )

        active_mask = get_active_mask(
            df,
            active_start_s=args.active_start_s,
            active_end_s=args.active_end_s,
        )

        stable_mask = get_stable_mask(
            df,
            active_mask,
            stable_margin_s=args.stable_margin_s,
        )

        active_power = df.loc[active_mask, power_col]
        stable_power = df.loc[stable_mask, power_col]

        mean_active = active_power.mean()
        mean_stable = stable_power.mean()

        k_mw = mean_stable - p_idle_mw

        rows.append({
            "run": file.stem,
            "source_file": file.name,

            "n_total_samples": len(df),
            "n_active_samples": int(active_mask.sum()),
            "n_stable_samples": int(stable_mask.sum()),

            "mean_power_cpu_100_active_mW": mean_active,
            "sd_power_cpu_100_active_mW": active_power.std(ddof=1),
            "min_power_cpu_100_active_mW": active_power.min(),
            "max_power_cpu_100_active_mW": active_power.max(),

            "mean_power_cpu_100_stable_mW": mean_stable,
            "sd_power_cpu_100_stable_mW": stable_power.std(ddof=1),
            "min_power_cpu_100_stable_mW": stable_power.min(),
            "max_power_cpu_100_stable_mW": stable_power.max(),

            "P_idle_mW": p_idle_mw,
            "K_mW": k_mw,
        })

    summary = pd.DataFrame(rows)

    per_run_path = output_dir / "cpu100_k_per_run_summary.csv"
    summary.to_csv(per_run_path, index=False)

    overall = pd.DataFrame([{
        "P_idle_mW": p_idle_mw,

        "K_mean_mW": summary["K_mW"].mean(),
        "K_sd_mW": summary["K_mW"].std(ddof=1),
        "K_min_mW": summary["K_mW"].min(),
        "K_max_mW": summary["K_mW"].max(),

        "mean_power_cpu_100_stable_overall_mW":
            summary["mean_power_cpu_100_stable_mW"].mean(),

        "sd_between_runs_power_cpu_100_stable_mW":
            summary["mean_power_cpu_100_stable_mW"].std(ddof=1),

        "n_runs": len(summary),
    }])

    overall_path = output_dir / "cpu100_k_overall_summary.csv"
    overall.to_csv(overall_path, index=False)

    print("Input folder:")
    print(input_dir)
    print()

    print("Processed CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    print()

    print("Saved:")
    print(f"  - {per_run_path}")
    print(f"  - {overall_path}")
    print()

    print("Overall K summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()