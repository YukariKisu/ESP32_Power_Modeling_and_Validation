from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/raw/v2_timer_driven_polling/cpu_only_100_10ms")
PATTERN = "cpu_only_100_10ms_run*.csv"

P_IDLE = 168.956222
P_IDLE_SD = 1.398753
DROP_THRESHOLD = P_IDLE + 3 * P_IDLE_SD   # about 173.15 mW

WORKLOAD_CYCLE_MS = 100

rows = []

for file in sorted(INPUT_DIR.glob(PATTERN)):

    if any(word in file.name.lower() for word in ["summary", "jitter", "baseline"]):
        continue

    df = pd.read_csv(file, comment="#")

    # Normalize timestamp
    if "timestamp_us" in df.columns:
        df["t_ms"] = df["timestamp_us"] / 1000.0
    elif "timestamp_ms" in df.columns:
        df["t_ms"] = df["timestamp_ms"]
    else:
        raise ValueError(f"No timestamp column in {file}")

    df["t_ms"] = df["t_ms"] - df["t_ms"].iloc[0]

    # Detect active window from workload_state
    active_df = df[df["workload_state"] == 1].copy()

    if active_df.empty:
        print(f"skip: no active window in {file.name}")
        continue

    active_start_ms = active_df["t_ms"].iloc[0]
    active_end_ms = active_df["t_ms"].iloc[-1]

    # Detect idle-like drops inside active window
    drops = active_df[active_df["power_mW"] <= DROP_THRESHOLD].copy()

    if drops.empty:
        print(f"{file.name}: no drops")
        continue

    drops["active_elapsed_ms"] = drops["t_ms"] - active_start_ms
    drops["phase_in_100ms_cycle"] = drops["active_elapsed_ms"] % WORKLOAD_CYCLE_MS
    drops["dt_from_previous_drop_ms"] = drops["t_ms"].diff()

    for _, r in drops.iterrows():
        rows.append({
            "file": file.name,
            "t_ms": r["t_ms"],
            "power_mW": r["power_mW"],
            "active_elapsed_ms": r["active_elapsed_ms"],
            "phase_in_100ms_cycle": r["phase_in_100ms_cycle"],
            "dt_from_previous_drop_ms": r["dt_from_previous_drop_ms"],
        })

    print()
    print("FILE:", file.name)
    print("active_start_ms:", active_start_ms)
    print("active_end_ms:", active_end_ms)
    print("active_samples:", len(active_df))
    print("drop_samples:", len(drops))
    print("drop_ratio:", len(drops) / len(active_df))
    print()
    print("phase_in_100ms_cycle summary:")
    print(drops["phase_in_100ms_cycle"].describe())
    print()
    print("first 20 drops:")
    print(drops[["t_ms", "power_mW", "active_elapsed_ms", "phase_in_100ms_cycle", "dt_from_previous_drop_ms"]].head(20))

result = pd.DataFrame(rows)

out_dir = Path("data/processed/drop_timing_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "active_window_drop_timing.csv"
result.to_csv(out_file, index=False)

print()
print("Saved:", out_file)

if not result.empty:
    print()
    print("Overall phase distribution, 10 ms bins:")
    bins = np.arange(0, 110, 10)
    result["phase_bin"] = pd.cut(
        result["phase_in_100ms_cycle"],
        bins=bins,
        right=False,
        include_lowest=True
    )
    print(result["phase_bin"].value_counts().sort_index())

    summary_file = out_dir / "active_window_drop_phase_bin_summary.csv"
    result["phase_bin"].value_counts().sort_index().to_csv(summary_file)
    print("Saved:", summary_file)