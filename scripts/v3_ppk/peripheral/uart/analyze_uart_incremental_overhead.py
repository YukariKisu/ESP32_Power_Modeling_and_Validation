import re
from pathlib import Path

import pandas as pd

# ---------------- Settings ----------------

WINDOW_START_S = 20.0
WINDOW_END_S = 30.0

D1_IDLE_DIR = Path("data/raw/v3_ppk/idle_baseline")

D3_ENABLED_SUMMARY = Path(
    "results/v3_ppk/peripheral/uart_tx_only/uart_enabled/"
    "uart_enabled_idle_mean_20_30s.csv"
)

# If your folder is accidentally named "peripheal", change this to that path.
UART_TX_BASE_DIR = Path("data/raw/v3_ppk/peripheral/uart_tx_only")

D4_CONDITIONS = {
    "UART_64B_100ms": UART_TX_BASE_DIR / "uart_64B_100ms",
    "UART_256B_100ms": UART_TX_BASE_DIR / "uart_256B_100ms",
    "UART_512B_100ms": UART_TX_BASE_DIR / "uart_512B_100ms",
}

OUTPUT_DIR = Path("results/v3_ppk/peripheral/uart_tx_only/incremental_overhead")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RUN_CSV = OUTPUT_DIR / "uart_incremental_overhead_runs_20_30s.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "uart_incremental_overhead_summary_20_30s.csv"


# ---------------- Helpers ----------------

def natural_key(path: Path):
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else 0


def load_ppk2_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Timestamp(ms)" not in df.columns or "Current(uA)" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")

    df["time_s"] = df["Timestamp(ms)"] / 1000.0
    df["current_mA"] = df["Current(uA)"] / 1000.0

    return df


def mean_current_in_window(path: Path) -> float:
    df = load_ppk2_csv(path)

    window = df[
        (df["time_s"] >= WINDOW_START_S)
        & (df["time_s"] <= WINDOW_END_S)
    ]

    if window.empty:
        raise ValueError(
            f"No samples in {WINDOW_START_S}-{WINDOW_END_S}s window: {path}"
        )

    return window["current_mA"].mean()


def collect_run_means(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No files found: {input_dir}/{pattern}")

    rows = []
    for path in files:
        rows.append({
            "run_index": natural_key(path),
            "file": str(path),
            "mean_current_mA": mean_current_in_window(path),
        })

    return pd.DataFrame(rows)


# ---------------- Main ----------------

# D1: ESP32 idle baseline
d1_df = collect_run_means(D1_IDLE_DIR, "ppk_idle_baseline_run*.csv")
d1_df = d1_df.rename(columns={"mean_current_mA": "D1_idle_baseline_mA"})

# D3: UART enabled idle baseline
d3_df = pd.read_csv(D3_ENABLED_SUMMARY)

if "mean_current_mA" not in d3_df.columns:
    raise ValueError(f"D3 summary does not contain mean_current_mA: {D3_ENABLED_SUMMARY}")

d3_df = d3_df.copy()
d3_df["run_index"] = d3_df.index + 1
d3_df = d3_df[["run_index", "mean_current_mA"]]
d3_df = d3_df.rename(columns={"mean_current_mA": "D3_uart_enabled_idle_mA"})

# Merge D1 and D3 by run index
base_df = pd.merge(
    d1_df[["run_index", "D1_idle_baseline_mA"]],
    d3_df,
    on="run_index",
    how="inner",
)

if len(base_df) == 0:
    raise ValueError("No matching run indexes between D1 and D3")

all_rows = []

for condition, input_dir in D4_CONDITIONS.items():
    d4_df = collect_run_means(input_dir, "*.csv")
    d4_df = d4_df.rename(columns={"mean_current_mA": "D4_uart_tx_active_mA"})

    merged = pd.merge(
        base_df,
        d4_df[["run_index", "D4_uart_tx_active_mA"]],
        on="run_index",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(f"No matching run indexes for {condition}")

    merged["condition"] = condition

    merged["D3_minus_D1_uart_enable_overhead_mA"] = (
        merged["D3_uart_enabled_idle_mA"] - merged["D1_idle_baseline_mA"]
    )

    merged["D4_minus_D3_tx_activity_overhead_mA"] = (
        merged["D4_uart_tx_active_mA"] - merged["D3_uart_enabled_idle_mA"]
    )

    all_rows.append(merged)

run_result = pd.concat(all_rows, ignore_index=True)

run_result = run_result[
    [
        "condition",
        "run_index",
        "D1_idle_baseline_mA",
        "D3_uart_enabled_idle_mA",
        "D4_uart_tx_active_mA",
        "D3_minus_D1_uart_enable_overhead_mA",
        "D4_minus_D3_tx_activity_overhead_mA",
    ]
]

run_result.to_csv(OUTPUT_RUN_CSV, index=False)

summary = (
    run_result
    .groupby("condition")
    .agg(
        D1_mean_mA=("D1_idle_baseline_mA", "mean"),
        D3_mean_mA=("D3_uart_enabled_idle_mA", "mean"),
        D4_mean_mA=("D4_uart_tx_active_mA", "mean"),
        uart_enable_overhead_mean_mA=("D3_minus_D1_uart_enable_overhead_mA", "mean"),
        uart_enable_overhead_std_mA=("D3_minus_D1_uart_enable_overhead_mA", "std"),
        tx_activity_overhead_mean_mA=("D4_minus_D3_tx_activity_overhead_mA", "mean"),
        tx_activity_overhead_std_mA=("D4_minus_D3_tx_activity_overhead_mA", "std"),
        tx_activity_overhead_min_mA=("D4_minus_D3_tx_activity_overhead_mA", "min"),
        tx_activity_overhead_max_mA=("D4_minus_D3_tx_activity_overhead_mA", "max"),
        runs=("run_index", "count"),
    )
    .reset_index()
)

summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print("UART incremental overhead analysis")
print(f"Window: {WINDOW_START_S:.1f}-{WINDOW_END_S:.1f} s")
print()

print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print()
print(f"Saved run results: {OUTPUT_RUN_CSV}")
print(f"Saved summary:     {OUTPUT_SUMMARY_CSV}")