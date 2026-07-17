import pandas as pd
from pathlib import Path

# ---------------- Settings ----------------

INPUT_DIR = Path("data/raw/v3_ppk/peripheral/uart_tx_only/uart_enabled")
OUTPUT_DIR = Path("results/v3_ppk/peripheral/uart_tx_only/uart_enabled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START_S = 20.0
WINDOW_END_S = 30.0

OUTPUT_CSV = OUTPUT_DIR / "uart_enabled_idle_mean_20_30s.csv"

# ---------------- Helpers ----------------

def load_ppk2_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Expected PPK2 columns:
    # Timestamp(ms), Current(uA)
    if "Timestamp(ms)" not in df.columns or "Current(uA)" not in df.columns:
        raise ValueError(f"Unexpected columns in {path.name}: {df.columns.tolist()}")

    df["time_s"] = df["Timestamp(ms)"] / 1000.0
    df["current_mA"] = df["Current(uA)"] / 1000.0

    return df

# ---------------- Main ----------------

rows = []

for path in sorted(INPUT_DIR.glob("uart_enabled_run*.csv")):
    df = load_ppk2_csv(path)

    window = df[
        (df["time_s"] >= WINDOW_START_S) &
        (df["time_s"] <= WINDOW_END_S)
    ]

    if window.empty:
        raise ValueError(f"No samples found in {WINDOW_START_S}-{WINDOW_END_S}s for {path.name}")

    mean_mA = window["current_mA"].mean()
    std_mA = window["current_mA"].std()
    min_mA = window["current_mA"].min()
    max_mA = window["current_mA"].max()

    rows.append({
        "run": path.stem,
        "window_start_s": WINDOW_START_S,
        "window_end_s": WINDOW_END_S,
        "mean_current_mA": mean_mA,
        "std_current_mA": std_mA,
        "min_current_mA": min_mA,
        "max_current_mA": max_mA,
        "num_samples": len(window),
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_CSV, index=False)

summary = result["mean_current_mA"].agg(["mean", "std", "min", "max"])

print("UART enabled idle baseline")
print(f"Input dir: {INPUT_DIR}")
print(f"Window: {WINDOW_START_S:.1f} - {WINDOW_END_S:.1f} s")
print()
print(result[["run", "mean_current_mA", "std_current_mA", "num_samples"]].to_string(index=False))
print()
print("Summary of run means [mA]")
print(f"mean = {summary['mean']:.6f}")
print(f"std  = {summary['std']:.6f}")
print(f"min  = {summary['min']:.6f}")
print(f"max  = {summary['max']:.6f}")
print()
print(f"Saved: {OUTPUT_CSV}")