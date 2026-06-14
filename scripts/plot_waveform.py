import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/raw/serial_test_run01.csv")
OUTPUT_FILE = Path("results/plots/serial_test_run01.png")

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    print(df.head())
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["current_mA"])
    plt.xlabel("Time [s]")
    plt.ylabel("Current [mA]")
    plt.title("Serial Test Fake Current Data")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
