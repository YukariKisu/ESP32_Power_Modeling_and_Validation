import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATTERN = "data/raw/v3_ppk/cpu_100/ppk_cpu_100_run*.csv"
OUT_DIR = "data/processed/v3_ppk/cpu_100"

FILES = sorted(glob.glob(DATA_PATTERN))

print("Files to plot:")
for f in FILES:
    print(f)

if not FILES:
    raise FileNotFoundError(f"No CSV files found: {DATA_PATTERN}")

os.makedirs(OUT_DIR, exist_ok=True)


def detect_columns(csv_path):
    cols = pd.read_csv(csv_path, nrows=0).columns

    timestamp_col = None
    current_col = None

    for col in cols:
        name = col.lower()

        if timestamp_col is None and ("timestamp" in name or "time" in name):
            timestamp_col = col

        if current_col is None and "current" in name:
            current_col = col

    if timestamp_col is None or current_col is None:
        print("Columns found:")
        print(list(cols))
        raise ValueError(f"Could not detect timestamp/current columns in {csv_path}")

    return timestamp_col, current_col


def load_ppk2_csv(csv_path):
    timestamp_col, current_col = detect_columns(csv_path)

    df = pd.read_csv(csv_path, usecols=[timestamp_col, current_col])

    t = df[timestamp_col].astype(float)
    current = df[current_col].astype(float)

    # PPK2 timestamp is usually ms in your export.
    time_s = (t - t.iloc[0]) / 1000.0

    return time_s, current


plt.figure(figsize=(12, 6))

for csv_path in FILES:
    time_s, current = load_ppk2_csv(csv_path)

    # Downsample for plotting
    step = max(len(time_s) // 20000, 1)

    label = os.path.basename(csv_path).replace(".csv", "")
    plt.plot(time_s.iloc[::step], current.iloc[::step], label=label, alpha=0.7)

plt.xlabel("Time [s]")
plt.ylabel("Current")
plt.title("PPK2 CPU-only 100% busy step: all runs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ppk2_cpu_only_100busy_all_runs_overlay.png")
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")

# ---------------- Zoomed plot without boot spike ----------------

plt.figure(figsize=(12, 6))

for csv_path in FILES:
    time_s, current = load_ppk2_csv(csv_path)

    # Convert current to mA
    # PPK2 export is likely in uA, so uA / 1000 = mA
    current_mA = current / 1000.0

    step = max(len(time_s) // 20000, 1)

    label = os.path.basename(csv_path).replace(".csv", "")
    plt.plot(time_s.iloc[::step], current_mA.iloc[::step], label=label, alpha=0.7)

plt.xlabel("Time [s]")
plt.ylabel("Current [mA]")
plt.title("PPK2 CPU-only 100% busy step: zoomed view")

plt.xlim(2, 45)

plt.ylim(40, 85)

plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ppk2_cpu_only_100busy_all_runs_zoom.png")
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")