import pandas as pd
import matplotlib.pyplot as plt

csv_path = "data/raw/v3_ppk/cpu_100/ppk_test_marker.csv"

df = pd.read_csv(csv_path)

print("Columns:")
print(df.columns)

# ---- column auto-detection ----
timestamp_col = None
current_col = None
d0_col = None

for col in df.columns:
    name = col.lower()
    if "timestamp" in name or "time" in name:
        timestamp_col = col
    if "current" in name:
        current_col = col
    if name.strip() in ["d0", "digital 0", "digital0"] or "d0" in name:
        d0_col = col

print("timestamp_col =", timestamp_col)
print("current_col   =", current_col)
print("d0_col        =", d0_col)

if timestamp_col is None or current_col is None or d0_col is None:
    raise ValueError("Could not detect timestamp/current/D0 columns. Check printed column names.")

# ---- normalize time ----
t = df[timestamp_col].astype(float)

time_s = (t - t.iloc[0]) / 1000.0

print("time range [s]:", time_s.iloc[0], "to", time_s.iloc[-1])
print("duration [s]:", time_s.iloc[-1] - time_s.iloc[0])

current = df[current_col].astype(float)
d0 = df[d0_col].astype(float)

# ---- downsample for plotting ----
# 100 kS/s is huge, so plot every Nth sample
step = max(len(df) // 20000, 1)

plt.figure(figsize=(12, 6))
plt.plot(time_s.iloc[::step], current.iloc[::step], label="Current")

# Scale D0 so it appears on the same graph
d0_scaled = d0 * (current.max() - current.min()) + current.min()
plt.plot(time_s.iloc[::step], d0_scaled.iloc[::step], label="D0 marker scaled", alpha=0.8)

plt.xlabel("Time [s]")
plt.ylabel("Current")
plt.title("PPK2 current waveform with GPIO D0 marker")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- find D0 edges ----
d0_diff = d0.diff()

rising_edges = df.index[d0_diff == 1].tolist()
falling_edges = df.index[d0_diff == -1].tolist()

print("\nD0 rising edges:")
for idx in rising_edges:
    print(time_s.iloc[idx])

print("\nD0 falling edges:")
for idx in falling_edges:
    print(time_s.iloc[idx])