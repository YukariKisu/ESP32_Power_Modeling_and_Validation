
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/cpu_only_25busy_step/cpu_only_25_busy_run1.csv")

df["time_s"] = (
    df["timestamp_ms"]
    - df["timestamp_ms"].iloc[0]
) / 1000

plt.figure(figsize=(10,4))
plt.plot(df["time_s"], df["power_mW"])

plt.axvline(30, linestyle="--", label="active -> idle")

plt.xlim(29.5, 30.5)

plt.xlabel("Time [s]")
plt.ylabel("Power [mW]")
plt.title("CPU only 25% busy Run 1: Idle to active transition")
plt.grid(True)
plt.legend()

plt.show()