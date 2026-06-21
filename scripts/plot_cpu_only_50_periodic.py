import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/cpu_only_50_periodic/esp32_cpu_only_50_periodic_run1.csv")

df["time_s"] = (
    df["timestamp_ms"]
    - df["timestamp_ms"].iloc[0]
) / 1000

plt.figure(figsize=(10,4))
plt.plot(df["time_s"], df["power_mW"])

plt.axvline(30, linestyle="--", label="active 3 -> idle 3")

plt.xlim(19.5, 20.5)

plt.xlabel("Time [s]")
plt.ylabel("Power [mW]")
plt.title("CPU only 25% busy Run 1: Idle to active transition")
plt.grid(True)
plt.legend()

plt.show()