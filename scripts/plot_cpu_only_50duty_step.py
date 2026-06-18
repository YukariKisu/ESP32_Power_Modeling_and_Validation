import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/cpu_only_50duty_step/cpu_only_50duty_step_run1.csv")

df["time_s"] = (
    df["timestamp_ms"]
    - df["timestamp_ms"].iloc[0]
) / 1000

plt.figure(figsize=(10,4))
plt.plot(df["time_s"], df["power_mW"])

plt.axvline(30, linestyle="--", label="active -> idle")

plt.xlim(29.5, 30.5);

plt.xlabel("Time [s]")
plt.ylabel("Power [mW]")
plt.title("First measurement Run 1: Active to idle transition")
plt.grid(True)
plt.legend()

plt.show()