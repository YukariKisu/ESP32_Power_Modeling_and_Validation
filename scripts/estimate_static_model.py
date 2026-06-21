import pandas as pd

df = pd.read_csv("data/raw/cpu_only_100busy_step/cpu_only_100busy_step_run3.csv")

idle1 = df[(df["timestamp_ms"] >= 2000) & (df["timestamp_ms"] <= 9000)]["power_mW"].mean()
busy  = df[(df["timestamp_ms"] >= 12000) & (df["timestamp_ms"] <= 29000)]["power_mW"].mean()
idle2 = df[(df["timestamp_ms"] >= 32000) & (df["timestamp_ms"] <= 39000)]["power_mW"].mean()

P_idle = (idle1 + idle2) / 2
P_busy = busy

print("Idle: ", P_idle)
print("Busy: ", P_busy)