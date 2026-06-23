import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


files = {
    50: [
        "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run1.csv",
        "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run2.csv",
        "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run3.csv",
    ],
    100: [
        "data/raw/cpu_only_100busy_step/cpu_only_100busy_step_run1.csv",
        "data/raw/cpu_only_100busy_step/cpu_only_100busy_step_run2.csv",
        "data/raw/cpu_only_100busy_step/cpu_only_100busy_step_run3.csv",
    ],
}


IDLE_START_MS = 0
IDLE_END_MS = 10000

ACTIVE_START_MS = 10000
ACTIVE_END_MS = 30000


def load_csv(file_path):
    return pd.read_csv(file_path)

def average_power_in_window(df, start_ms, end_ms):
    window = df[
        (df["timestamp_ms"] >= start_ms) &
        (df["timestamp_ms"] < end_ms)
    ]
    return window["power_mW"].mean()

def condition_average(file_list, start_ms, end_ms):
    values = []

    for file_path in file_list:
        df = load_csv(file_path)
        avg_power = average_power_in_window(df, start_ms, end_ms)
        values.append(avg_power)

    return np.mean(values), np.std(values)


# Use idle phase from 100% busy runs as 0% workload / idle.
p_idle_mean, p_idle_std = condition_average(
    files[100],
    IDLE_START_MS,
    IDLE_END_MS
)

p_50_mean, p_50_std = condition_average(
    files[50],
    ACTIVE_START_MS,
    ACTIVE_END_MS
)

p_100_mean, p_100_std = condition_average(
    files[100],
    ACTIVE_START_MS,
    ACTIVE_END_MS
)

duty = np.array([0, 50, 100])
power = np.array([p_idle_mean, p_50_mean, p_100_mean])
power_std = np.array([p_idle_std, p_50_std, p_100_std])


# Model: P = a * duty + b
a, b = np.polyfit(duty, power, 1)

duty_fit = np.linspace(0, 100, 101)
power_fit = a * duty_fit + b

print("Static intensity model:")
print(f"P(duty) = {b:.3f} + {a:.3f} * duty_percent")
print()
print("Identification points:")
print(f"0% idle:    {p_idle_mean:.3f} mW  (std={p_idle_std:.3f})")
print(f"50% busy:   {p_50_mean:.3f} mW  (std={p_50_std:.3f})")
print(f"100% busy:  {p_100_mean:.3f} mW  (std={p_100_std:.3f})")


output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7, 5))

plt.errorbar(
    duty,
    power,
    yerr=power_std,
    fmt="o",
    capsize=5,
    label="Measured average power"
)

plt.plot(
    duty_fit,
    power_fit,
    linestyle="--",
    label="Linear fit"
)

plt.xlabel("Duty ratio / workload intensity [%]")
plt.ylabel("Average active-phase power [mW]")
plt.title("CPU-only static intensity model identification")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "cpu_only_static_intensity_model.png",
    dpi=300
)

plt.show()