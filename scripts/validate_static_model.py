import pandas as pd
import numpy as np

# Model 2: static intensity model
# P_active(duty) = A + B * duty_percent
P_IDLE = 166.67
A = 168.330
B = 0.526

files = {
    "cpu_only_50busy_step": {
        "duty": 50,
        "paths": [
            "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run1.csv",
            "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run2.csv",
            "data/raw/cpu_only_50busy_step/cpu_only_50buty_step_run3.csv",
        ],
    },
    "cpu_only_25busy_step": {
        "duty": 25,
        "paths": [
            "data/raw/cpu_only_25busy_step/cpu_only_25_busy_run1.csv",
            "data/raw/cpu_only_25busy_step/cpu_only_25_busy_run2.csv",
            "data/raw/cpu_only_25busy_step/cpu_only_25_busy_run3.csv",
        ],
    },
    "cpu_only_50periodic": {
        # In this periodic condition, each busy burst is 100% busy.
        "duty": 100,
        "paths": [
            "data/raw/cpu_only_50_periodic/esp32_cpu_only_50_periodic_run1.csv",
            "data/raw/cpu_only_50_periodic/esp32_cpu_only_50_periodic_run2.csv",
            "data/raw/cpu_only_50_periodic/esp32_cpu_only_50_periodic_run3.csv",
        ],
    },
}

def active_power_from_duty(duty_percent):
    return A + B * duty_percent

def make_step_prediction(df, duty_percent):
    active = (
        (df["timestamp_ms"] >= 10000) &
        (df["timestamp_ms"] < 30000)
    )

    p_active = active_power_from_duty(duty_percent)

    return np.where(active, p_active, P_IDLE)

def make_periodic_prediction(df, duty_percent):
    active = (
        ((df["timestamp_ms"] >= 10000) & (df["timestamp_ms"] < 12000)) |
        ((df["timestamp_ms"] >= 14000) & (df["timestamp_ms"] < 16000)) |
        ((df["timestamp_ms"] >= 18000) & (df["timestamp_ms"] < 20000))
    )

    p_active = active_power_from_duty(duty_percent)

    return np.where(active, p_active, P_IDLE)

for condition, info in files.items():
    duty_percent = info["duty"]
    file_list = info["paths"]
    maes = []

    p_active = active_power_from_duty(duty_percent)

    print("####", condition)
    print("duty =", duty_percent, "%")
    print("predicted active power =", round(p_active, 3), "mW")
    print()

    for file_path in file_list:
        df = pd.read_csv(file_path)

        if "periodic" in condition:
            df["predicted_power"] = make_periodic_prediction(df, duty_percent)
        else:
            df["predicted_power"] = make_step_prediction(df, duty_percent)

        df["error"] = df["power_mW"] - df["predicted_power"]
        mae = df["error"].abs().mean()
        maes.append(mae)

        print(file_path, "MAE =", round(mae, 3), "mW")

    print("Average MAE =", round(np.mean(maes), 3), "mW")
    print()