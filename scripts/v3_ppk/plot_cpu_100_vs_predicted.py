import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CSV_PATH = (
    "data/processed/v3_ppk/cpu_100/"
    "ppk_cpu_100_aligned_mean_waveform.csv"
)

OUT_DIR = "data/processed/v3_ppk/cpu_100"
OUT_PATH = os.path.join(
    OUT_DIR,
    "ppk_cpu_100_identified_model_vs_mean_waveform.png",
)

os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Identified CPU model parameters
# ------------------------------------------------------------

I_IDLE_MA = 47.2668
DELTA_I_MA = 20.1888
TAU_S = 0.00049

ACTIVE_START_S = 0.0
ACTIVE_END_S = 20.0


# ------------------------------------------------------------
# Load measured mean waveform
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_columns = {
    "time_from_active_start_s",
    "mean_current_mA",
}

missing = required_columns - set(df.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

t = df["time_from_active_start_s"].to_numpy(dtype=float)
measured = df["mean_current_mA"].to_numpy(dtype=float)


# ------------------------------------------------------------
# First-order CPU model
# ------------------------------------------------------------

def cpu_model_current(time_s):
    """
    Identified first-order CPU model.

    idle -> CPU100 at t = 0 s
    CPU100 -> idle at t = 20 s
    """

    current = np.full_like(
        time_s,
        I_IDLE_MA,
        dtype=float,
    )

    # Rise: 0 <= t < 20 s
    active_mask = (
        (time_s >= ACTIVE_START_S)
        & (time_s < ACTIVE_END_S)
    )

    current[active_mask] = (
        I_IDLE_MA
        + DELTA_I_MA
        * (
            1.0
            - np.exp(
                -(time_s[active_mask] - ACTIVE_START_S)
                / TAU_S
            )
        )
    )

    # Current just before the falling edge
    active_end_current = (
        I_IDLE_MA
        + DELTA_I_MA
        * (
            1.0
            - np.exp(
                -(ACTIVE_END_S - ACTIVE_START_S)
                / TAU_S
            )
        )
    )

    # Fall: t >= 20 s
    fall_mask = time_s >= ACTIVE_END_S

    current[fall_mask] = (
        I_IDLE_MA
        + (active_end_current - I_IDLE_MA)
        * np.exp(
            -(time_s[fall_mask] - ACTIVE_END_S)
            / TAU_S
        )
    )

    return current


predicted = cpu_model_current(t)


# ------------------------------------------------------------
# Plot window
# ------------------------------------------------------------

plot_mask = (t >= -5.0) & (t <= 25.0)

t_plot = t[plot_mask]
measured_plot = measured[plot_mask]
predicted_plot = predicted[plot_mask]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 5.5))

plt.plot(
    t_plot,
    measured_plot,
    linewidth=1.2,
    label="PPK2 10-run mean",
)

plt.plot(
    t_plot,
    predicted_plot,
    linewidth=2.0,
    label="identified first-order model",
)

plt.axvline(
    ACTIVE_START_S,
    linestyle="--",
    linewidth=1,
    label="active start",
)

plt.axvline(
    ACTIVE_END_S,
    linestyle="--",
    linewidth=1,
    label="active end",
)

plt.xlabel("Time from active start [s]")
plt.ylabel("Current [mA]")

plt.title(
    "CPU100 training response: measured mean vs identified model"
)

plt.xlim(-5, 25)
plt.ylim(40, 75)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


print("Saved:")
print(OUT_PATH)