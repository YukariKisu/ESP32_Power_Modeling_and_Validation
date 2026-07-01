import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


CSV_PATH = "data/processed/v3_ppk/cpu_100/ppk_cpu_100_aligned_mean_waveform.csv"
OUT_DIR = "data/processed/v3_ppk/cpu_100"

os.makedirs(OUT_DIR, exist_ok=True)


df = pd.read_csv(CSV_PATH)

t = df["time_from_active_start_s"].to_numpy()
i = df["mean_current_mA"].to_numpy()

# Use stable regions to estimate idle and active levels
idle_mask = (t >= -5.0) & (t <= -1.0)
active_mask = (t >= 5.0) & (t <= 15.0)

I_idle = np.nanmean(i[idle_mask])
I_active = np.nanmean(i[active_mask])
delta_I = I_active - I_idle

rise_threshold = I_idle + 0.632 * delta_I
fall_threshold = I_idle + 0.368 * delta_I

print(f"I_idle = {I_idle:.4f} mA")
print(f"I_active = {I_active:.4f} mA")
print(f"delta_I = {delta_I:.4f} mA")
print(f"rise_threshold = {rise_threshold:.4f} mA")
print(f"fall_threshold = {fall_threshold:.4f} mA")


def rise_model(t, t0, tau):
    """
    First-order rise model.
    Before t0: idle level.
    After t0: approaches active level.
    """
    return np.where(
        t < t0,
        I_idle,
        I_idle + delta_I * (1.0 - np.exp(-(t - t0) / tau))
    )


def fall_model(t, t0, tau):
    """
    First-order fall model.
    Before t0: active level.
    After t0: approaches idle level.
    """
    return np.where(
        t < t0,
        I_active,
        I_idle + delta_I * np.exp(-(t - t0) / tau)
    )


# Transition windows
# These are narrow windows around the rise and fall transitions.
rise_window_mask = (t >= -0.005) & (t <= 0.020)
fall_window_mask = (t >= 19.985) & (t <= 20.010)

t_rise_data = t[rise_window_mask]
i_rise_data = i[rise_window_mask]

t_fall_data = t[fall_window_mask]
i_fall_data = i[fall_window_mask]

# Remove NaN if any
rise_valid = ~np.isnan(i_rise_data)
fall_valid = ~np.isnan(i_fall_data)

t_rise_data = t_rise_data[rise_valid]
i_rise_data = i_rise_data[rise_valid]

t_fall_data = t_fall_data[fall_valid]
i_fall_data = i_fall_data[fall_valid]


# Initial guesses and bounds
# tau is limited to a positive value.
rise_initial_guess = [-0.001, 0.001]   # t0, tau
rise_bounds = ([-0.010, 1e-6], [0.010, 0.050])

fall_initial_guess = [19.995, 0.001]   # t0, tau
fall_bounds = ([19.980, 1e-6], [20.005, 0.050])


rise_params, rise_cov = curve_fit(
    rise_model,
    t_rise_data,
    i_rise_data,
    p0=rise_initial_guess,
    bounds=rise_bounds,
    maxfev=20000,
)

fall_params, fall_cov = curve_fit(
    fall_model,
    t_fall_data,
    i_fall_data,
    p0=fall_initial_guess,
    bounds=fall_bounds,
    maxfev=20000,
)

t0_rise, tau_rise = rise_params
t0_fall, tau_fall = fall_params

print()
print(f"t0_rise = {t0_rise:.6f} s")
print(f"tau_rise = {tau_rise:.6f} s = {tau_rise * 1000:.3f} ms")

print(f"t0_fall = {t0_fall:.6f} s")
print(f"tau_fall = {tau_fall:.6f} s = {tau_fall * 1000:.3f} ms")


# Check time resolution
dt = np.nanmedian(np.diff(t))
print()
print(f"time grid interval = {dt:.7f} s = {dt * 1000:.4f} ms")

if tau_rise < 3 * dt:
    print("Note: tau_rise is close to the time resolution. Treat it as a rough estimate.")

if tau_fall < 3 * dt:
    print("Note: tau_fall is close to the time resolution. Treat it as a rough estimate.")


# Save summary
summary_df = pd.DataFrame([
    {"name": "I_idle", "value": I_idle, "unit": "mA"},
    {"name": "I_active", "value": I_active, "unit": "mA"},
    {"name": "delta_I", "value": delta_I, "unit": "mA"},
    {"name": "rise_threshold", "value": rise_threshold, "unit": "mA"},
    {"name": "fall_threshold", "value": fall_threshold, "unit": "mA"},
    {"name": "t0_rise", "value": t0_rise, "unit": "s"},
    {"name": "tau_rise", "value": tau_rise, "unit": "s"},
    {"name": "t0_fall", "value": t0_fall, "unit": "s"},
    {"name": "tau_fall", "value": tau_fall, "unit": "s"},
    {"name": "time_grid_interval", "value": dt, "unit": "s"},
])

summary_path = os.path.join(OUT_DIR, "ppk_cpu_100_tau_fit_summary.csv")
summary_df.to_csv(summary_path, index=False)


# Plot fitting results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rise plot
t_rise_fit = np.linspace(t_rise_data.min(), t_rise_data.max(), 1000)
i_rise_fit = rise_model(t_rise_fit, t0_rise, tau_rise)

axes[0].plot(t_rise_data, i_rise_data, ".", markersize=3, label="mean waveform")
axes[0].plot(t_rise_fit, i_rise_fit, linewidth=2, label="first-order fit")
axes[0].axhline(rise_threshold, linestyle="--", linewidth=1, label="63.2% threshold")
axes[0].axvline(t0_rise, linestyle="--", linewidth=1, label="t0")
axes[0].set_title("Rise transition fit")
axes[0].set_xlabel("Time from active start [s]")
axes[0].set_ylabel("Current [mA]")
axes[0].grid(True)
axes[0].legend(fontsize=8)

# Fall plot
t_fall_fit = np.linspace(t_fall_data.min(), t_fall_data.max(), 1000)
i_fall_fit = fall_model(t_fall_fit, t0_fall, tau_fall)

axes[1].plot(t_fall_data, i_fall_data, ".", markersize=3, label="mean waveform")
axes[1].plot(t_fall_fit, i_fall_fit, linewidth=2, label="first-order fit")
axes[1].axhline(fall_threshold, linestyle="--", linewidth=1, label="36.8% threshold")
axes[1].axvline(t0_fall, linestyle="--", linewidth=1, label="t0")
axes[1].set_title("Fall transition fit")
axes[1].set_xlabel("Time from active start [s]")
axes[1].set_ylabel("Current [mA]")
axes[1].grid(True)
axes[1].legend(fontsize=8)

plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "ppk_cpu_100_tau_fit.png")
plt.savefig(plot_path, dpi=200)
plt.show()

print()
print("Saved:")
print(summary_path)
print(plot_path)