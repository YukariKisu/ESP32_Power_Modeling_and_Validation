from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


BASE_DIR = Path(
    "data/processed/v3_ppk/peripheral/adc/"
    "final_predictioned/adc_remodel_train"
)

AVERAGE_CSV_PATH = BASE_DIR / "adc_continuous_train_aligned_average.csv"
DECOMPOSITION_SUMMARY_PATH = (
    BASE_DIR / "adc_continuous_train_decomposition_summary.csv"
)

OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_START_ALIGNED_S = 16.0
ACTIVE_DURATION_S = 20.0
ACTIVE_END_RELATIVE_S = 20.0

RISE_WINDOW_START_S = -0.005
RISE_WINDOW_END_S = 0.020
FALL_WINDOW_START_S = 19.985
FALL_WINDOW_END_S = 20.010


def load_decomposition_value(summary_df: pd.DataFrame, metric: str) -> float:
    row = summary_df.loc[summary_df["metric"] == metric]
    if row.empty:
        available = ", ".join(summary_df["metric"].astype(str))
        raise KeyError(
            f"Metric '{metric}' was not found.\nAvailable metrics: {available}"
        )
    return float(row.iloc[0]["mean"])


def rise_model(
    time_s: np.ndarray,
    t0_s: float,
    tau_s: float,
    i_idle_ma: float,
    delta_i_ma: float,
) -> np.ndarray:
    return np.where(
        time_s < t0_s,
        i_idle_ma,
        i_idle_ma
        + delta_i_ma * (1.0 - np.exp(-(time_s - t0_s) / tau_s)),
    )


def fall_model(
    time_s: np.ndarray,
    t0_s: float,
    tau_s: float,
    i_idle_ma: float,
    i_active_ma: float,
    delta_i_ma: float,
) -> np.ndarray:
    return np.where(
        time_s < t0_s,
        i_active_ma,
        i_idle_ma
        + delta_i_ma * np.exp(-(time_s - t0_s) / tau_s),
    )


def calculate_rmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((measured - predicted) ** 2)))


if not AVERAGE_CSV_PATH.exists():
    raise FileNotFoundError(f"Aligned average CSV not found: {AVERAGE_CSV_PATH}")

if not DECOMPOSITION_SUMMARY_PATH.exists():
    raise FileNotFoundError(
        f"Decomposition summary CSV not found: {DECOMPOSITION_SUMMARY_PATH}"
    )

wave_df = pd.read_csv(AVERAGE_CSV_PATH)
summary_df = pd.read_csv(DECOMPOSITION_SUMMARY_PATH)

required_wave_columns = {"aligned_time_s", "mean_current_ma"}
missing = required_wave_columns - set(wave_df.columns)
if missing:
    raise KeyError(f"Missing waveform columns: {sorted(missing)}")

t = (
    wave_df["aligned_time_s"].to_numpy(dtype=float)
    - ACTIVE_START_ALIGNED_S
)
i = wave_df["mean_current_ma"].to_numpy(dtype=float)

I_idle = load_decomposition_value(
    summary_df,
    "I3_adc_preconditioned_idle_ma",
)
I_active = load_decomposition_value(
    summary_df,
    "I4_continuous_adc_active_ma",
)
delta_I = I_active - I_idle

rise_threshold = I_idle + 0.632 * delta_I
fall_threshold = I_idle + 0.368 * delta_I

print(f"I_idle = {I_idle:.6f} mA")
print(f"I_active = {I_active:.6f} mA")
print(f"delta_I = {delta_I:.6f} mA")
print(f"rise_threshold = {rise_threshold:.6f} mA")
print(f"fall_threshold = {fall_threshold:.6f} mA")


def rise_fit_function(time_s: np.ndarray, t0_s: float, tau_s: float) -> np.ndarray:
    return rise_model(time_s, t0_s, tau_s, I_idle, delta_I)


def fall_fit_function(time_s: np.ndarray, t0_s: float, tau_s: float) -> np.ndarray:
    return fall_model(time_s, t0_s, tau_s, I_idle, I_active, delta_I)


rise_mask = (t >= RISE_WINDOW_START_S) & (t <= RISE_WINDOW_END_S)
fall_mask = (t >= FALL_WINDOW_START_S) & (t <= FALL_WINDOW_END_S)

t_rise_data = t[rise_mask]
i_rise_data = i[rise_mask]
t_fall_data = t[fall_mask]
i_fall_data = i[fall_mask]

rise_valid = np.isfinite(t_rise_data) & np.isfinite(i_rise_data)
fall_valid = np.isfinite(t_fall_data) & np.isfinite(i_fall_data)

t_rise_data = t_rise_data[rise_valid]
i_rise_data = i_rise_data[rise_valid]
t_fall_data = t_fall_data[fall_valid]
i_fall_data = i_fall_data[fall_valid]

if len(t_rise_data) < 5:
    raise ValueError("Too few valid samples in the rise fit window.")

if len(t_fall_data) < 5:
    raise ValueError("Too few valid samples in the fall fit window.")

rise_initial_guess = [-0.001, 0.001]
rise_bounds = ([-0.010, 1e-6], [0.010, 0.050])

fall_initial_guess = [19.999, 0.001]
fall_bounds = ([19.980, 1e-6], [20.010, 0.050])

rise_params, rise_cov = curve_fit(
    rise_fit_function,
    t_rise_data,
    i_rise_data,
    p0=rise_initial_guess,
    bounds=rise_bounds,
    maxfev=50000,
)

fall_params, fall_cov = curve_fit(
    fall_fit_function,
    t_fall_data,
    i_fall_data,
    p0=fall_initial_guess,
    bounds=fall_bounds,
    maxfev=50000,
)

t0_rise, tau_rise = rise_params
t0_fall, tau_fall = fall_params

rise_std_errors = np.sqrt(np.diag(rise_cov))
fall_std_errors = np.sqrt(np.diag(fall_cov))

t0_rise_se, tau_rise_se = rise_std_errors
t0_fall_se, tau_fall_se = fall_std_errors

rise_prediction = rise_fit_function(t_rise_data, t0_rise, tau_rise)
fall_prediction = fall_fit_function(t_fall_data, t0_fall, tau_fall)

rise_rmse_ma = calculate_rmse(i_rise_data, rise_prediction)
fall_rmse_ma = calculate_rmse(i_fall_data, fall_prediction)

print()
print(f"t0_rise = {t0_rise:.9f} s")
print(f"tau_rise = {tau_rise:.9f} s = {tau_rise * 1000:.6f} ms")
print(f"tau_rise standard error = {tau_rise_se * 1000:.6f} ms")
print(f"rise RMSE = {rise_rmse_ma:.6f} mA")

print()
print(f"t0_fall = {t0_fall:.9f} s")
print(f"tau_fall = {tau_fall:.9f} s = {tau_fall * 1000:.6f} ms")
print(f"tau_fall standard error = {tau_fall_se * 1000:.6f} ms")
print(f"fall RMSE = {fall_rmse_ma:.6f} mA")

finite_time = t[np.isfinite(t)]
dt = float(np.nanmedian(np.diff(finite_time)))

print()
print(f"time grid interval = {dt:.9f} s = {dt * 1000:.6f} ms")
print(f"rise tau / dt = {tau_rise / dt:.3f} samples")
print(f"fall tau / dt = {tau_fall / dt:.3f} samples")

if tau_rise < 3 * dt:
    print(
        "WARNING: tau_rise is represented by fewer than three time-grid "
        "intervals. Treat it as a rough PPK2 estimate."
    )

if tau_fall < 3 * dt:
    print(
        "WARNING: tau_fall is represented by fewer than three time-grid "
        "intervals. Treat it as a rough PPK2 estimate."
    )

tau_mean = float((tau_rise + tau_fall) / 2.0)
tau_geometric_mean = float(np.sqrt(tau_rise * tau_fall))

print()
print(f"provisional arithmetic-mean tau = {tau_mean * 1000:.6f} ms")
print(f"provisional geometric-mean tau = {tau_geometric_mean * 1000:.6f} ms")

summary_output_df = pd.DataFrame(
    [
        {"name": "I_idle", "value": I_idle, "unit": "mA"},
        {"name": "I_active", "value": I_active, "unit": "mA"},
        {"name": "delta_I", "value": delta_I, "unit": "mA"},
        {"name": "rise_threshold", "value": rise_threshold, "unit": "mA"},
        {"name": "fall_threshold", "value": fall_threshold, "unit": "mA"},
        {"name": "t0_rise", "value": t0_rise, "unit": "s"},
        {"name": "tau_rise", "value": tau_rise, "unit": "s"},
        {
            "name": "tau_rise_standard_error",
            "value": tau_rise_se,
            "unit": "s",
        },
        {"name": "rise_fit_rmse", "value": rise_rmse_ma, "unit": "mA"},
        {"name": "t0_fall", "value": t0_fall, "unit": "s"},
        {"name": "tau_fall", "value": tau_fall, "unit": "s"},
        {
            "name": "tau_fall_standard_error",
            "value": tau_fall_se,
            "unit": "s",
        },
        {"name": "fall_fit_rmse", "value": fall_rmse_ma, "unit": "mA"},
        {
            "name": "provisional_tau_arithmetic_mean",
            "value": tau_mean,
            "unit": "s",
        },
        {
            "name": "provisional_tau_geometric_mean",
            "value": tau_geometric_mean,
            "unit": "s",
        },
        {"name": "time_grid_interval", "value": dt, "unit": "s"},
        {
            "name": "rise_tau_over_time_grid",
            "value": tau_rise / dt,
            "unit": "samples",
        },
        {
            "name": "fall_tau_over_time_grid",
            "value": tau_fall / dt,
            "unit": "samples",
        },
    ]
)

summary_path = OUT_DIR / "adc_continuous_tau_fit_summary.csv"
summary_output_df.to_csv(summary_path, index=False)

rise_output_df = pd.DataFrame(
    {
        "time_from_active_start_s": t_rise_data,
        "measured_mean_current_ma": i_rise_data,
        "fitted_current_ma": rise_prediction,
        "residual_ma": i_rise_data - rise_prediction,
    }
)
rise_output_path = OUT_DIR / "adc_continuous_tau_fit_rise.csv"
rise_output_df.to_csv(rise_output_path, index=False)

fall_output_df = pd.DataFrame(
    {
        "time_from_active_start_s": t_fall_data,
        "measured_mean_current_ma": i_fall_data,
        "fitted_current_ma": fall_prediction,
        "residual_ma": i_fall_data - fall_prediction,
    }
)
fall_output_path = OUT_DIR / "adc_continuous_tau_fit_fall.csv"
fall_output_df.to_csv(fall_output_path, index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t_rise_fit = np.linspace(t_rise_data.min(), t_rise_data.max(), 1000)
i_rise_fit = rise_fit_function(t_rise_fit, t0_rise, tau_rise)

axes[0].plot(
    t_rise_data * 1000,
    i_rise_data,
    ".",
    markersize=3,
    label="PPK2 10-run mean",
)
axes[0].plot(
    t_rise_fit * 1000,
    i_rise_fit,
    linewidth=2,
    label="first-order fit",
)
axes[0].axhline(
    rise_threshold,
    linestyle="--",
    linewidth=1,
    label="63.2% threshold",
)
axes[0].axvline(
    t0_rise * 1000,
    linestyle="--",
    linewidth=1,
    label="fitted t0",
)
axes[0].set_title(
    f"ADC continuous-read rise\n"
    f"tau = {tau_rise * 1000:.3f} ms"
)
axes[0].set_xlabel("Time from ADC active start [ms]")
axes[0].set_ylabel("Current [mA]")
axes[0].grid(True)
axes[0].legend(fontsize=8)

t_fall_fit = np.linspace(t_fall_data.min(), t_fall_data.max(), 1000)
i_fall_fit = fall_fit_function(t_fall_fit, t0_fall, tau_fall)

axes[1].plot(
    (t_fall_data - ACTIVE_END_RELATIVE_S) * 1000,
    i_fall_data,
    ".",
    markersize=3,
    label="PPK2 10-run mean",
)
axes[1].plot(
    (t_fall_fit - ACTIVE_END_RELATIVE_S) * 1000,
    i_fall_fit,
    linewidth=2,
    label="first-order fit",
)
axes[1].axhline(
    fall_threshold,
    linestyle="--",
    linewidth=1,
    label="36.8% threshold",
)
axes[1].axvline(
    (t0_fall - ACTIVE_END_RELATIVE_S) * 1000,
    linestyle="--",
    linewidth=1,
    label="fitted t0",
)
axes[1].set_title(
    f"ADC continuous-read fall\n"
    f"tau = {tau_fall * 1000:.3f} ms"
)
axes[1].set_xlabel("Time from ADC active end [ms]")
axes[1].set_ylabel("Current [mA]")
axes[1].grid(True)
axes[1].legend(fontsize=8)

plt.tight_layout()

plot_path = OUT_DIR / "adc_continuous_tau_fit.png"
plt.savefig(plot_path, dpi=200)
plt.show()

print()
print("Saved:")
print(summary_path)
print(rise_output_path)
print(fall_output_path)
print(plot_path)