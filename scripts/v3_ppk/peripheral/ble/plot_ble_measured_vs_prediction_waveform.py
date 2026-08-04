from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_ble_mae_input_definitions import (
    ACTIVE_START_S,
    INPUT_DIRS,
    OUTPUT_DIR as MAE_OUTPUT_DIR,
    UNIFORM_DT_S,
    active_uniform_measurement,
    detect_sync_midpoint,
    infer_run_id,
    make_occupancy_input,
    make_pulse_input,
    read_ppk_csv,
    simulate_first_order_ode,
)


OUTPUT_DIR = Path("results/v3_ppk/peripheral/ble_adv_only/waveform_overlay")

# Representative run used for the LinkedIn-style plots.
# If the exact run is not found, the first CSV in the condition folder is used.
PREFERRED_RUN_ID = "run1"

# Zoom windows from active start. These keep the BLE pulses visible.
ZOOM_WINDOWS = {
    "ble_adv_100ms": (5.0, 7.0),
    "ble_adv_500ms": (4.5, 9.5),
    "ble_adv_1000ms": (4.5, 14.5),
}

LABELS = {
    "ble_adv_100ms": "100 ms",
    "ble_adv_500ms": "500 ms",
    "ble_adv_1000ms": "1000 ms",
}


def collect_csv_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def choose_representative_file(condition: str) -> Path:
    files = collect_csv_files(INPUT_DIRS[condition])
    if not files:
        raise FileNotFoundError(f"No CSV files found for {condition}: {INPUT_DIRS[condition]}")

    for path in files:
        if infer_run_id(path).lower() == PREFERRED_RUN_ID.lower():
            return path

    return files[0]


def moving_average(values: np.ndarray, window_s: float) -> np.ndarray:
    samples = max(1, int(round(window_s / UNIFORM_DT_S)))
    if samples == 1:
        return values
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(values, kernel, mode="same")


def load_waveforms(condition: str, source_file: Path) -> dict[str, np.ndarray | float | str]:
    df = read_ppk_csv(source_file)
    sync_mid_s = detect_sync_midpoint(df)
    df["aligned_time_s"] = df["time_s"] - sync_mid_s

    t, measured_delta, initial_mean_ma = active_uniform_measurement(df)

    u_occ = make_occupancy_input(condition, t)
    u_pulse = make_pulse_input(condition, t)
    pred_occ = simulate_first_order_ode(u_occ, UNIFORM_DT_S)
    pred_pulse = simulate_first_order_ode(u_pulse, UNIFORM_DT_S)

    return {
        "condition": condition,
        "source_file": str(source_file),
        "run_id": infer_run_id(source_file),
        "time_s": t,
        "measured_delta_ma": measured_delta,
        "measured_delta_smooth_ma": moving_average(measured_delta, window_s=0.002),
        "pred_occ_delta_ma": pred_occ,
        "pred_pulse_delta_ma": pred_pulse,
        "initial_mean_ma": initial_mean_ma,
        "sync_mid_s": sync_mid_s,
    }


def plot_condition(wave: dict[str, np.ndarray | float | str], zoom: bool) -> Path:
    condition = str(wave["condition"])
    t = wave["time_s"]  # type: ignore[assignment]
    measured = wave["measured_delta_ma"]  # type: ignore[assignment]
    measured_smooth = wave["measured_delta_smooth_ma"]  # type: ignore[assignment]
    pred_occ = wave["pred_occ_delta_ma"]  # type: ignore[assignment]
    pred_pulse = wave["pred_pulse_delta_ma"]  # type: ignore[assignment]

    if zoom:
        start_s, end_s = ZOOM_WINDOWS[condition]
        mask = (t >= start_s) & (t <= end_s)
        suffix = "zoom"
    else:
        start_s = 0.0
        end_s = float(t[-1])
        mask = np.ones_like(t, dtype=bool)
        suffix = "full_active"

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(
        t[mask],
        measured[mask],
        color="0.75",
        linewidth=0.4,
        alpha=0.55,
        label="Measured delta current, raw",
    )
    ax.plot(
        t[mask],
        measured_smooth[mask],
        color="black",
        linewidth=1.1,
        alpha=0.9,
        label="Measured delta current, 2 ms smooth",
    )
    ax.plot(
        t[mask],
        pred_occ[mask],
        color="#1f77b4",
        linewidth=2.0,
        label="Prediction: occupancy input",
    )
    ax.plot(
        t[mask],
        pred_pulse[mask],
        color="#ff7f0e",
        linewidth=2.0,
        label="Prediction: pulse input",
    )

    ax.axhline(0.0, color="0.3", linewidth=0.8, alpha=0.6)
    ax.set_xlim(start_s, end_s)
    ax.set_xlabel("Time from BLE advertising start (s)")
    ax.set_ylabel("Delta current from initial idle (mA)")
    ax.set_title(
        f"BLE advertising {LABELS[condition]}: measured vs model prediction "
        f"({wave['run_id']})"
    )
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{condition}_{wave['run_id']}_{suffix}_measured_vs_prediction.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_three_panel(waves: list[dict[str, np.ndarray | float | str]], zoom: bool) -> Path:
    fig, axes = plt.subplots(len(waves), 1, figsize=(12, 10), sharex=False)

    for ax, wave in zip(axes, waves):
        condition = str(wave["condition"])
        t = wave["time_s"]  # type: ignore[assignment]
        measured_smooth = wave["measured_delta_smooth_ma"]  # type: ignore[assignment]
        pred_occ = wave["pred_occ_delta_ma"]  # type: ignore[assignment]
        pred_pulse = wave["pred_pulse_delta_ma"]  # type: ignore[assignment]

        if zoom:
            start_s, end_s = ZOOM_WINDOWS[condition]
            mask = (t >= start_s) & (t <= end_s)
            suffix = "zoom"
        else:
            start_s = 0.0
            end_s = float(t[-1])
            mask = np.ones_like(t, dtype=bool)
            suffix = "full_active"

        ax.plot(t[mask], measured_smooth[mask], color="black", linewidth=1.0, label="Measured")
        ax.plot(t[mask], pred_occ[mask], color="#1f77b4", linewidth=1.8, label="Occupancy prediction")
        ax.plot(t[mask], pred_pulse[mask], color="#ff7f0e", linewidth=1.8, label="Pulse prediction")
        ax.axhline(0.0, color="0.3", linewidth=0.8, alpha=0.6)
        ax.set_xlim(start_s, end_s)
        ax.set_ylabel("Delta current (mA)")
        ax.set_title(f"{LABELS[condition]} advertising interval")
        ax.grid(True, alpha=0.28)

    axes[-1].set_xlabel("Time from BLE advertising start (s)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("BLE advertising: measured waveform vs two input-definition predictions", y=0.995)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"ble_three_interval_{suffix}_measured_vs_prediction.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    waves = []
    for condition in ("ble_adv_100ms", "ble_adv_500ms", "ble_adv_1000ms"):
        source_file = choose_representative_file(condition)
        print(f"Loading {condition}: {source_file}", flush=True)
        wave = load_waveforms(condition, source_file)
        waves.append(wave)

        for zoom in (True, False):
            output_path = plot_condition(wave, zoom=zoom)
            print(f"Wrote: {output_path}", flush=True)

    for zoom in (True, False):
        output_path = plot_three_panel(waves, zoom=zoom)
        print(f"Wrote: {output_path}", flush=True)

    print()
    print("Use these plots together with:")
    print(f"  {MAE_OUTPUT_DIR / 'summary_mae.csv'}")


if __name__ == "__main__":
    main()