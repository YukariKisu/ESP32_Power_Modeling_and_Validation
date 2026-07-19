from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Raw data paths.
CONDITIONS = {
    "ble_adv_100ms": {
        "interval_s": 0.100,
        "expected_bursts": 200,
        "input_dir": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_100ms"),
        "pattern": "ble_100ms_run*.csv",
    },
    "ble_adv_500ms": {
        "interval_s": 0.500,
        "expected_bursts": 40,
        "input_dir": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_500ms"),
        "pattern": "ble_500ms_run*.csv",
    },
    "ble_adv_1000ms": {
        "interval_s": 1.000,
        "expected_bursts": 20,
        "input_dir": Path("data/raw/v3_ppk/peripheral/ble_adv_only/ble_1000ms"),
        "pattern": "ble_1000ms_run*.csv",
    },
}

OUTPUT_DIR = Path("results/v3_ppk/peripheral/ble_adv_only/burst_percentiles")


# Experiment timing, relative to sync pulse midpoint.
SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_EXPECTED_MID_S = SYNC_EXPECTED_START_S + SYNC_DURATION_S / 2.0
SYNC_SEARCH_MARGIN_S = 2.0
SYNC_SMOOTH_WINDOW_S = 0.050

INITIAL_IDLE_START_REL_S = 0.5 + 5.0
INITIAL_IDLE_END_REL_S = INITIAL_IDLE_START_REL_S + 10.0
ACTIVE_START_REL_S = INITIAL_IDLE_END_REL_S
ACTIVE_DURATION_S = 20.0
ACTIVE_END_REL_S = ACTIVE_START_REL_S + ACTIVE_DURATION_S

PHASE_TRIM_S = 0.5


# Burst extraction.
# The center is detected from measured current inside each interval slot.
BURST_HALF_WINDOW_S = 0.0025
DETECTION_THRESHOLD_MAD_MULTIPLIERS = [10.0, 8.0, 6.0, 4.0, 3.0]
MIN_DETECTED_FRACTION = 0.80


# Prediction model. Keep these identical to the MAE script.
ODE_GAIN_MA = 21.310
ODE_TAU_RISE_S = 0.050
ODE_TAU_FALL_S = 0.050
BLE_EVENT_DURATION_S = 0.001128
PREDICTION_DT_S = 0.0001


INTERVAL_ORDER = ["ble_adv_100ms", "ble_adv_500ms", "ble_adv_1000ms"]
INTERVAL_LABELS = {
    "ble_adv_100ms": "100 ms",
    "ble_adv_500ms": "500 ms",
    "ble_adv_1000ms": "1000 ms",
}


def natural_run_key(path: Path) -> int:
    match = re.search(r"run(\d+)", path.stem)
    return int(match.group(1)) if match else 10**9


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(
        "Could not find required column. Available columns: "
        + ", ".join(df.columns)
    )


def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = column_name.lower()
    if "ms" in name:
        return values / 1000.0
    if "us" in name or "micro" in name:
        return values / 1_000_000.0
    if np.nanmax(values) > 1000:
        return values / 1000.0
    return values


def convert_current_to_ma(values: np.ndarray, column_name: str) -> np.ndarray:
    name = column_name.lower()
    if "ua" in name or "micro" in name:
        return values / 1000.0
    if "ma" in name:
        return values
    if "(a)" in name or name.endswith("a"):
        return values * 1000.0
    return values / 1000.0


def read_ppk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty")

    time_col = find_column(
        df,
        [
            "Timestamp(ms)",
            "Timestamp (ms)",
            "Time(ms)",
            "Time (ms)",
            "Timestamp",
            "Time",
        ],
    )
    current_col = find_column(
        df,
        [
            "Current(uA)",
            "Current (uA)",
            "Current(mA)",
            "Current (mA)",
            "Current(A)",
            "Current (A)",
            "Current",
        ],
    )

    time_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    current_raw = pd.to_numeric(df[current_col], errors="coerce").to_numpy()
    valid = np.isfinite(time_raw) & np.isfinite(current_raw)
    time_raw = time_raw[valid]
    current_raw = current_raw[valid]

    if len(time_raw) < 10:
        raise ValueError("Not enough valid samples")

    time_s = convert_time_to_s(time_raw, time_col)
    time_s = time_s - time_s[0]
    current_ma = convert_current_to_ma(current_raw, current_col)

    return pd.DataFrame({"time_s": time_s, "current_ma": current_ma})


def downsample_for_sync(
    time_s: np.ndarray, current_ma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 2:
        return time_s, current_ma

    dt = np.median(np.diff(time_s[: min(len(time_s), 10000)]))
    step = max(1, int(round(0.001 / dt))) if dt > 0 else 1
    return time_s[::step], current_ma[::step]


def moving_average(values: np.ndarray, window_s: float, dt_s: float) -> np.ndarray:
    width = max(1, int(round(window_s / dt_s)))
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def detect_sync_midpoint(time_s: np.ndarray, current_ma: np.ndarray) -> float:
    t_ds, i_ds = downsample_for_sync(time_s, current_ma)
    if len(t_ds) < 10:
        raise ValueError("Not enough samples for sync detection")

    dt_ds = np.median(np.diff(t_ds))
    smooth = moving_average(i_ds, SYNC_SMOOTH_WINDOW_S, dt_ds)

    search_start = SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S
    search_end = SYNC_EXPECTED_START_S + SYNC_DURATION_S + SYNC_SEARCH_MARGIN_S
    search_mask = (t_ds >= search_start) & (t_ds <= search_end)
    if not np.any(search_mask):
        raise ValueError("Sync search window has no samples")

    t_search = t_ds[search_mask]
    y_search = smooth[search_mask]

    baseline_mask = (t_ds >= 0.0) & (t_ds < max(0.5, SYNC_EXPECTED_START_S - 0.5))
    baseline = (
        np.median(smooth[baseline_mask])
        if np.any(baseline_mask)
        else np.median(y_search)
    )
    peak = np.max(y_search)
    threshold = baseline + 0.5 * (peak - baseline)

    above = y_search >= threshold
    if not np.any(above):
        return float(t_search[np.argmax(y_search)])

    indices = np.flatnonzero(above)
    return float((t_search[indices[0]] + t_search[indices[-1]]) / 2.0)


def simulate_ode(
    active_time_s: np.ndarray, interval_s: float, input_definition: str
) -> np.ndarray:
    if len(active_time_s) == 0:
        return np.array([], dtype=float)

    t = active_time_s - active_time_s[0]
    duty = BLE_EVENT_DURATION_S / interval_s

    if input_definition == "occupancy":
        u = np.full_like(t, duty, dtype=float)
    elif input_definition == "pulse":
        phase = np.mod(t, interval_s)
        u = (phase < BLE_EVENT_DURATION_S).astype(float)
    else:
        raise ValueError(f"Unknown input definition: {input_definition}")

    y = np.zeros_like(t, dtype=float)
    for idx in range(1, len(t)):
        dt = t[idx] - t[idx - 1]
        target = ODE_GAIN_MA * u[idx]
        tau = ODE_TAU_RISE_S if target >= y[idx - 1] else ODE_TAU_FALL_S
        alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        y[idx] = y[idx - 1] + alpha * (target - y[idx - 1])
    return y


def robust_sigma(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(values))
    return sigma


def burst_detection_threshold(
    measured_delta_active: np.ndarray,
    expected_bursts: int,
    slot_t: np.ndarray,
    slot_y: np.ndarray,
) -> tuple[float, float, int]:
    median = float(np.median(measured_delta_active))
    sigma = robust_sigma(measured_delta_active)
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1e-12

    best = None
    target_min = int(np.ceil(expected_bursts * MIN_DETECTED_FRACTION))

    for multiplier in DETECTION_THRESHOLD_MAD_MULTIPLIERS:
        threshold = median + multiplier * sigma
        detected = int(np.sum(slot_y >= threshold))
        score = abs(detected - expected_bursts)
        candidate = (score, detected < target_min, threshold, multiplier, detected)
        if best is None or candidate < best:
            best = candidate
        if detected >= target_min:
            break

    _, _, threshold, multiplier, detected = best
    return float(threshold), float(multiplier), int(detected)


def detect_burst_centers_by_current(
    active_t: np.ndarray,
    measured_delta_active: np.ndarray,
    interval_s: float,
    expected_bursts: int,
) -> tuple[list[dict], float, float]:
    slot_peaks = []

    for burst_idx in range(expected_bursts):
        slot_start = burst_idx * interval_s
        slot_end = min(slot_start + interval_s, ACTIVE_DURATION_S)
        slot_mask = (active_t >= slot_start) & (active_t < slot_end)
        if not np.any(slot_mask):
            continue

        local_indices = np.flatnonzero(slot_mask)
        local_y = measured_delta_active[slot_mask]
        peak_local = int(np.argmax(local_y))
        peak_idx = int(local_indices[peak_local])

        slot_peaks.append(
            {
                "burst_index": burst_idx,
                "center_active_s": float(active_t[peak_idx]),
                "peak_delta_ma": float(measured_delta_active[peak_idx]),
                "slot_start_active_s": float(slot_start),
                "slot_end_active_s": float(slot_end),
            }
        )

    if not slot_peaks:
        raise ValueError("No interval slots available for burst detection")

    slot_y = np.array([row["peak_delta_ma"] for row in slot_peaks], dtype=float)
    slot_t = np.array([row["center_active_s"] for row in slot_peaks], dtype=float)
    threshold, multiplier, _ = burst_detection_threshold(
        measured_delta_active,
        expected_bursts,
        slot_t,
        slot_y,
    )

    detected = [row for row in slot_peaks if row["peak_delta_ma"] >= threshold]
    return detected, threshold, multiplier


def top5_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    threshold = np.percentile(values, 95)
    return float(np.mean(values[values >= threshold]))


def percentile_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_p5_ma": np.nan,
            f"{prefix}_median_ma": np.nan,
            f"{prefix}_top5_mean_ma": np.nan,
        }
    return {
        f"{prefix}_p5_ma": float(np.percentile(values, 5)),
        f"{prefix}_median_ma": float(np.median(values)),
        f"{prefix}_top5_mean_ma": top5_mean(values),
    }


def analyze_file(condition: str, path: Path) -> tuple[list[dict], dict]:
    cfg = CONDITIONS[condition]
    interval_s = cfg["interval_s"]
    expected_bursts = cfg["expected_bursts"]

    df = read_ppk_csv(path)
    time_s = df["time_s"].to_numpy()
    current_ma = df["current_ma"].to_numpy()

    sync_mid_s = detect_sync_midpoint(time_s, current_ma)
    t_rel = time_s - sync_mid_s

    initial_mask = (
        (t_rel >= INITIAL_IDLE_START_REL_S + PHASE_TRIM_S)
        & (t_rel <= INITIAL_IDLE_END_REL_S - PHASE_TRIM_S)
    )
    active_mask = (t_rel >= ACTIVE_START_REL_S) & (t_rel < ACTIVE_END_REL_S)

    if initial_mask.sum() < 10:
        raise ValueError("Initial idle window does not contain enough samples")
    if active_mask.sum() < 10:
        raise ValueError("Active window does not contain enough samples")

    initial_mean_ma = float(np.mean(current_ma[initial_mask]))
    measured_delta = current_ma - initial_mean_ma

    active_t = t_rel[active_mask] - ACTIVE_START_REL_S
    measured_delta_active = measured_delta[active_mask]

    detected_centers, threshold_ma, threshold_mad_multiplier = detect_burst_centers_by_current(
        active_t,
        measured_delta_active,
        interval_s,
        expected_bursts,
    )

    pred_t = np.arange(0.0, ACTIVE_DURATION_S + PREDICTION_DT_S, PREDICTION_DT_S)
    pred_occ = simulate_ode(pred_t, interval_s, "occupancy")
    pred_pulse = simulate_ode(pred_t, interval_s, "pulse")

    burst_rows = []
    for center_info in detected_centers:
        center_active_s = center_info["center_active_s"]
        burst_mask_active = (
            (active_t >= center_active_s - BURST_HALF_WINDOW_S)
            & (active_t <= center_active_s + BURST_HALF_WINDOW_S)
        )
        if burst_mask_active.sum() < 5:
            continue

        burst_t = active_t[burst_mask_active]
        measured_burst_delta = measured_delta_active[burst_mask_active]
        pred_occ_burst = np.interp(burst_t, pred_t, pred_occ)
        pred_pulse_burst = np.interp(burst_t, pred_t, pred_pulse)

        row = {
            "condition": condition,
            "interval_ms": int(round(interval_s * 1000)),
            "run_id": f"run{natural_run_key(path)}",
            "source_file": str(path),
            "sync_mid_s": sync_mid_s,
            "initial_mean_ma": initial_mean_ma,
            "burst_index": center_info["burst_index"],
            "burst_center_active_s": center_active_s,
            "burst_center_rel_s": center_active_s + ACTIVE_START_REL_S,
            "burst_peak_delta_ma": center_info["peak_delta_ma"],
            "slot_start_active_s": center_info["slot_start_active_s"],
            "slot_end_active_s": center_info["slot_end_active_s"],
            "burst_window_s": 2.0 * BURST_HALF_WINDOW_S,
            "n_samples": int(burst_mask_active.sum()),
            "detection_threshold_ma": threshold_ma,
            "detection_threshold_mad_multiplier": threshold_mad_multiplier,
        }
        row.update(percentile_metrics(measured_burst_delta, "measured_delta"))
        row.update(percentile_metrics(pred_occ_burst, "pred_occ_delta"))
        row.update(percentile_metrics(pred_pulse_burst, "pred_pulse_delta"))
        burst_rows.append(row)

    if not burst_rows:
        raise ValueError(
            f"No burst windows could be extracted. threshold={threshold_ma:.3f} mA"
        )

    run_summary = {
        "condition": condition,
        "interval_ms": int(round(interval_s * 1000)),
        "run_id": f"run{natural_run_key(path)}",
        "source_file": str(path),
        "sync_mid_s": sync_mid_s,
        "initial_mean_ma": initial_mean_ma,
        "detected_bursts": len(burst_rows),
        "expected_bursts": expected_bursts,
        "detection_ratio": len(burst_rows) / expected_bursts,
        "detection_threshold_ma": threshold_ma,
        "detection_threshold_mad_multiplier": threshold_mad_multiplier,
    }

    burst_df = pd.DataFrame(burst_rows)
    metric_cols = [
        c
        for c in burst_df.columns
        if c.endswith("_p5_ma")
        or c.endswith("_median_ma")
        or c.endswith("_top5_mean_ma")
        or c == "burst_peak_delta_ma"
    ]
    for col in metric_cols:
        run_summary[f"{col}_mean"] = float(burst_df[col].mean())
        run_summary[f"{col}_std_across_bursts"] = float(burst_df[col].std(ddof=1))

    return burst_rows, run_summary


def summarize_conditions(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in run_df.columns
        if c.endswith("_p5_ma_mean")
        or c.endswith("_median_ma_mean")
        or c.endswith("_top5_mean_ma_mean")
        or c == "burst_peak_delta_ma_mean"
    ]
    agg = {
        "detected_bursts": ["mean", "std", "min", "max"],
        "detection_ratio": ["mean", "std", "min", "max"],
        "detection_threshold_ma": ["mean", "std"],
    }
    for col in metric_cols:
        agg[col] = ["mean", "std"]

    summary = run_df.groupby("condition", observed=True).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary["interval_ms"] = (
        summary["condition"].astype(str).str.extract(r"(\d+)ms").astype(int)
    )
    summary = summary.sort_values("interval_ms")
    summary.insert(0, "interval_ms", summary.pop("interval_ms"))
    return summary


def make_compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        out = {
            "interval_ms": row["interval_ms"],
            "condition": row["condition"],
            "detected_bursts_mean": f"{row['detected_bursts_mean']:.1f}",
            "detection_ratio_mean": f"{row['detection_ratio_mean']:.3f}",
            "threshold_mean_std_ma": (
                f"{row['detection_threshold_ma_mean']:.3f} +/- "
                f"{row['detection_threshold_ma_std']:.3f}"
            ),
            "measured_peak_mean_std_ma": (
                f"{row['burst_peak_delta_ma_mean_mean']:.3f} +/- "
                f"{row['burst_peak_delta_ma_mean_std']:.3f}"
            ),
        }
        for source in ["measured_delta", "pred_occ_delta", "pred_pulse_delta"]:
            for metric in ["p5", "median", "top5_mean"]:
                mean_col = f"{source}_{metric}_ma_mean_mean"
                std_col = f"{source}_{metric}_ma_mean_std"
                label = f"{source}_{metric}_mean_std_ma"
                out[label] = f"{row[mean_col]:.3f} +/- {row[std_col]:.3f}"
        rows.append(out)
    return pd.DataFrame(rows)


def plot_burst_metric(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    conditions = INTERVAL_ORDER
    x = np.arange(len(conditions))
    width = 0.25

    sources = [
        ("measured_delta", "Measured"),
        ("pred_occ_delta", "Predicted occupancy"),
        ("pred_pulse_delta", "Predicted pulse"),
    ]

    for idx, (source, label) in enumerate(sources):
        means = []
        stds = []
        for condition in conditions:
            row = summary[summary["condition"] == condition].iloc[0]
            means.append(row[f"{source}_{metric}_ma_mean_mean"])
            stds.append(row[f"{source}_{metric}_ma_mean_std"])

        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([INTERVAL_LABELS[c] for c in conditions])
    ax.set_xlabel("Advertising interval")
    ax.set_ylabel("Burst delta current (mA)")
    ax.set_title(f"Burst {metric}: measured vs predicted")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_measured_burst_levels(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    conditions = INTERVAL_ORDER
    x = np.arange(len(conditions))
    width = 0.25

    metrics = [
        ("p5", "p5"),
        ("median", "median"),
        ("top5_mean", "top5 mean"),
    ]

    for idx, (metric, label) in enumerate(metrics):
        means = []
        stds = []
        for condition in conditions:
            row = summary[summary["condition"] == condition].iloc[0]
            means.append(row[f"measured_delta_{metric}_ma_mean_mean"])
            stds.append(row[f"measured_delta_{metric}_ma_mean_std"])

        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([INTERVAL_LABELS[c] for c in conditions])
    ax.set_xlabel("Advertising interval")
    ax.set_ylabel("Measured burst delta current (mA)")
    ax.set_title("Measured burst p5 / median / top5 mean")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_detection_overlay(burst_df: pd.DataFrame, output_dir: Path) -> None:
    debug_dir = output_dir / "detection_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    zoom_s_by_condition = {
        "ble_adv_100ms": 2.0,
        "ble_adv_500ms": 5.0,
        "ble_adv_1000ms": 10.0,
    }

    for condition in INTERVAL_ORDER:
        condition_rows = burst_df[burst_df["condition"] == condition].copy()
        if condition_rows.empty:
            continue

        run_ids = sorted(
            condition_rows["run_id"].unique(),
            key=lambda s: int(str(s).replace("run", "")),
        )
        run_id = run_ids[0]
        rows = condition_rows[condition_rows["run_id"] == run_id].copy()
        if rows.empty:
            continue

        source_file = Path(rows["source_file"].iloc[0])
        sync_mid_s = float(rows["sync_mid_s"].iloc[0])
        initial_mean_ma = float(rows["initial_mean_ma"].iloc[0])

        raw = read_ppk_csv(source_file)
        t_rel = raw["time_s"].to_numpy() - sync_mid_s
        current_delta = raw["current_ma"].to_numpy() - initial_mean_ma
        active_t = t_rel - ACTIVE_START_REL_S

        active_mask = (active_t >= 0.0) & (active_t <= ACTIVE_DURATION_S)
        if not np.any(active_mask):
            continue

        zoom_s = zoom_s_by_condition[condition]
        zoom_mask = active_mask & (active_t <= zoom_s)

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(
            active_t[zoom_mask],
            current_delta[zoom_mask],
            color="black",
            linewidth=0.8,
            label="Measured delta current",
        )

        visible_rows = rows[
            (rows["burst_center_active_s"] >= 0.0)
            & (rows["burst_center_active_s"] <= zoom_s)
        ]

        for _, row in visible_rows.iterrows():
            center = float(row["burst_center_active_s"])
            half = float(row["burst_window_s"]) / 2.0
            ax.axvspan(center - half, center + half, color="tab:orange", alpha=0.22)
            ax.axvline(center, color="tab:red", linewidth=0.8, alpha=0.75)

        threshold = float(rows["detection_threshold_ma"].iloc[0])
        ax.axhline(
            threshold,
            color="tab:purple",
            linestyle="--",
            linewidth=0.9,
            label="Detection threshold",
        )

        ax.set_title(
            f"Current-detected burst windows: {INTERVAL_LABELS[condition]}, "
            f"{run_id}, first {zoom_s:g}s"
        )
        ax.set_xlabel("Time from active start (s)")
        ax.set_ylabel("Measured delta current (mA)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(debug_dir / f"{condition}_{run_id}_detected_bursts_zoom.png", dpi=200)
        plt.close(fig)

        first_center = float(rows["burst_center_active_s"].iloc[0])
        event_mask = (
            active_mask
            & (active_t >= first_center - 0.030)
            & (active_t <= first_center + 0.030)
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(
            active_t[event_mask],
            current_delta[event_mask],
            color="black",
            linewidth=0.9,
            label="Measured delta current",
        )

        half = float(rows["burst_window_s"].iloc[0]) / 2.0
        ax.axvspan(first_center - half, first_center + half, color="tab:orange", alpha=0.25)
        ax.axvline(first_center, color="tab:red", linewidth=0.9, label="Detected center")
        ax.axhline(
            float(rows["detection_threshold_ma"].iloc[0]),
            color="tab:purple",
            linestyle="--",
            linewidth=0.9,
            label="Detection threshold",
        )

        ax.set_title(
            f"First current-detected burst detail: {INTERVAL_LABELS[condition]}, {run_id}"
        )
        ax.set_xlabel("Time from active start (s)")
        ax.set_ylabel("Measured delta current (mA)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(debug_dir / f"{condition}_{run_id}_first_burst_detail.png", dpi=200)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_bursts = []
    run_summaries = []
    errors = []

    for condition in INTERVAL_ORDER:
        cfg = CONDITIONS[condition]
        files = sorted(cfg["input_dir"].glob(cfg["pattern"]), key=natural_run_key)
        if not files:
            errors.append(
                {
                    "condition": condition,
                    "source_file": str(cfg["input_dir"] / cfg["pattern"]),
                    "error": "No input files found",
                }
            )
            continue

        for path in files:
            try:
                burst_rows, run_summary = analyze_file(condition, path)
                all_bursts.extend(burst_rows)
                run_summaries.append(run_summary)
            except Exception as exc:
                errors.append(
                    {
                        "condition": condition,
                        "source_file": str(path),
                        "error": str(exc),
                    }
                )

    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / "errors.csv", index=False)

    if not all_bursts:
        print("No burst data could be analyzed. See errors.csv.")
        return

    burst_df = pd.DataFrame(all_bursts)
    run_df = pd.DataFrame(run_summaries)

    burst_df.to_csv(OUTPUT_DIR / "per_burst_percentiles.csv", index=False)
    run_df.to_csv(OUTPUT_DIR / "per_run_burst_percentiles.csv", index=False)

    condition_summary = summarize_conditions(run_df)
    compact_summary = make_compact_summary(condition_summary)

    condition_summary.to_csv(OUTPUT_DIR / "summary_burst_percentiles.csv", index=False)
    compact_summary.to_csv(
        OUTPUT_DIR / "summary_burst_percentiles_compact.csv", index=False
    )

    plot_burst_metric(condition_summary, "p5", OUTPUT_DIR / "burst_p5_comparison.png")
    plot_burst_metric(
        condition_summary, "median", OUTPUT_DIR / "burst_median_comparison.png"
    )
    plot_burst_metric(
        condition_summary, "top5_mean", OUTPUT_DIR / "burst_top5_comparison.png"
    )
    plot_measured_burst_levels(
        condition_summary, OUTPUT_DIR / "measured_burst_levels.png"
    )
    plot_detection_overlay(burst_df, OUTPUT_DIR)

    print("Wrote:")
    for name in [
        "per_burst_percentiles.csv",
        "per_run_burst_percentiles.csv",
        "summary_burst_percentiles.csv",
        "summary_burst_percentiles_compact.csv",
        "burst_p5_comparison.png",
        "burst_median_comparison.png",
        "burst_top5_comparison.png",
        "measured_burst_levels.png",
        "detection_debug",
    ]:
        print(f"  {OUTPUT_DIR / name}")
    if errors:
        print(f"  {OUTPUT_DIR / 'errors.csv'}")

    print()
    print(compact_summary.to_string(index=False))


if __name__ == "__main__":
    main()