from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


INPUT_ROOT = Path("data/processed/v3_ppk/peripheral/adc/oscilloscope")
OUTPUT_ROOT = INPUT_ROOT / "analysis"
FILE_PATTERN = "*_current_ma.csv"

PLATEAU_FRACTION = 0.20
SMOOTH_WINDOW = 7

PPK2_IDLE_MA = 45.58040756473625
PPK2_ACTIVE_MA = 58.52386984335895
PPK2_DELTA_MA = 12.943462278622704
PPK2_RISE_TAU_MS = 0.5791261523155534
PPK2_FALL_TAU_MS = 3.2034973


def smooth(y: np.ndarray, requested: int = SMOOTH_WINDOW) -> np.ndarray:
    n = len(y)
    w = min(requested, n if n % 2 else n - 1)
    w = max(w, 1)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return y.copy()
    return (
        pd.Series(y)
        .rolling(w, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )


def infer_direction(path: Path) -> str:
    text = str(path).lower()
    if "rise" in text:
        return "rise"
    if "fall" in text:
        return "fall"
    raise ValueError(f"Cannot infer rise/fall from: {path}")


def infer_scale_us(path: Path) -> int:
    m = re.search(r"(100|200|500)us", str(path).lower())
    if not m:
        raise ValueError(f"Cannot infer time scale from: {path}")
    return int(m.group(1))


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    needed = {"time_s", "current_ma"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"Required columns are time_s and current_ma. Found: {list(df.columns)}"
        )

    df = (
        df[["time_s", "current_ma"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("time_s")
        .drop_duplicates("time_s")
    )

    if len(df) < 20:
        raise ValueError(f"Too few valid samples: {len(df)}")

    t = df["time_s"].to_numpy(float)
    y = df["current_ma"].to_numpy(float)
    t = t - t[0]
    return t, y


def estimate_plateaus(y: np.ndarray, direction: str) -> dict[str, float]:
    n = len(y)
    count = max(5, int(round(n * PLATEAU_FRACTION)))
    count = min(count, max(5, n // 3))

    start = y[:count]
    end = y[-count:]

    start_level = float(np.median(start))
    end_level = float(np.median(end))
    start_std = float(np.std(start, ddof=1))
    end_std = float(np.std(end, ddof=1))

    if direction == "rise":
        idle, active = start_level, end_level
        idle_std, active_std = start_std, end_std
    else:
        active, idle = start_level, end_level
        active_std, idle_std = start_std, end_std

    delta = active - idle
    if delta <= 0:
        raise ValueError(
            f"Estimated delta is not positive: {delta:.6f} mA. "
            "The capture may not include enough plateau."
        )

    return {
        "idle_ma": idle,
        "active_ma": active,
        "delta_ma": delta,
        "idle_std_ma": idle_std,
        "active_std_ma": active_std,
        "plateau_samples_each_side": count,
    }


def find_crossing(
    t: np.ndarray,
    y: np.ndarray,
    threshold: float,
    direction: str,
) -> float:
    if direction == "rise":
        idxs = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    else:
        idxs = np.where((y[:-1] > threshold) & (y[1:] <= threshold))[0]

    if len(idxs) == 0:
        return float(t[np.argmin(np.abs(y - threshold))])

    jumps = np.abs(y[idxs + 1] - y[idxs])
    i = int(idxs[np.argmax(jumps)])

    t1, t2 = t[i], t[i + 1]
    y1, y2 = y[i], y[i + 1]

    if y2 == y1:
        return float(t1)

    return float(t1 + (threshold - y1) * (t2 - t1) / (y2 - y1))


def rise_response(
    t: np.ndarray,
    t0: float,
    tau: float,
    idle: float,
    delta: float,
) -> np.ndarray:
    elapsed = np.maximum(t - t0, 0.0)
    return idle + delta * (1.0 - np.exp(-elapsed / tau))


def fall_response(
    t: np.ndarray,
    t0: float,
    tau: float,
    idle: float,
    delta: float,
) -> np.ndarray:
    elapsed = np.maximum(t - t0, 0.0)
    return idle + delta * np.exp(-elapsed / tau)


def fit_tau(
    t_aligned: np.ndarray,
    y: np.ndarray,
    direction: str,
    idle: float,
    delta: float,
) -> dict[str, object]:
    dt = float(np.median(np.diff(t_aligned)))
    span = float(t_aligned[-1] - t_aligned[0])

    if direction == "rise":
        def model(t, t0, tau):
            return rise_response(t, t0, tau, idle, delta)
    else:
        def model(t, t0, tau):
            return fall_response(t, t0, tau, idle, delta)

    tau_guess = max(span / 10.0, 2.0 * dt)
    t0_guess = -tau_guess * np.log(2.0)

    tau_min = max(0.1 * dt, 1e-9)
    tau_max = max(5.0 * span, 10.0 * tau_min)
    t0_min = float(t_aligned[0] - span)
    t0_max = float(t_aligned[-1] + span)

    popt, pcov = curve_fit(
        model,
        t_aligned,
        y,
        p0=[t0_guess, tau_guess],
        bounds=([t0_min, tau_min], [t0_max, tau_max]),
        maxfev=50000,
    )

    t0, tau = map(float, popt)
    predicted = model(t_aligned, t0, tau)
    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))

    if pcov.shape == (2, 2) and np.all(np.isfinite(pcov)):
        se = np.sqrt(np.diag(pcov))
        t0_se, tau_se = float(se[0]), float(se[1])
    else:
        t0_se, tau_se = np.nan, np.nan

    return {
        "t0_s": t0,
        "tau_s": tau,
        "t0_se_s": t0_se,
        "tau_se_s": tau_se,
        "predicted_ma": predicted,
        "rmse_ma": rmse,
    }


def analyze_one(path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    direction = infer_direction(path)
    scale_us = infer_scale_us(path)

    t, y_raw = load_csv(path)
    y_smoothed = smooth(y_raw)

    plateau = estimate_plateaus(y_smoothed, direction)
    i50 = plateau["idle_ma"] + 0.5 * plateau["delta_ma"]

    crossing = find_crossing(t, y_smoothed, i50, direction)
    t_aligned = t - crossing

    fit = fit_tau(
        t_aligned,
        y_smoothed,
        direction,
        plateau["idle_ma"],
        plateau["delta_ma"],
    )

    result = {
        "file": str(path),
        "filename": path.name,
        "direction": direction,
        "scale_us_per_div": scale_us,
        "n_samples": len(t),
        "sample_interval_us": float(np.median(np.diff(t)) * 1e6),
        "capture_span_ms": float((t[-1] - t[0]) * 1e3),
        **plateau,
        "i50_ma": i50,
        "crossing_time_original_ms": crossing * 1e3,
        "t0_aligned_ms": fit["t0_s"] * 1e3,
        "tau_ms": fit["tau_s"] * 1e3,
        "t0_se_ms": fit["t0_se_s"] * 1e3,
        "tau_se_ms": fit["tau_se_s"] * 1e3,
        "fit_rmse_ma": fit["rmse_ma"],
    }

    waveform = pd.DataFrame({
        "time_aligned_ms": t_aligned * 1e3,
        "current_raw_ma": y_raw,
        "current_smoothed_ma": y_smoothed,
        "current_fit_ma": fit["predicted_ma"],
        "filename": path.name,
        "direction": direction,
        "scale_us_per_div": scale_us,
    })

    return result, waveform


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "idle_ma",
        "active_ma",
        "delta_ma",
        "tau_ms",
        "fit_rmse_ma",
        "sample_interval_us",
        "capture_span_ms",
    ]

    rows = []
    for (direction, scale), group in results.groupby(
        ["direction", "scale_us_per_div"], sort=True
    ):
        row = {
            "direction": direction,
            "scale_us_per_div": scale,
            "n_runs": len(group),
        }

        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1
                else np.nan
            )

        ppk_tau = (
            PPK2_RISE_TAU_MS
            if direction == "rise"
            else PPK2_FALL_TAU_MS
        )

        row.update({
            "ppk2_idle_ma": PPK2_IDLE_MA,
            "ppk2_active_ma": PPK2_ACTIVE_MA,
            "ppk2_delta_ma": PPK2_DELTA_MA,
            "ppk2_tau_ms": ppk_tau,
            "idle_difference_vs_ppk2_ma":
                row["idle_ma_mean"] - PPK2_IDLE_MA,
            "active_difference_vs_ppk2_ma":
                row["active_ma_mean"] - PPK2_ACTIVE_MA,
            "delta_difference_vs_ppk2_ma":
                row["delta_ma_mean"] - PPK2_DELTA_MA,
            "tau_difference_vs_ppk2_ms":
                row["tau_ms_mean"] - ppk_tau,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def plot_group(
    waveforms: list[pd.DataFrame],
    group_results: pd.DataFrame,
    direction: str,
    scale_us: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for wf in waveforms:
        name = Path(wf["filename"].iloc[0]).stem
        ax.plot(
            wf["time_aligned_ms"],
            wf["current_smoothed_ma"],
            linewidth=1.2,
            label=f"{name} measured",
        )
        ax.plot(
            wf["time_aligned_ms"],
            wf["current_fit_ma"],
            linestyle="--",
            linewidth=1.0,
            label=f"{name} fit",
        )

    tau_mean = group_results["tau_ms"].mean()
    tau_sd = group_results["tau_ms"].std(ddof=1)
    delta_mean = group_results["delta_ma"].mean()
    delta_sd = group_results["delta_ma"].std(ddof=1)

    ax.axvline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Aligned time from 50% crossing [ms]")
    ax.set_ylabel("Current [mA]")
    ax.set_title(
        f"ADC {direction.capitalize()} — {scale_us} µs/div\n"
        f"tau = {tau_mean:.4f} ± {tau_sd:.4f} ms, "
        f"delta I = {delta_mean:.3f} ± {delta_sd:.3f} mA"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not INPUT_ROOT.exists():
        raise SystemExit(f"Input root does not exist: {INPUT_ROOT}")

    files = [
        p for p in sorted(INPUT_ROOT.rglob(FILE_PATTERN))
        if OUTPUT_ROOT not in p.parents
    ]

    if not files:
        raise SystemExit(
            f"No files matching {FILE_PATTERN} found under {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    waveforms_by_group = {}

    print(f"Found {len(files)} files.\n")

    for path in files:
        try:
            result, waveform = analyze_one(path)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
            continue

        results.append(result)
        key = (result["direction"], result["scale_us_per_div"])
        waveforms_by_group.setdefault(key, []).append(waveform)

        print(
            f"[OK] {path.name}: "
            f"idle={result['idle_ma']:.3f} mA, "
            f"active={result['active_ma']:.3f} mA, "
            f"delta={result['delta_ma']:.3f} mA, "
            f"tau={result['tau_ms']:.4f} ms, "
            f"RMSE={result['fit_rmse_ma']:.3f} mA"
        )

    if not results:
        raise SystemExit("No file was successfully analyzed.")

    results_df = pd.DataFrame(results).sort_values(
        ["direction", "scale_us_per_div", "filename"]
    )

    per_run_path = OUTPUT_ROOT / "oscilloscope_transition_analysis_by_run.csv"
    results_df.to_csv(per_run_path, index=False)

    summary_df = make_summary(results_df)
    summary_path = OUTPUT_ROOT / "oscilloscope_transition_summary_by_scale.csv"
    summary_df.to_csv(summary_path, index=False)

    for (direction, scale), waveforms in waveforms_by_group.items():
        group_results = results_df[
            (results_df["direction"] == direction)
            & (results_df["scale_us_per_div"] == scale)
        ]

        plot_group(
            waveforms,
            group_results,
            direction,
            scale,
            plots_dir / f"{direction}_{scale}us_aligned_runs_and_fit.png",
        )

    print("\nSaved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {plots_dir}")


if __name__ == "__main__":
    main()