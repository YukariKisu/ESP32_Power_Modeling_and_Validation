from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_ROOT = Path('data/raw/v3_ppk/peripheral/adc/final_predictioned')

INPUT_DIRS = {
    'adc_single_1ms': RAW_ROOT / 'adc_periodic_single' / 'adc_single_1ms',
    'adc_single_10ms': RAW_ROOT / 'adc_periodic_single' / 'adc_single_10ms',
    'adc_single_100ms': RAW_ROOT / 'adc_periodic_single' / 'adc_single_100ms',
    'adc_burst_100ms_10samples': RAW_ROOT / 'adc_periodic_burst' / 'adc_burst_100ms_10samples',
    'adc_burst_100ms_100samples': RAW_ROOT / 'adc_periodic_burst' / 'adc_burst_100ms_100samples',
    'adc_burst_100ms_1000samples': RAW_ROOT / 'adc_periodic_burst' / 'adc_burst_100ms_1000samples',
}

OUTPUT_DIR = Path(
    'results/v3_ppk/peripheral/adc/final_predictioned/ppk_vs_scope_validation'
)

REFERENCE_I_IDLE_MA = 45.580
REFERENCE_I_ACTIVE_MA = 58.524

MODEL_PARAMETERS = {
    'ppk2': {
        'delta_i_ma': 12.943,
        'tau_rise_s': 0.000579,
        'tau_fall_s': 0.000579,
    },
    'scope': {
        'delta_i_ma': 12.000,
        'tau_rise_s': 0.000440,
        'tau_fall_s': 0.000440,
    },
}

ACTIVE_DURATION_S = 20.0

EVENT_DURATION_S = {
    'adc_single_1ms': 44e-6,
    'adc_single_10ms': 44e-6,
    'adc_single_100ms': 44e-6,
    'adc_burst_100ms_10samples': 537e-6,
    'adc_burst_100ms_100samples': 4397e-6,
    'adc_burst_100ms_1000samples': 43987e-6,
}

EVENT_INTERVAL_S = {
    'adc_single_1ms': 0.001,
    'adc_single_10ms': 0.010,
    'adc_single_100ms': 0.100,
    'adc_burst_100ms_10samples': 0.100,
    'adc_burst_100ms_100samples': 0.100,
    'adc_burst_100ms_1000samples': 0.100,
}

# Phase windows relative to the ACTUAL measured sync-start edge.
# Firmware timeline:
#   sync:         0.0–1.0 s
#   recovery:     1.0–6.0 s
#   initial idle: 6.0–16.0 s
#   ADC active:  16.0–36.0 s
#   final idle:  36.0–46.0 s
INITIAL_START_S = 6.0
INITIAL_END_S = 16.0
ACTIVE_START_S = 16.0
ACTIVE_END_S = 36.0
FINAL_START_S = 36.0
FINAL_END_S = 46.0
WINDOW_TRIM_S = 0.2
UNIFORM_DT_S = 0.00001

SYNC_EXPECTED_START_S = 3.0
SYNC_DURATION_S = 1.0
SYNC_SEARCH_MARGIN_S = 1.5
SYNC_SMOOTH_WINDOW_S = 0.02

TIME_COLUMN_CANDIDATES = (
    'time', 'timestamp', 'timestamp_s', 'time_s', 'time (s)', 'time[s]', 'timestamp(ms)'
)
CURRENT_COLUMN_CANDIDATES = (
    'current', 'current_ma', 'current (ma)', 'current[ma]',
    'current_ua', 'current (ua)', 'current[ua]',
    'current_a', 'current (a)', 'current[a]'
)


@dataclass(frozen=True)
class RunResult:
    condition: str
    model_name: str
    run_id: str
    source_file: str
    sync_start_s: float
    initial_mean_ma: float
    final_mean_ma: float
    adc_idle_mean_ma: float
    model_delta_i_ma: float
    model_tau_rise_ms: float
    model_tau_fall_ms: float
    measured_active_delta_mean_ma: float
    predicted_active_delta_mean_ma: float
    mae_ma: float
    me_ma: float
    rmse_ma: float


def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        candidate_normalized = normalize_name(candidate)
        if candidate_normalized in normalized:
            return normalized[candidate_normalized]
    for column in columns:
        column_normalized = normalize_name(column)
        if kind == 'time' and 'time' in column_normalized:
            return column
        if kind == 'current' and 'current' in column_normalized:
            return column
    raise ValueError(f'Could not find {kind} column in columns: {list(columns)}')


def convert_time_to_s(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if 'ms' in name:
        return values / 1000.0
    if 'us' in name or 'µs' in name:
        return values / 1_000_000.0
    return values


def convert_current_to_ma(values: np.ndarray, column_name: str) -> np.ndarray:
    name = normalize_name(column_name)
    if 'ua' in name or 'µa' in name or 'micro' in name:
        return values / 1000.0
    if 'ma' in name:
        return values
    if re.search(r'\ba\b', name) or '(a' in name or '[a' in name:
        return values * 1000.0

    median_abs = float(np.nanmedian(np.abs(values)))
    if median_abs < 1.0:
        return values * 1000.0
    if median_abs > 1000.0:
        return values / 1000.0
    return values


def read_ppk_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, comment='#')
    if dataframe.empty:
        raise ValueError(f'CSV is empty: {path}')

    time_column = find_column(dataframe.columns, TIME_COLUMN_CANDIDATES, 'time')
    current_column = find_column(dataframe.columns, CURRENT_COLUMN_CANDIDATES, 'current')

    time_raw = pd.to_numeric(dataframe[time_column], errors='coerce').to_numpy(dtype=float)
    current_raw = pd.to_numeric(dataframe[current_column], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(time_raw) & np.isfinite(current_raw)
    time_raw = time_raw[valid]
    current_raw = current_raw[valid]

    if len(time_raw) < 10:
        raise ValueError(f'Not enough valid samples in {path}')

    time_s = convert_time_to_s(time_raw, time_column)
    time_s = time_s - time_s[0]
    current_ma = convert_current_to_ma(current_raw, current_column)
    return pd.DataFrame({'time_s': time_s, 'current_ma': current_ma})


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(values, kernel, mode='same')


def detect_sync_start(dataframe: pd.DataFrame) -> float:
    """Detect the actual measured start edge of the 1 s CPU sync pulse."""
    time_s = dataframe['time_s'].to_numpy()
    current_ma = dataframe['current_ma'].to_numpy()

    dt_s = float(np.median(np.diff(time_s)))
    smooth_samples = max(1, round(SYNC_SMOOTH_WINDOW_S / dt_s))
    smooth_current = rolling_mean(current_ma, smooth_samples)

    search_start_s = max(0.0, SYNC_EXPECTED_START_S - SYNC_SEARCH_MARGIN_S)
    search_end_s = SYNC_EXPECTED_START_S + SYNC_DURATION_S + SYNC_SEARCH_MARGIN_S
    search_mask = (time_s >= search_start_s) & (time_s <= search_end_s)

    if not np.any(search_mask):
        raise ValueError('Sync-pulse search window does not contain samples')

    search_time = time_s[search_mask]
    search_current = smooth_current[search_mask]

    baseline = float(np.percentile(search_current, 20))
    peak = float(np.percentile(search_current, 99))
    threshold = baseline + 0.45 * (peak - baseline)

    high = search_current >= threshold

    segments: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, is_high in enumerate(high):
        if is_high and start_index is None:
            start_index = index
        elif not is_high and start_index is not None:
            segments.append((start_index, index - 1))
            start_index = None

    if start_index is not None:
        segments.append((start_index, len(high) - 1))

    if not segments:
        raise ValueError('No sync-like high-current segment detected')

    def score(segment: tuple[int, int]) -> tuple[float, float]:
        s, e = segment
        duration_s = float(search_time[e] - search_time[s])
        duration_error_s = abs(duration_s - SYNC_DURATION_S)
        strength_ma = float(np.mean(search_current[s:e + 1]) - baseline)
        return duration_error_s, -strength_ma

    best_start_index, _ = min(segments, key=score)

    if best_start_index > 0:
        i0 = best_start_index - 1
        i1 = best_start_index
        y0 = search_current[i0]
        y1 = search_current[i1]
        t0 = search_time[i0]
        t1 = search_time[i1]

        if y1 != y0:
            fraction = (threshold - y0) / (y1 - y0)
            return float(t0 + fraction * (t1 - t0))

    return float(search_time[best_start_index])


def infer_run_id(path: Path) -> str:
    match = re.search(r'(run[_-]?\d+|\d+)$', path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else path.stem


def make_pulse_input(condition: str, time_s: np.ndarray) -> np.ndarray:
    interval_s = EVENT_INTERVAL_S[condition]
    event_duration_s = EVENT_DURATION_S[condition]
    phase_s = np.mod(time_s, interval_s)
    return (phase_s < event_duration_s).astype(float)


def simulate_first_order_ode(
    input_u: np.ndarray,
    dt_s: float,
    delta_i_ma: float,
    tau_rise_s: float,
    tau_fall_s: float,
) -> np.ndarray:
    prediction_delta_ma = np.zeros_like(input_u, dtype=float)
    for index in range(1, len(input_u)):
        target_delta_ma = delta_i_ma * input_u[index]
        tau_s = (
            tau_rise_s
            if target_delta_ma >= prediction_delta_ma[index - 1]
            else tau_fall_s
        )
        alpha = 1.0 - math.exp(-dt_s / tau_s)
        prediction_delta_ma[index] = (
            prediction_delta_ma[index - 1]
            + alpha * (target_delta_ma - prediction_delta_ma[index - 1])
        )
    return prediction_delta_ma


def calculate_phase_mean(dataframe: pd.DataFrame, start_s: float, end_s: float) -> float:
    phase_mask = (
        (dataframe['aligned_time_s'] >= start_s + WINDOW_TRIM_S)
        & (dataframe['aligned_time_s'] <= end_s - WINDOW_TRIM_S)
    )
    if not np.any(phase_mask):
        raise ValueError(f'No samples in phase {start_s}–{end_s} s')
    return float(dataframe.loc[phase_mask, 'current_ma'].mean())


def active_uniform_measurement(dataframe: pd.DataFrame):
    initial_mean_ma = calculate_phase_mean(dataframe, INITIAL_START_S, INITIAL_END_S)
    final_mean_ma = calculate_phase_mean(dataframe, FINAL_START_S, FINAL_END_S)
    adc_idle_mean_ma = (initial_mean_ma + final_mean_ma) / 2.0

    active_mask = (
        (dataframe['aligned_time_s'] >= ACTIVE_START_S)
        & (dataframe['aligned_time_s'] <= ACTIVE_END_S)
    )
    active_time_s = (
        dataframe.loc[active_mask, 'aligned_time_s'].to_numpy() - ACTIVE_START_S
    )
    active_current_ma = dataframe.loc[active_mask, 'current_ma'].to_numpy()
    measured_delta_ma = active_current_ma - adc_idle_mean_ma

    if len(active_time_s) < 10:
        raise ValueError('Active window does not contain enough samples')

    uniform_time_s = np.arange(0.0, ACTIVE_DURATION_S, UNIFORM_DT_S)
    uniform_measured_delta_ma = np.interp(
        uniform_time_s, active_time_s, measured_delta_ma
    )
    return (
        uniform_time_s,
        uniform_measured_delta_ma,
        initial_mean_ma,
        final_mean_ma,
        adc_idle_mean_ma,
    )


def analyze_one_file(condition: str, path: Path) -> list[RunResult]:
    dataframe = read_ppk_csv(path)
    sync_start_s = detect_sync_start(dataframe)
    dataframe['aligned_time_s'] = dataframe['time_s'] - sync_start_s

    (
        time_s,
        measured_delta_ma,
        initial_mean_ma,
        final_mean_ma,
        adc_idle_mean_ma,
    ) = active_uniform_measurement(dataframe)

    input_u = make_pulse_input(condition, time_s)
    run_id = infer_run_id(path)
    results: list[RunResult] = []

    for model_name, parameters in MODEL_PARAMETERS.items():
        predicted_delta_ma = simulate_first_order_ode(
            input_u=input_u,
            dt_s=UNIFORM_DT_S,
            delta_i_ma=parameters['delta_i_ma'],
            tau_rise_s=parameters['tau_rise_s'],
            tau_fall_s=parameters['tau_fall_s'],
        )

        # Positive ME = overprediction, negative ME = underprediction.
        error_ma = predicted_delta_ma - measured_delta_ma

        results.append(
            RunResult(
                condition=condition,
                model_name=model_name,
                run_id=run_id,
                source_file=str(path),
                sync_start_s=sync_start_s,
                initial_mean_ma=initial_mean_ma,
                final_mean_ma=final_mean_ma,
                adc_idle_mean_ma=adc_idle_mean_ma,
                model_delta_i_ma=parameters['delta_i_ma'],
                model_tau_rise_ms=parameters['tau_rise_s'] * 1000.0,
                model_tau_fall_ms=parameters['tau_fall_s'] * 1000.0,
                measured_active_delta_mean_ma=float(np.mean(measured_delta_ma)),
                predicted_active_delta_mean_ma=float(np.mean(predicted_delta_ma)),
                mae_ma=float(np.mean(np.abs(error_ma))),
                me_ma=float(np.mean(error_ma)),
                rmse_ma=float(np.sqrt(np.mean(error_ma * error_ma))),
            )
        )

    return results


def collect_csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    return sorted(path for path in input_dir.rglob('*.csv') if path.is_file())


def summarize(per_run_dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_run_dataframe.groupby(['condition', 'model_name'], sort=True)

    for (condition, model_name), group in grouped:
        rows.append({
            'condition': condition,
            'model_name': model_name,
            'n_runs': int(len(group)),
            'model_delta_i_ma': float(group['model_delta_i_ma'].iloc[0]),
            'model_tau_rise_ms': float(group['model_tau_rise_ms'].iloc[0]),
            'model_tau_fall_ms': float(group['model_tau_fall_ms'].iloc[0]),
            'mae_mean_ma': float(group['mae_ma'].mean()),
            'mae_std_ma': float(group['mae_ma'].std(ddof=1)) if len(group) > 1 else 0.0,
            'me_mean_ma': float(group['me_ma'].mean()),
            'me_std_ma': float(group['me_ma'].std(ddof=1)) if len(group) > 1 else 0.0,
            'rmse_mean_ma': float(group['rmse_ma'].mean()),
            'rmse_std_ma': float(group['rmse_ma'].std(ddof=1)) if len(group) > 1 else 0.0,
            'measured_delta_mean_ma': float(group['measured_active_delta_mean_ma'].mean()),
            'measured_delta_std_ma': float(group['measured_active_delta_mean_ma'].std(ddof=1)) if len(group) > 1 else 0.0,
            'predicted_delta_mean_ma': float(group['predicted_active_delta_mean_ma'].mean()),
            'predicted_delta_std_ma': float(group['predicted_active_delta_mean_ma'].std(ddof=1)) if len(group) > 1 else 0.0,
        })

    return pd.DataFrame(rows)


def build_model_parameter_table() -> pd.DataFrame:
    rows = []
    for model_name, parameters in MODEL_PARAMETERS.items():
        rows.append({
            'model_name': model_name,
            'reference_i_idle_ma': REFERENCE_I_IDLE_MA,
            'reference_i_active_ma': REFERENCE_I_ACTIVE_MA,
            'delta_i_ma': parameters['delta_i_ma'],
            'tau_rise_ms': parameters['tau_rise_s'] * 1000.0,
            'tau_fall_ms': parameters['tau_fall_s'] * 1000.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    all_results: list[RunResult] = []
    errors: list[dict[str, str]] = []

    for condition, input_dir in INPUT_DIRS.items():
        for path in collect_csv_files(input_dir):
            try:
                print(f'Processing {condition}: {path}', flush=True)
                all_results.extend(analyze_one_file(condition, path))
            except Exception as exception:
                errors.append({
                    'condition': condition,
                    'source_file': str(path),
                    'error': str(exception),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not all_results:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / 'errors.csv', index=False)
        raise SystemExit('No runs could be analyzed. See errors.csv.')

    per_run_dataframe = pd.DataFrame([result.__dict__ for result in all_results])
    summary_dataframe = summarize(per_run_dataframe)
    model_parameter_dataframe = build_model_parameter_table()

    per_run_path = OUTPUT_DIR / 'per_run_metrics.csv'
    summary_path = OUTPUT_DIR / 'summary_metrics.csv'
    model_parameter_path = OUTPUT_DIR / 'model_parameters.csv'

    per_run_dataframe.to_csv(per_run_path, index=False)
    summary_dataframe.to_csv(summary_path, index=False)
    model_parameter_dataframe.to_csv(model_parameter_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT_DIR / 'errors.csv', index=False)

    print(f'Analyzed validation rows: {len(per_run_dataframe)}')
    print(f'Skipped files: {len(errors)}')
    print(f'Wrote: {per_run_path}')
    print(f'Wrote: {summary_path}')
    print(f'Wrote: {model_parameter_path}')


if __name__ == '__main__':
    main()