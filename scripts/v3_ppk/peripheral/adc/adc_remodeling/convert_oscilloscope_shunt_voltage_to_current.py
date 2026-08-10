from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_ROOT = Path(
    "data/raw/v3_ppk/peripheral/adc/oscilloscope"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data/processed/v3_ppk/peripheral/adc/oscilloscope"
)


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_time_column(columns: list[str]) -> str:
    for col in columns:
        normalized = normalize(col)

        if normalized in {
            "s",
            "sec",
            "second",
            "seconds",
            "time",
            "times",
            "timestamps",
        }:
            return col

    for col in columns:
        if "time" in normalize(col):
            return col

    return columns[0]


def detect_voltage_column(columns: list[str]) -> str:
    for col in columns:
        normalized = normalize(col)

        if "ch1" in normalized and (
            "v" in normalized or "volt" in normalized
        ):
            return col

    for col in columns:
        normalized = normalize(col)

        if "volt" in normalized or normalized.endswith("v"):
            return col

    if len(columns) >= 2:
        return columns[1]

    raise ValueError("Could not detect a voltage column.")


def convert_csv(
    input_path: Path,
    output_path: Path,
    shunt_ohm: float,
    voltage_offset_v: float,
    invert: bool,
) -> dict[str, float | int | str]:
    df = pd.read_csv(input_path)

    if df.shape[1] < 2:
        raise ValueError(
            "Input CSV must contain at least time and voltage columns."
        )

    columns = [str(col) for col in df.columns]

    time_col = detect_time_column(columns)
    voltage_col = detect_voltage_column(columns)

    time_s = pd.to_numeric(df[time_col], errors="coerce")
    voltage_v = pd.to_numeric(df[voltage_col], errors="coerce")

    valid = time_s.notna() & voltage_v.notna()

    time_s = time_s[valid].reset_index(drop=True)
    voltage_v = voltage_v[valid].reset_index(drop=True)

    if len(time_s) == 0:
        raise ValueError("No valid numeric samples were found.")

    corrected_voltage_v = voltage_v - voltage_offset_v

    if invert:
        corrected_voltage_v = -corrected_voltage_v

    current_a = corrected_voltage_v / shunt_ohm
    current_ma = current_a * 1000.0

    output_df = pd.DataFrame(
        {
            "time_s": time_s,
            "shunt_voltage_v": voltage_v,
            "corrected_shunt_voltage_v": corrected_voltage_v,
            "current_a": current_a,
            "current_ma": current_ma,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "time_column": time_col,
        "voltage_column": voltage_col,
        "samples": len(output_df),
        "current_mean_ma": float(current_ma.mean()),
        "current_min_ma": float(current_ma.min()),
        "current_max_ma": float(current_ma.max()),
    }


def find_csv_files(input_root: Path) -> list[Path]:
    csv_files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    ]

    return sorted(csv_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert oscilloscope shunt-voltage CSV files "
            "to current CSV files."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input root directory. Default: {DEFAULT_INPUT_ROOT}",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--shunt-ohm",
        type=float,
        default=1.7,
        help="Shunt resistance in ohms. Default: 1.7",
    )

    parser.add_argument(
        "--voltage-offset-v",
        type=float,
        default=0.0,
        help="Voltage offset subtracted before conversion. Default: 0 V",
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert voltage polarity before conversion.",
    )

    args = parser.parse_args()

    if args.shunt_ohm <= 0:
        raise ValueError("--shunt-ohm must be positive.")

    if not args.input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {args.input_root}"
        )

    csv_files = find_csv_files(args.input_root)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {args.input_root}"
        )

    print(f"Input root:  {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Shunt:      {args.shunt_ohm} ohm")
    print(f"CSV files:  {len(csv_files)}")
    print()

    summary_rows: list[dict[str, float | int | str]] = []

    for input_path in csv_files:
        relative_path = input_path.relative_to(args.input_root)

        output_filename = (
            f"{input_path.stem}_current_ma.csv"
        )

        output_path = (
            args.output_root
            / relative_path.parent
            / output_filename
        )

        try:
            result = convert_csv(
                input_path=input_path,
                output_path=output_path,
                shunt_ohm=args.shunt_ohm,
                voltage_offset_v=args.voltage_offset_v,
                invert=args.invert,
            )

            summary_rows.append(result)

            print(
                f"[OK] {relative_path} "
                f"-> {output_path.relative_to(args.output_root)}"
            )
            print(
                f"     mean={result['current_mean_ma']:.6f} mA, "
                f"min={result['current_min_ma']:.6f} mA, "
                f"max={result['current_max_ma']:.6f} mA"
            )

        except Exception as exc:
            print(f"[ERROR] {relative_path}: {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        summary_path = (
            args.output_root
            / "oscilloscope_voltage_to_current_summary.csv"
        )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        print()
        print(f"Converted: {len(summary_rows)} file(s)")
        print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()