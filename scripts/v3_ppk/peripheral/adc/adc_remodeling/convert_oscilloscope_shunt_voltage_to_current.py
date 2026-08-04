from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_time_column(columns) -> str:
    for col in columns:
        n = normalize(col)
        if n in {"s", "sec", "second", "seconds", "time", "times", "timestamps"}:
            return col
    for col in columns:
        if "time" in normalize(col):
            return col
    return columns[0]


def detect_voltage_column(columns) -> str:
    for col in columns:
        n = normalize(col)
        if "ch1" in n and ("v" in n or "volt" in n):
            return col
    for col in columns:
        n = normalize(col)
        if "volt" in n or n.endswith("v"):
            return col
    if len(columns) >= 2:
        return columns[1]
    raise ValueError("Could not detect a voltage column.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert oscilloscope shunt-voltage CSV data to current."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--shunt-ohm",
        type=float,
        default=1.7,
        help="Shunt resistance in ohms (default: 1.7).",
    )
    parser.add_argument(
        "--voltage-offset-v",
        type=float,
        default=0.0,
        help="Voltage offset to subtract before conversion (default: 0).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert voltage polarity before conversion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input_stem>_current.csv",
    )
    args = parser.parse_args()

    if args.shunt_ohm <= 0:
        raise ValueError("--shunt-ohm must be positive.")

    df = pd.read_csv(args.input_csv)
    if df.shape[1] < 2:
        raise ValueError("Input CSV must contain at least time and voltage columns.")

    time_col = detect_time_column(list(df.columns))
    voltage_col = detect_voltage_column(list(df.columns))

    time_s = pd.to_numeric(df[time_col], errors="coerce")
    voltage_v = pd.to_numeric(df[voltage_col], errors="coerce")

    valid = time_s.notna() & voltage_v.notna()
    time_s = time_s[valid].reset_index(drop=True)
    voltage_v = voltage_v[valid].reset_index(drop=True)

    corrected_voltage_v = voltage_v - args.voltage_offset_v
    if args.invert:
        corrected_voltage_v = -corrected_voltage_v

    current_a = corrected_voltage_v / args.shunt_ohm
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

    output_path = args.output
    if output_path is None:
        output_path = args.input_csv.with_name(
            f"{args.input_csv.stem}_current.csv"
        )

    output_df.to_csv(output_path, index=False)

    print(f"Input: {args.input_csv}")
    print(f"Time column: {time_col}")
    print(f"Voltage column: {voltage_col}")
    print(f"Shunt resistance: {args.shunt_ohm:.9g} ohm")
    print(f"Voltage offset: {args.voltage_offset_v:.9g} V")
    print(f"Samples: {len(output_df)}")
    print(f"Current mean: {current_ma.mean():.6f} mA")
    print(f"Current min:  {current_ma.min():.6f} mA")
    print(f"Current max:  {current_ma.max():.6f} mA")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()