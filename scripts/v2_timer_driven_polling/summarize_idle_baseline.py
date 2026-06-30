import csv
import glob
import os
import sys
import statistics

def read_values(path):
    currents = []
    powers = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "status" in row and row["status"] != "OK":
                continue

            if "current_mA" in row and "power_mW" in row:
                currents.append(float(row["current_mA"]))
                powers.append(float(row["power_mW"]))

    return currents, powers


def summarize(values):
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/summarize_idle_baseline.py <csv_folder>")
        sys.exit(1)

    folder = sys.argv[1]

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    csv_files = [
        f for f in csv_files
        if not os.path.basename(f).startswith("timestamp_jitter_summary")
        and not os.path.basename(f).startswith("stable_idle_baseline_summary")
    ]

    if not csv_files:
        print("No CSV files found.")
        sys.exit(1)

    output_path = os.path.join(folder, "stable_idle_baseline_summary.csv")

    all_currents = []
    all_powers = []
    rows = []

    for path in csv_files:
        currents, powers = read_values(path)

        if not currents or not powers:
            print(f"Skipping {path}: no valid current/power data")
            continue

        current_summary = summarize(currents)
        power_summary = summarize(powers)

        all_currents.extend(currents)
        all_powers.extend(powers)

        rows.append({
            "file": os.path.basename(path),
            "samples": current_summary["samples"],

            "mean_current_mA": current_summary["mean"],
            "sd_current_mA": current_summary["sd"],
            "min_current_mA": current_summary["min"],
            "max_current_mA": current_summary["max"],

            "mean_power_mW": power_summary["mean"],
            "sd_power_mW": power_summary["sd"],
            "min_power_mW": power_summary["min"],
            "max_power_mW": power_summary["max"],
        })

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "file", "samples",
            "mean_current_mA", "sd_current_mA", "min_current_mA", "max_current_mA",
            "mean_power_mW", "sd_power_mW", "min_power_mW", "max_power_mW",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall_current = summarize(all_currents)
    overall_power = summarize(all_powers)

    print(f"Output: {output_path}")
    print()
    print("Overall stable idle baseline")
    print(f"samples: {overall_current['samples']}")

    print()
    print("Current:")
    print(f"mean_current_mA: {overall_current['mean']:.6f}")
    print(f"sd_current_mA: {overall_current['sd']:.6f}")
    print(f"min_current_mA: {overall_current['min']:.6f}")
    print(f"max_current_mA: {overall_current['max']:.6f}")
    print(f"noise_band_current_mA_mean_plus_minus_3sd: "
          f"{overall_current['mean'] - 3 * overall_current['sd']:.6f} "
          f"to {overall_current['mean'] + 3 * overall_current['sd']:.6f}")

    print()
    print("Power:")
    print(f"mean_power_mW: {overall_power['mean']:.6f}")
    print(f"sd_power_mW: {overall_power['sd']:.6f}")
    print(f"min_power_mW: {overall_power['min']:.6f}")
    print(f"max_power_mW: {overall_power['max']:.6f}")
    print(f"noise_band_power_mW_mean_plus_minus_3sd: "
          f"{overall_power['mean'] - 3 * overall_power['sd']:.6f} "
          f"to {overall_power['mean'] + 3 * overall_power['sd']:.6f}")


if __name__ == "__main__":
    main()