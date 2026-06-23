import csv
import glob
import os
import sys
import statistics

EXPECTED_DT_US = 20000

def analyze_file(path):
    timestamps = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if "timestamp_us" in reader.fieldnames:
            time_col = "timestamp_us"
            scale = 1
        elif "timestamp_ms" in reader.fieldnames:
            time_col = "timestamp_ms"
            scale = 1000
        else:
            raise ValueError(f"No timestamp column found in {path}")

        for row in reader:
            timestamps.append(int(row[time_col]) * scale)

    dt = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    mean_dt = statistics.mean(dt)
    sd_dt = statistics.stdev(dt) if len(dt) > 1 else 0.0
    min_dt = min(dt)
    max_dt = max(dt)
    max_abs_error = max(abs(x - EXPECTED_DT_US) for x in dt)

    return {
        "file": os.path.basename(path),
        "samples": len(timestamps),
        "mean_dt_us": mean_dt,
        "sd_dt_us": sd_dt,
        "min_dt_us": min_dt,
        "max_dt_us": max_dt,
        "max_abs_error_us": max_abs_error,
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_timestamp_jitter.py <csv_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not files:
        print(f"No CSV files found in {folder}")
        sys.exit(1)

    results = []

    for path in files:
        result = analyze_file(path)
        results.append(result)

    output_path = os.path.join(folder, "timestamp_jitter_summary.csv")

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "file",
            "samples",
            "mean_dt_us",
            "sd_dt_us",
            "min_dt_us",
            "max_dt_us",
            "max_abs_error_us",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output_path}")
    print()
    for r in results:
        print(
            f"{r['file']}: "
            f"mean={r['mean_dt_us']:.2f} us, "
            f"sd={r['sd_dt_us']:.2f} us, "
            f"min={r['min_dt_us']} us, "
            f"max={r['max_dt_us']} us, "
            f"max_error={r['max_abs_error_us']} us"
        )

if __name__ == "__main__":
    main()