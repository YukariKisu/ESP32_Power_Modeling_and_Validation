import csv
import sys
import statistics
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 summarize_timestamp_jitter.py <timestamp_jitter_summary.csv>")
        sys.exit(1)

    path = sys.argv[1]

    mean_dt = []
    sd_dt = []
    min_dt = []
    max_dt = []
    max_abs_error = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mean_dt.append(float(row["mean_dt_us"]))
            sd_dt.append(float(row["sd_dt_us"]))
            min_dt.append(float(row["min_dt_us"]))
            max_dt.append(float(row["max_dt_us"]))
            max_abs_error.append(float(row["max_abs_error_us"]))

    print(f"File: {path}")
    print(f"runs: {len(mean_dt)}")
    print(f"mean of mean_dt_us: {statistics.mean(mean_dt):.3f}")
    print(f"mean of sd_dt_us: {statistics.mean(sd_dt):.3f}")
    print(f"minimum min_dt_us: {min(min_dt):.0f}")
    print(f"maximum max_dt_us: {max(max_dt):.0f}")
    print(f"maximum max_abs_error_us: {max(max_abs_error):.0f}")

if __name__ == "__main__":
    main()