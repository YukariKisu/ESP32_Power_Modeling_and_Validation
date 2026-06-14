import serial
import csv
import time
from pathlib import Path

PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
OUTPUT_FILE = Path("data/raw/serial_test_run01.csv")
DURATION_SEC = 10

def is_data_line(line: str) -> bool:
    parts = line.split(",")
    if len(parts) != 4:
        return False
    try:
        float(parts[2])
        float(parts[3])
        return True
    except ValueError:
        return False

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser, open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "run_id", "time_s", "current_mA"])

        start = time.time()
        print(f"Logging serial data from {PORT} to {OUTPUT_FILE}")

        while time.time() - start < DURATION_SEC:
            raw = ser.readline().decode(errors="ignore").strip()

            if not raw:
                continue

            if is_data_line(raw):
                writer.writerow(raw.split(","))
                print(raw)

        print("Logging finished.")

if __name__ == "__main__":
    main()
