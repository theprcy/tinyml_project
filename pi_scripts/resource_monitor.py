"""
Name: resource_monitor.py

Description: Logs system-wide CPU and memory usage every 0.5 seconds to a CSV file,
intended to run as a separate background process while the inference
load test (run_dos_test.py) is running.
"""

import psutil
import csv
import time
import sys
import os


def monitor(output_path, interval_seconds=0.5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'memory_used_mb'])

        print(f"Monitoring started. Logging to {output_path} every {interval_seconds}s. Ctrl+C to stop.")

        try:
            while True:
                cpu = psutil.cpu_percent(interval=interval_seconds)
                mem = psutil.virtual_memory()
                writer.writerow([
                    time.time(),
                    cpu,
                    mem.percent,
                    mem.used / (1024 * 1024)
                ])
                f.flush()
        except KeyboardInterrupt:
            print("Monitoring stopped.")


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/monitor_log.csv"
    monitor(output_path)
