"""
Name: run_dos_test.py

Description: Runs the Denial-of-Service evaluation:
1. Request-flooding test: concurrency ramped through stages (1, 5, 10,
   25, 50, 100 simultaneous inference requests), each held for 30 seconds
2. Sustained-load control condition: continuous single-threaded inference
   for 300 seconds
"""
# import 
import time
import csv
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pi_model_utils import PiModel, MODEL_PATHS

# Configuration
CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]
STAGE_DURATION_SECONDS = 30
SUSTAINED_LOAD_DURATION_SECONDS = 300
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def make_dummy_image():
    rng = np.random.RandomState(42)
    return rng.rand(32, 32, 3).astype(np.float32)


def run_single_inference(model_path, image):
    """Creates its own interpreter instance per call which is  required because
    ai_edge_litert interpreters are not thread-safe and cannot be shared
    across concurrent threads"""
    model = PiModel(model_path)
    start = time.perf_counter()
    model.predict(image)
    return time.perf_counter() - start


def run_flood_stage(model_path, image, concurrency, duration_seconds):
    latencies = []
    stop_time = time.time() + duration_seconds

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        while time.time() < stop_time:
            while len(futures) < concurrency and time.time() < stop_time:
                futures.append(
                    executor.submit(run_single_inference, model_path, image)
                )
            done = [f for f in futures if f.done()]
            for f in done:
                latencies.append(f.result())
            futures = [f for f in futures if not f.done()]

        for f in as_completed(futures):
            latencies.append(f.result())

    throughput = len(latencies) / duration_seconds
    return {
        'concurrency': concurrency,
        'completed_requests': len(latencies),
        'throughput_per_sec': throughput,
        'avg_latency_ms': np.mean(latencies) * 1000 if latencies else None,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000 if latencies else None,
    }


def run_flood_test_for_model(model_name, model_path, image):
    print(f"\n=== Flood test: {model_name} ===")
    results = []
    for concurrency in CONCURRENCY_LEVELS:
        print(f"  Concurrency {concurrency} - running for {STAGE_DURATION_SECONDS}s...")
        stage_result = run_flood_stage(model_path, image, concurrency, STAGE_DURATION_SECONDS)
        stage_result['model'] = model_name
        results.append(stage_result)
        print(f"    Throughput: {stage_result['throughput_per_sec']:.2f} req/s, "
              f"avg latency: {stage_result['avg_latency_ms']:.2f} ms, "
              f"P95: {stage_result['p95_latency_ms']:.2f} ms")
    return results


def run_sustained_load_for_model(model_name, model_path, image, duration_seconds):
    print(f"\n=== Sustained load (control): {model_name} - {duration_seconds}s ===")
    # Sustained load is single-threaded so one shared instance is fine here
    model = PiModel(model_path)
    latencies = []
    stop_time = time.time() + duration_seconds
    while time.time() < stop_time:
        start = time.perf_counter()
        model.predict(image)
        latencies.append(time.perf_counter() - start)

    result = {
        'model': model_name,
        'duration_seconds': duration_seconds,
        'completed_requests': len(latencies),
        'avg_latency_ms': np.mean(latencies) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
    }
    print(f"  Completed {result['completed_requests']} requests, "
          f"avg latency: {result['avg_latency_ms']:.2f} ms, "
          f"P95: {result['p95_latency_ms']:.2f} ms")
    return result


def main():
    print("Starting DoS test...")
    image = make_dummy_image()

    flood_results = []
    sustained_results = []

    for model_name, model_path in MODEL_PATHS.items():
        full_path = f"models/{model_path.split('/')[-1]}"
        flood_results.extend(
            run_flood_test_for_model(model_name, full_path, image)
        )
        sustained_results.append(
            run_sustained_load_for_model(model_name, full_path, image, SUSTAINED_LOAD_DURATION_SECONDS)
        )

    flood_path = f"{RESULTS_DIR}/dos_flood_results.csv"
    with open(flood_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'concurrency', 'completed_requests',
            'throughput_per_sec', 'avg_latency_ms', 'p95_latency_ms'
        ])
        writer.writeheader()
        writer.writerows(flood_results)
    print(f"\nSaved flood test results to {flood_path}")

    sustained_path = f"{RESULTS_DIR}/dos_sustained_results.csv"
    with open(sustained_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'duration_seconds', 'completed_requests',
            'avg_latency_ms', 'p95_latency_ms'
        ])
        writer.writeheader()
        writer.writerows(sustained_results)
    print(f"Saved sustained load results to {sustained_path}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
