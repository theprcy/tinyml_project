"""
Name: measure_time_delay.py

Description: Measures inference time delay for all four model variants on the Pi.
"""

import numpy as np
import time
import csv
import os
from ai_edge_litert.interpreter import Interpreter

# paths to the models
MODELS = {
    'FP32':   'models/baseline.tflite',
    'PTQ':    'models/ptq_model.tflite',
    'QAT':    'models/qat_model.tflite',
    'Binary': 'models/binary_model.tflite',
}

# actual file sizes from the clean evaluation
MODEL_SIZES = {
    'FP32':   '4,252 KB',
    'PTQ':    '366 KB',
    'QAT':    '366 KB',
    'Binary': '1,402 KB',
}

# 1000 runs per model because it gives a stable average
NUM_INFERENCES = 1000


def get_file_size_kb(model_path):
    """
    -Returns the actual file size of the tflite file in KB.
    -The real storage size on the Pi's disk.
    """
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes / 1024, 2)


def measure_model(model_path, image):
    """
    Loads the model and runs NUM_INFERENCES inferences.
    Returns avg, P95, min and max time delay in milliseconds.

    Creates a fresh interpreter each time usng the same approach as the DoS test and demo scripts for consistency.
    """
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = np.expand_dims(image, axis=0).astype(np.float32)

    # PTQ and QAT are int8 models, needs to quantise the input first
    # FP32 and Binary are float32 so they can skip this step
    if inp['dtype'] == np.int8:
        scale, zero_point = inp['quantization']
        x = x / scale + zero_point
        x = np.clip(np.round(x), -128, 127).astype(np.int8)

    """one warm-up run before we start timing
    the first inference is always slower due to memory allocation
    so we exclude it from the measurements """
    interp.set_tensor(inp['index'], x)
    interp.invoke()

    # now run NUM_INFERENCES times and record each time delay
    time_delays = []
    for _ in range(NUM_INFERENCES):
        start = time.perf_counter()
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        time_delays.append((time.perf_counter() - start) * 1000)

    return {
        'avg_time_delay_ms': round(np.mean(time_delays), 3),
        'p95_time_delay_ms': round(np.percentile(time_delays, 95), 3),
        'min_time_delay_ms': round(np.min(time_delays), 3),
        'max_time_delay_ms': round(np.max(time_delays), 3),
    }


def main():
    # use the same dummy image as the DoS test and demo script
    # fixed seed 42 keeps everything consistent across all measurements
    image = np.random.RandomState(42).rand(32, 32, 3).astype(np.float32)

    print("=" * 65)
    print("Inference Time Delay Measurement — Raspberry Pi 4B")
    print(f"Running {NUM_INFERENCES} inferences per model")
    print(f"Input: fixed random image (seed 42, shape 32x32x3)")
    print("=" * 65)
    print()

    # print the header row
    print(f"{'Model':<10} {'File Size':<12} {'Avg (ms)':<12} "
          f"{'P95 (ms)':<12} {'Min (ms)':<12} {'Max (ms)'}")
    print("-" * 70)

    results = []

    for model_name, model_path in MODELS.items():
        # show that something is happening while it runs
        print(f"{model_name:<10} measuring...", end='\r')

        # get the actual file size from disk
        file_size_kb = get_file_size_kb(model_path)

        # measure the time delay
        result = measure_model(model_path, image)
        result['model'] = model_name
        result['file_size_kb'] = file_size_kb
        results.append(result)

        # print the result row
        print(
            f"{model_name:<10} "
            f"{file_size_kb:<12.1f} "
            f"{result['avg_time_delay_ms']:<12.3f} "
            f"{result['p95_time_delay_ms']:<12.3f} "
            f"{result['min_time_delay_ms']:<12.3f} "
            f"{result['max_time_delay_ms']:<12.3f}"
        )

    # summary section -- same numbers as Table II in the paper
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for r in results:
        print(
            f"{r['model']:<8} | "
            f"Size: {r['file_size_kb']:.1f} KB | "
            f"Avg time delay: {r['avg_time_delay_ms']:.3f} ms | "
            f"P95: {r['p95_time_delay_ms']:.3f} ms"
        )
    print("=" * 65)

    # save everything to CSV so you have a record
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/pi_time_delay.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model',
            'file_size_kb',
            'avg_time_delay_ms',
            'p95_time_delay_ms',
            'min_time_delay_ms',
            'max_time_delay_ms',
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {csv_path}")
    print("Copy to your Mac with:")
    print("  scp admin@172.20.10.3:~/tinyml_project/results/pi_time_delay.csv ~/Downloads/")


if __name__ == "__main__":
    main()
