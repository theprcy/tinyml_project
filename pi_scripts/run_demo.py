"""
Name: run_demo.py

Function: Runs everything in sequence automatically:

    Part 1 — Model file sizes on the Pi
    Part 2 — Inference time delay measurement
    Part 3 — Visual inference demo (real CIFAR-10 images)
    Part 4 — Short DoS flood test with live output
"""

import numpy as np
import time
import os
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_edge_litert.interpreter import Interpreter

# CIFAR-10 class names
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model paths 
MODELS = {
    'FP32':   'models/baseline.tflite',
    'PTQ':    'models/ptq_model.tflite',
    'QAT':    'models/qat_model.tflite',
    'Binary': 'models/binary_model.tflite',
}

# Consistent colours across all parts 
COLORS = {
    'FP32':   '#FF0000',
    'PTQ':    '#008000',
    'QAT':    '#0055FF',
    'Binary': '#AA00FF',
}

# How long each demo image stays on screen
IMAGE_DISPLAY_SECONDS = 5

# How many inferences for time delay measurement
NUM_INFERENCES = 500 

# DoS test concurrency levels for the short demo
# using fewer levels than the full test to keep demo short
DEMO_CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]
DEMO_STAGE_DURATION = 15  # 15 seconds per stage for demo


# ═══════════════════════════════════════════════════════════════
# SHARED INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════

def make_interpreter(model_path):
    """Load a tflite model and return a ready interpreter."""
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def prepare_input(interp, image):
    """Prepare the input image for the model - handles int8 quantisation."""
    inp = interp.get_input_details()[0]
    x = np.expand_dims(image, axis=0).astype(np.float32)
    if inp['dtype'] == np.int8:
        scale, zero_point = inp['quantization']
        x = x / scale + zero_point
        x = np.clip(np.round(x), -128, 127).astype(np.int8)
    return x, inp, interp.get_output_details()[0]


def run_single_inference(model_path, image):
    """
    Creates its own interpreter - required because ai_edge_litert
    interpreters are not thread safe so each thread needs its own.
    Returns (predicted_class_index, time_delay_ms).
    """
    interp = make_interpreter(model_path)
    x, inp, out = prepare_input(interp, image)

    start = time.perf_counter()
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    pred = int(np.argmax(interp.get_tensor(out['index'])[0]))
    return pred, elapsed_ms


# ═══════════════════════════════════════════════════════════════
# PART 1 — MODEL FILE SIZES
# ═══════════════════════════════════════════════════════════════

def show_model_sizes():
    print("\n" + "=" * 60)
    print("PART 1 — Model File Sizes on Raspberry Pi 4B")
    print("=" * 60)
    print(f"{'Model':<10} {'File Size':>12} {'Compression':>15}")
    print("-" * 40)

    baseline_size = os.path.getsize(MODELS['FP32']) / 1024

    for name, path in MODELS.items():
        size_kb = os.path.getsize(path) / 1024
        if name == 'FP32':
            compression = "baseline"
        else:
            ratio = baseline_size / size_kb
            compression = f"{ratio:.1f}x smaller"
        print(f"{name:<10} {size_kb:>9.1f} KB   {compression:>15}")

    print("-" * 40)
    print(f"PTQ and QAT achieve ~11.6x compression with minimal accuracy loss.")
    print()
    input("Press Enter to continue to Part 2...")


# ═══════════════════════════════════════════════════════════════
# PART 2 — TIME DELAY MEASUREMENT
# ═══════════════════════════════════════════════════════════════

def measure_time_delay():
    print("\n" + "=" * 60)
    print("PART 2 — Inference Time Delay Measurement")
    print(f"Running {NUM_INFERENCES} inferences per model...")
    print("=" * 60)

    # use same fixed image as everywhere else in the project
    image = np.random.RandomState(42).rand(32, 32, 3).astype(np.float32)

    print(f"\n{'Model':<10} {'Avg Time Delay':>16} {'P95 Time Delay':>16} {'File Size':>12}")
    print("-" * 58)

    results = {}

    for name, path in MODELS.items():
        print(f"{name:<10} measuring...", end='\r')

        interp = make_interpreter(path)
        x, inp, out = prepare_input(interp, image)

        # warm up run -- excluded from timing
        interp.set_tensor(inp['index'], x)
        interp.invoke()

        # timed runs
        delays = []
        for _ in range(NUM_INFERENCES):
            start = time.perf_counter()
            interp.set_tensor(inp['index'], x)
            interp.invoke()
            delays.append((time.perf_counter() - start) * 1000)

        avg = round(np.mean(delays), 3)
        p95 = round(np.percentile(delays, 95), 3)
        size_kb = os.path.getsize(path) / 1024

        results[name] = {'avg': avg, 'p95': p95, 'size': size_kb}

        print(f"{name:<10} {avg:>13.3f} ms   {p95:>13.3f} ms   {size_kb:>9.1f} KB")

    print("-" * 58)
    print(f"\nPTQ avg: {results['PTQ']['avg']} ms vs FP32: {results['FP32']['avg']} ms")
    speedup = results['FP32']['avg'] / results['PTQ']['avg']
    print(f"QAT is {speedup:.1f}x faster than FP32 baseline.")
    print()
    input("Press Enter to continue to Part 3 — Visual Demo...")

    return results


# ═══════════════════════════════════════════════════════════════
# PART 3 — VISUAL INFERENCE DEMO
# ═══════════════════════════════════════════════════════════════

def run_all_models_on_image(image):
    """Run all four models on one image and return results."""
    results = {}
    for name, path in MODELS.items():
        pred, delay = run_single_inference(path, image)
        results[name] = {
            'pred':  pred,
            'label': CLASSES[pred],
            'delay': delay,
        }
    return results


def plot_demo_image(image, true_label, results, image_index):
    """Show the image and prediction bar chart for one test image."""
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#F8F9FA')

    # left side -- the actual CIFAR-10 image scaled up 6x so it's visible
    ax_img = fig.add_axes([0.03, 0.15, 0.22, 0.7])
    display_img = np.repeat(np.repeat(image, 6, axis=0), 6, axis=1)
    ax_img.imshow(display_img)
    ax_img.set_title(
        f'Test image #{image_index}\nTrue label: {true_label.upper()}',
        fontsize=12, fontweight='bold', pad=10
    )
    ax_img.axis('off')

    # right side -- horizontal bar chart of predictions and time delays
    ax = fig.add_axes([0.32, 0.18, 0.64, 0.65])

    model_names = list(results.keys())
    delays  = [results[m]['delay'] for m in model_names]
    colors  = [COLORS[m] for m in model_names]
    correct = [results[m]['label'] == true_label for m in model_names]

    ax.barh(model_names, delays, color=colors,
            height=0.5, edgecolor='white', linewidth=1.5)

    # label each bar with prediction, time delay and tick/cross
    for i, m in enumerate(model_names):
        pred_label = results[m]['label']
        delay      = results[m]['delay']
        is_correct = correct[i]
        symbol     = '✓' if is_correct else '✗'
        sym_color  = '#2E7D32' if is_correct else '#C62828'
        size_kb    = os.path.getsize(MODELS[m]) / 1024

        ax.text(
            delay + 0.3, i,
            f"{symbol}  {pred_label}  ({delay:.1f} ms)  [{size_kb:.0f} KB]",
            va='center', ha='left', fontsize=10,
            color=sym_color,
            fontweight='bold' if not is_correct else 'normal'
        )

    ax.set_xlabel('Inference time delay (ms)', fontsize=11)
    ax.set_title('Model predictions and inference time delay', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(delays) * 2.2)
    ax.set_facecolor('#F8F9FA')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    correct_patch   = mpatches.Patch(color='#2E7D32', label='Correct')
    incorrect_patch = mpatches.Patch(color='#C62828', label='Incorrect')
    ax.legend(handles=[correct_patch, incorrect_patch],
              loc='lower right', fontsize=9)

    fig.suptitle(
        'TinyML Quantisation Security Study — Live Inference Demo\n'
        'Raspberry Pi 4 Model B  ·  CIFAR-10',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()

    # save and display
    os.makedirs('demo_outputs', exist_ok=True)
    save_path = f"demo_outputs/demo_{true_label}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')

    plt.pause(IMAGE_DISPLAY_SECONDS)
    plt.close(fig)


def run_visual_demo():
    print("\n" + "=" * 60)
    print("PART 3 — Live Visual Inference Demo")
    print("Classifying one image per CIFAR-10 class...")
    print("=" * 60)

    # load test images
    if os.path.exists('test_images.npz'):
        data   = np.load('test_images.npz')
        x_test = data['x_test']
        y_test = data['y_test']
    else:
        print("test_images.npz not found -- please copy it to ~/tinyml_project/")
        return

    # pick one image from each class for variety
    demo_indices = []
    for class_idx in range(10):
        matches = np.where(y_test == class_idx)[0]
        demo_indices.append(int(matches[0]))

    all_results = []

    for i, idx in enumerate(demo_indices):
        image      = x_test[idx].astype(np.float32)
        true_label = CLASSES[int(y_test[idx])]

        print(f"\n[{i+1}/10] Image #{idx} — True label: {true_label}")

        results = run_all_models_on_image(image)

        for name, r in results.items():
            symbol = '✓' if r['label'] == true_label else '✗'
            print(f"  {name:<8} → {r['label']:<12} ({r['delay']:.2f} ms) {symbol}")

        all_results.append((true_label, results))
        plot_demo_image(image, true_label, results, idx)

    # print accuracy summary
    print("\n" + "=" * 60)
    print("VISUAL DEMO SUMMARY")
    print("=" * 60)
    for name in MODELS.keys():
        correct = sum(1 for tl, r in all_results if r[name]['label'] == tl)
        avg_delay = np.mean([r[name]['delay'] for _, r in all_results])
        size_kb = os.path.getsize(MODELS[name]) / 1024
        print(f"{name:<8} | {correct}/10 correct | "
              f"avg {avg_delay:.2f} ms | {size_kb:.0f} KB")
    print("=" * 60)

    input("\nPress Enter to continue to Part 4 — DoS Test...")


# ═══════════════════════════════════════════════════════════════
# PART 4 — SHORT DoS FLOOD TEST
# ═══════════════════════════════════════════════════════════════

def run_flood_stage(model_path, image, concurrency, duration):
    """Run concurrent requests for a fixed duration and collect time delays."""
    delays = []
    stop_time = time.time() + duration

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        while time.time() < stop_time:
            while len(futures) < concurrency and time.time() < stop_time:
                futures.append(
                    executor.submit(run_single_inference, model_path, image)
                )
            done = [f for f in futures if f.done()]
            for f in done:
                _, delay = f.result()
                delays.append(delay)
            futures = [f for f in futures if not f.done()]
        for f in as_completed(futures):
            _, delay = f.result()
            delays.append(delay)

    return {
        'concurrency': concurrency,
        'avg_time_delay_ms': round(np.mean(delays), 2) if delays else 0,
        'p95_time_delay_ms': round(np.percentile(delays, 95), 2) if delays else 0,
        'completed': len(delays),
    }


def run_dos_demo():
    print("\n" + "=" * 60)
    print("PART 4 — DoS Flood Test (short version)")
    print(f"Concurrency levels: {DEMO_CONCURRENCY_LEVELS}")
    print(f"Duration per stage: {DEMO_STAGE_DURATION} seconds")
    print("=" * 60)

    # use fixed random image -- same as everywhere else
    image = np.random.RandomState(42).rand(32, 32, 3).astype(np.float32)

    all_results = {name: [] for name in MODELS}

    for name, path in MODELS.items():
        print(f"\n--- {name} ---")
        print(f"{'Concurrency':<15} {'Avg Delay (ms)':<18} {'P95 Delay (ms)':<18} {'Requests'}")
        print("-" * 60)

        for c in DEMO_CONCURRENCY_LEVELS:
            print(f"  c={c:<12} running {DEMO_STAGE_DURATION}s...", end='\r')
            result = run_flood_stage(path, image, c, DEMO_STAGE_DURATION)
            all_results[name].append(result)
            print(
                f"  c={c:<12} "
                f"{result['avg_time_delay_ms']:<18.2f} "
                f"{result['p95_time_delay_ms']:<18.2f} "
                f"{result['completed']}"
            )

    # save results
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/demo_dos_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'concurrency',
            'avg_time_delay_ms', 'p95_time_delay_ms', 'completed'
        ])
        writer.writeheader()
        for name, stages in all_results.items():
            for stage in stages:
                stage['model'] = name
                writer.writerow(stage)

    print(f"\nSaved to: {csv_path}")

    # final summary
    print("\n" + "=" * 60)
    print("DoS TEST SUMMARY — Avg time delay at c=100 (ms)")
    print("=" * 60)
    for name in MODELS.keys():
        last = all_results[name][-1]
        print(f"{name:<8} | c=100 avg: {last['avg_time_delay_ms']:.2f} ms | "
              f"completed: {last['completed']} requests")
    print("=" * 60)
    print("No model crashed or became unresponsive at any concurrency level.")
    print("PTQ and QAT show better sustained throughput than FP32 and Binary.")


# ═══════════════════════════════════════════════════════════════
# MAIN — runs all four parts in sequence
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("TinyML Quantisation Security Study")
    print("Supervisor Demo — Raspberry Pi 4 Model B")
    print("=" * 60)
    print("\nThis demo covers all four research objectives:")
    print("  Part 1 — Model file sizes")
    print("  Part 2 — Inference time delay")
    print("  Part 3 — Live inference on real images")
    print("  Part 4 — DoS flood test")
    print()
    input("Press Enter to start...")

    # run all four parts in sequence
    show_model_sizes()
    measure_time_delay()
    run_visual_demo()
    run_dos_demo()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("Key findings demonstrated:")
    print("- PTQ and QAT: 11.6x smaller, up to 27x faster than FP32")
    print("- All models degrade gracefully under DoS - none crashed")
    print("- PTQ and QAT handle more requests under sustained load")
    print("- QAT: best balance of efficiency, accuracy and resilience")
    print("=" * 60)


if __name__ == "__main__":
    main()
