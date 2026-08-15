"""
Name: demo_visual.py

Description: 
Loads real CIFAR-10 test images, runs all four model variants,
and shows a visual of the image + predictions + time delay
"""

import numpy as np
import time
import os
import matplotlib
matplotlib.use('TkAgg') # use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ai_edge_litert.interpreter import Interpreter

# the 10 CIFAR-10 categories in the right order
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# paths to the four tflite files sitting on the Pi
MODELS = {
    'FP32':   'models/baseline.tflite',
    'PTQ':    'models/ptq_model.tflite',
    'QAT':    'models/qat_model.tflite',
    'Binary': 'models/binary_model.tflite',
}

MODEL_COLORS = {
    'FP32':   '#4C72B0',
    'PTQ':    '#C44E52',
    'QAT':    '#DD8452',
    'Binary': '#55A868',
}

# file sizes to show on the chart — these match your results table
MODEL_SIZES = {
    'FP32':   '4,252 KB',
    'PTQ':    '366 KB',
    'QAT':    '366 KB',
    'Binary': '1,402 KB',
}


def run_inference(model_path, image):
    """
    Takes a model file path and a single image (32x32x3, float32, range 0-1).
    Returns the predicted class index and how long it took in milliseconds.
    """
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = np.expand_dims(image, axis=0).astype(np.float32)

    # PTQ and QAT are int8 models so we need to quantise the input
    # before passing it in — FP32 and Binary skip this step
    if inp['dtype'] == np.int8:
        scale, zero_point = inp['quantization']
        x = x / scale + zero_point
        x = np.clip(np.round(x), -128, 127).astype(np.int8)

    # one warm-up run first — the first call is always slower
    # because of memory allocation, so we don't count it
    interp.set_tensor(inp['index'], x)
    interp.invoke()

    # now the actual timed run
    start = time.perf_counter()
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # grab the output and find the highest scoring class
    pred = int(np.argmax(interp.get_tensor(out['index'])[0]))
    return pred, elapsed_ms


def run_all_models(image):
    """
    Runs all four models on the same image one by one
    Returns a dict with the prediction and time delay for each
    """
    results = {}
    for name, path in MODELS.items():
        pred, delay = run_inference(path, image)
        results[name] = {
            'pred':  pred,
            'label': CLASSES[pred],
            'delay': delay,
        }
    return results


def plot_demo(image, true_label, results, image_index, save_path=None):
    """
    Makes the plot with the predictions
    """
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#F8F9FA')

    # left side: show the actual CIFAR-10 image
    # CIFAR-10 is only 32x32 which looks tiny, so we scale it up 6x
    ax_img = fig.add_axes([0.03, 0.15, 0.22, 0.7])
    display_img = np.repeat(np.repeat(image, 6, axis=0), 6, axis=1)
    ax_img.imshow(display_img)
    ax_img.set_title(
        f'Test image #{image_index}\nTrue label: {true_label.upper()}',
        fontsize=12, fontweight='bold', pad=10
    )
    ax_img.axis('off')

    # right side: horizontal bar chart — longer bar = more time delay
    ax = fig.add_axes([0.32, 0.18, 0.64, 0.65])

    model_names = list(results.keys())
    delays      = [results[m]['delay'] for m in model_names]
    colors      = [MODEL_COLORS[m] for m in model_names]
    correct     = [results[m]['label'] == true_label for m in model_names]

    ax.barh(model_names, delays, color=colors,
            height=0.5, edgecolor='white', linewidth=1.5)

    # add a label at the end of each bar showing prediction + delay + size
    for i, m in enumerate(model_names):
        pred_label = results[m]['label']
        delay      = results[m]['delay']
        is_correct = correct[i]

        # green tick if right, red cross if wrong
        symbol    = '✓' if is_correct else '✗'
        sym_color = '#2E7D32' if is_correct else '#C62828'

        ax.text(
            delay + 0.3, i,
            f"{symbol}  {pred_label}  ({delay:.1f} ms)  [{MODEL_SIZES[m]}]",
            va='center', ha='left', fontsize=10,
            color=sym_color,
            fontweight='bold' if not is_correct else 'normal'
        )

    ax.set_xlabel('Inference time delay (ms)', fontsize=11)
    ax.set_title(
        'Model predictions and inference time delay',
        fontsize=12, fontweight='bold'
    )
    # give enough room on the right for the text labels
    ax.set_xlim(0, max(delays) * 2.2)
    ax.set_facecolor('#F8F9FA')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # small legend so it's obvious what tick and cross mean
    correct_patch   = mpatches.Patch(color='#2E7D32', label='Correct prediction')
    incorrect_patch = mpatches.Patch(color='#C62828', label='Incorrect prediction')
    ax.legend(handles=[correct_patch, incorrect_patch],
              loc='lower right', fontsize=9)

    fig.suptitle(
        'TinyML Quantisation Security Study — Live Inference Demo\n'
        'Raspberry Pi 4 Model B  ·  CIFAR-10',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()

    #if save_path:
    #    fig.savefig(save_path, dpi=150, bbox_inches='tight',
    #                facecolor='#F8F9FA')
    #    print(f"  Saved: {save_path}"
    plt.pause(5)
    plt.close(fig)


def print_summary(all_image_results):
    """
    Prints a clean summary table after all images have been processed.
    Shows accuracy and average time delay per model — same numbers
    as in the paper but printed live from actual Pi measurements.
    """
    print("\n" + "="*65)
    print("DEMO SUMMARY")
    print("="*65)

    for model_name in MODELS.keys():
        # count how many images this model got right
        correct_count = sum(
            1 for true_label, results in all_image_results
            if results[model_name]['label'] == true_label
        )
        total     = len(all_image_results)
        avg_delay = np.mean([
            results[model_name]['delay']
            for _, results in all_image_results
        ])
        print(
            f"{model_name:<8} | "
            f"Accuracy: {correct_count}/{total} "
            f"({100*correct_count/total:.0f}%)  | "
            f"Avg delay: {avg_delay:.2f} ms  | "
            f"Size: {MODEL_SIZES[model_name]}"
        )

    print("="*65)
    print("Recommendation: QAT offers the best balance of")
    print("efficiency, accuracy, and security for TinyML deployment.")
    print("="*65)


def main():
    # try to load from the saved test split first
    # if it's not there, just download CIFAR-10 directly
    print("Loading CIFAR-10 test images...")

    if os.path.exists('test_images.npz'):
        # this is the same split used throughout the project (seed 42)
        data   = np.load('test_images.npz')
        x_test = data['x_test']
        y_test = data['y_test']
        print(f"Loaded from data_split.npz — {len(x_test)} images")

    else:
        # test_images.npz not found, download the raw dataset instead
        print("test_images.npz not found, downloading CIFAR-10...")
        import urllib.request, pickle, tarfile, io

        url      = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        response = urllib.request.urlopen(url)
        tar      = tarfile.open(fileobj=io.BytesIO(response.read()))
        member   = tar.getmember('cifar-10-batches-py/test_batch')
        f        = tar.extractfile(member)
        batch    = pickle.loads(f.read(), encoding='bytes')

        # CIFAR-10 stores images as (N, 3, 32, 32) — rearrange to (N, 32, 32, 3)
        x_test = batch[b'data'].reshape(-1, 3, 32, 32)
        x_test = np.transpose(x_test, (0, 2, 3, 1)).astype(np.float32) / 255.0
        y_test = np.array(batch[b'labels'])
        print(f"Downloaded — {len(x_test)} test images")

    # pick one image from each class so we get a nice variety
    # this means 10 images total, one airplane, one cat, one truck, etc.
    demo_indices = []
    for class_idx in range(10):
        matches = np.where(y_test == class_idx)[0]
        demo_indices.append(int(matches[0]))

    os.makedirs('demo_outputs', exist_ok=True)

    all_image_results = []

    print(f"\nRunning inference on {len(demo_indices)} images...\n")

    for i, idx in enumerate(demo_indices):
        image      = x_test[idx].astype(np.float32)
        true_label = CLASSES[int(y_test[idx])]

        print(f"[{i+1}/{len(demo_indices)}] Image #{idx} — True label: {true_label}")

        results = run_all_models(image)

        # print a quick result per model so the terminal isn't silent
        for name, r in results.items():
            symbol = '✓' if r['label'] == true_label else '✗'
            print(f"  {name:<8} → {r['label']:<12} ({r['delay']:.2f} ms) {symbol}")

        all_image_results.append((true_label, results))

        # save each chart as a PNG in demo_outputs/
        save_path = f"demo_outputs/demo_{i+1:02d}_{true_label}.png"
        plot_demo(image, true_label, results, idx, save_path=save_path)

    # final summary once all images are done
    print_summary(all_image_results)
    print(f"\nAll demo figures saved to: demo_outputs/")


if __name__ == "__main__":
    main()
