# Developer Guide

**Project:** Evaluating the Security Impact of Quantisation on TinyML Models Against Adversarial and Denial-of-Service Attacks  

## 1. Project Overview

This project evaluates how three quantisation methods: Post-Training Quantisation (PTQ), Quantisation-Aware Training (QAT), and binary quantisation, affect the security of a TinyML model deployed on the Raspberry Pi 4B and Raspberry Pi 5.

The pipeline is split across two hardware environments:

- **Google Colaboratory (T4 GPU):** model training, quantisation, adversarial evaluation
- **Raspberry Pi 4B and Raspberry Pi 5:** hardware deployment, DoS testing, inference timing

---

## 2. Github Repository Structure

```
tinyml_security/
│
├── colab_notebook/    # Google Colab notebooks
│   ├── baseline_training.py    # Train FP32 baseline CNN
│   ├── ptq.py    # Post-Training Quantisation
│   ├── qat.py    # Quantisation-Aware Training
│   ├── binary.py    # Binary quantisation
│   ├── adversarial_attack_evaluation.py    # FGSM and PGD adversarial evaluation
│
├── pi_scripts/  # Raspberry Pi deployment scripts
│   ├── pi_model_utils.py    # TFLite model loader utility
│   ├── resource_monitor.py    # CPU and memory logger
│   ├── run_dos_test.py    # DoS flood and sustained load test
│   ├── measure_time_delay.py    # Inference time delay measurement
│   ├── demo_visual.py    # Visual inference demo
│   └── run_demo.py    # Master demo script (4 parts)
│
├── figures/    # Generated result figures
│   ├── fig1_whitebox_asr.png    # White-box ASR results
│   ├── fig2_transfer_asr.png    # Transfer ASR results
│   ├── fig3_whitebox_vs_transfer_pgd.png    # Gradient masking overlay
│   ├── dos_fig1_avg_latency.png    # DoS flood test latency
│   └── dos_fig4_sustained.png    # Sustained load bar chart
│
└── README.md    # This file
```

---

## 3. Requirements

### Google Colab

All Colab notebooks run on Google Colaboratory with a T4 GPU runtime. No local installation is required. The following libraries are used and installed within each notebook:

```
tensorflow==2.20
keras==3.x
numpy
matplotlib
adversarial-robustness-toolbox (ART)
```

### Raspberry Pi 4B and Raspberry Pi 5

| Component | Specification |
|---|---|
| OS | Debian GNU/Linux 13 (Trixie) 64-bit |
| Python | 3.13 |
| Inference runtime | ai-edge-litert 1.0.1 |

Required Python packages:

```
ai-edge-litert
numpy
psutil
matplotlib
pillow
```

---

## 4. Google Colab Setup

### Step 1 — Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.

### Step 2 — Set runtime to GPU

```
Runtime → Change runtime type → T4 GPU → Save
```

### Step 3 — Mount Google Drive

Each notebook begins with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run this cell and authorise access when prompted.

### Step 4 — Set the project directory

At the top of each notebook, set:

```python
PROJECT_DIR = '/content/drive/MyDrive/tinyml-quant-security'
```

Change this path to match where you have stored the project on your Google Drive.

### Step 5 — Run the notebooks in order

Run the notebooks in the following order. Each notebook depends on the outputs of the previous one.

| Order | Notebook | Output |
|---|---|---|
| 1 | `baseline_training.ipynb` | `baseline.keras`, `baseline.tflite` |
| 2 | `ptq.ipynb` | `ptq_model.tflite` |
| 3 | `qat.ipynb` | `qat_model.tflite` |
| 4 | `binary.ipynb` | `binary_model.tflite`, `binary_packed.npz` |
| 5 | `adversarial.ipynb` | ASR results, adversarial figures |

> **Important:** Always run all cells from top to bottom in each notebook. Do not skip cells.

### Step 6 — Transfer model files to Raspberry Pi

Once all four `.tflite` files are saved to Google Drive, download them to your Mac and copy them to the Pi using `scp`:

```bash
scp ~/Downloads/baseline.tflite admin@172.20.10.3:~/tinyml_project/models/
scp ~/Downloads/ptq_model.tflite admin@172.20.10.3:~/tinyml_project/models/
scp ~/Downloads/qat_model.tflite admin@172.20.10.3:~/tinyml_project/models/
scp ~/Downloads/binary_model.tflite admin@172.20.10.3:~/tinyml_project/models/
```

Replace `172.20.10.3` with your RPi 4B IP address and `172.20.10.2` with your RPi 5 IP address.

---

## 5. Raspberry Pi Setup

These steps apply to both RPi 4B and RPi 5 unless stated otherwise.

### Step 1 — Find the Pi's IP address

On your Mac terminal:

```bash
arp -a
```

Look for the device at `172.20.10.x`. In this project:
- RPi 4B: `172.20.10.3`
- RPi 5: `172.20.10.2`

### Step 2 — SSH into the Pi

```bash
ssh admin@172.20.10.3   # RPi 4B
ssh admin@172.20.10.2   # RPi 5
```

### Step 3 — Create the project directory

```bash
mkdir -p ~/tinyml_project/models
mkdir -p ~/tinyml_project/results
mkdir -p ~/tinyml_project/demo_outputs
```

### Step 4 — Install required packages

```bash
pip3 install ai-edge-litert --break-system-packages
pip3 install numpy psutil matplotlib pillow --break-system-packages
```

### Step 5 — Copy scripts to the Pi

From your Mac terminal:

```bash
scp pi_scripts/pi_model_utils.py admin@172.20.10.3:~/tinyml_project/
scp pi_scripts/resource_monitor.py admin@172.20.10.3:~/tinyml_project/
scp pi_scripts/run_dos_test.py admin@172.20.10.3:~/tinyml_project/
scp pi_scripts/measure_time_delay.py admin@172.20.10.3:~/tinyml_project/
scp pi_scripts/demo_visual.py admin@172.20.10.3:~/tinyml_project/
scp pi_scripts/run_demo.py admin@172.20.10.3:~/tinyml_project/
```

### Step 6 — Set up VNC (for visual demo only)

VNC is required only for running the visual demo. It is not needed for DoS testing or time delay measurement.

```bash
# Install VNC server and desktop
sudo apt-get install realvnc-vnc-server lightdm lightdm-gtk-greeter lxde --yes

# Fix lightdm configuration
sudo sed -i 's/user-session=rpd-labwc/user-session=LXDE/' /etc/lightdm/lightdm.conf
sudo sed -i 's/greeter-session=lightdm-gtk-greeter-labwc/greeter-session=lightdm-gtk-greeter/' /etc/lightdm/lightdm.conf
sudo sed -i 's/autologin-session=rpd-labwc/autologin-session=LXDE/' /etc/lightdm/lightdm.conf

# Start services
sudo systemctl enable vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced
sudo systemctl start lightdm
```

Connect from your Mac using RealVNC Viewer to the Pi's IP address.

---

## 6. Running the Experiments

### 6.1 Measure Inference Time Delay

SSH into the Pi and run:

```bash
cd ~/tinyml_project
python3 measure_time_delay.py
```

This runs 500 inferences per model using a fixed test image (seed 42) and saves results to:

```
results/pi_time_delay.csv
```

Expected output:

```
Model      Avg (ms)    P95 (ms)    Size (KB)
FP32       2.415       2.564       1402.0
PTQ        1.612       1.658       365.6
QAT        1.614       1.660       365.6
Binary     2.378       2.454       1402.0
```

### 6.2 Run DoS Flood Test and Sustained Load

Start the resource monitor in the background first, then run the DoS test:

```bash
cd ~/tinyml_project

# Start CPU and memory monitor in background
python3 resource_monitor.py results/monitor_log.csv &

# Run DoS test
python3 run_dos_test.py
```

This runs both the flood test (concurrency levels 1, 5, 10, 25, 50, 100 for 30 seconds each) and the sustained load test (300 seconds single-threaded). Results are saved to:

```
results/dos_flood_results.csv
results/dos_sustained_results.csv
results/monitor_log.csv
```

> **Note:** The full DoS test takes approximately 40 minutes to complete. Do not close the terminal while it is running.

### 6.3 Copy Results Back to Mac

```bash
# From your Mac terminal:
scp -r admin@172.20.10.3:~/tinyml_project/results/ ~/Downloads/rpi4_results/
scp -r admin@172.20.10.2:~/tinyml_project/results/ ~/Downloads/rpi5_results/
```

---

## 7. Running the Demo

The demo requires VNC to be running. Connect via RealVNC Viewer before proceeding.

### Option A — Visual Demo Only

Open a terminal on the Pi desktop via VNC and run:

```bash
cd ~/tinyml_project
DISPLAY=:0 python3 demo_visual.py
```

This classifies one image from each of the 10 CIFAR-10 classes using all four models and displays the results visually.

### Option B — Full Demo (Parts 1 to 3)

```bash
cd ~/tinyml_project
DISPLAY=:0 python3 run_demo.py
```

This runs:
- **Part 1:** Model file sizes and loading
- **Part 2:** Inference time delay measurement
- **Part 3:** Visual inference demo

> **Note:** Part 4 (DoS test) is included in `run_demo.py` but is skipped for demo recording purposes as it takes 40 minutes. To run Parts 1 to 3 only, comment out `run_part4()` at the bottom of `run_demo.py`.

### Faster Demo for Recording

To show fewer images during recording, edit `run_demo.py` and change the number of classes from 10 to 4:

```python
# Find:
for class_idx in range(10):

# Change to:
for class_idx in range(4):
```

---

## 8. Results Files

| File | Contents |
|---|---|
| `pi4_pi_time_delay.csv` | RPi 4B: avg delay, P95, min, max per model |
| `pi4_dos_flood_results.csv` | RPi 4B: avg delay per concurrency level |
| `pi4_dos_sustained_results.csv` | RPi 4B: avg delay, P95, requests in 300s |
| `pi4_monitor_log.csv` | RPi 4B: CPU%, memory%, memory_used_mb every 0.5s |
| `pi5_pi_time_delay.csv` | RPi 5: same as above |
| `pi5_dos_flood_results.csv` | RPi 5: same as above |
| `pi5_dos_sustained_results.csv` | RPi 5: same as above |
| `pi5_monitor_log_rpi5.csv` | RPi 5: same as above |

---

## 9. Troubleshooting/ debugging

### ImportError: cannot import name 'ImageTk' from 'PIL'

```bash
pip3 install pillow --upgrade --break-system-packages
```

### RuntimeError: main thread is not in main loop

This is a harmless tkinter cleanup error during the DoS test. It does not affect results. To suppress it, add at the top of `run_demo.py`:

```python
import warnings
warnings.filterwarnings('ignore')
```

### VNC shows blank screen or fails to connect

```bash
sudo systemctl restart lightdm
sleep 5
sudo ls /tmp/.X11-unix/
```

If `X0` appears, try connecting again via RealVNC Viewer.

### Cannot connect to Pi via SSH

```bash
# Check the Pi is on the network:
ping 172.20.10.3

# If no response, check the Pi is powered on and connected to the hotspot
```

### ai-edge-litert not found

```bash
pip3 install ai-edge-litert --break-system-packages
```

## Key Experimental Parameters

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Train / Val / Test split | 45,000 / 5,000 / 10,000 |
| Random seed | 42 (all experiments) |
| PTQ calibration samples | 300 (from training set) |
| QAT fine-tune epochs | 15 (early stop patience 6) |
| Binary fine-tune epochs | 25 (early stop patience 8) |
| Fine-tune learning rate | 1e-4 |
| Attack tool | Adversarial Robustness Toolbox (ART) |
| Attack sample size | 1,000 images (seed 42) |
| Epsilon values | 0.01, 0.0314, 0.05, 0.1 |
| PGD iterations | 20 |
| PGD step size | epsilon / 4 |
| DoS concurrency levels | 1, 5, 10, 25, 50, 100 |
| Flood stage duration | 30 seconds per level |
| Sustained load duration | 300 seconds |
