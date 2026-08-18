# Developer Guide
**Project** Evaluating the Security Impact of Quantisation on TinyML Models Against Adversarial and Denial of Service Attacks

## 1. Project Overview
Evaluates how three quantisation methods: Post-Training Quantisation (PTQ), Quantisation-Aware Training (QAT), and binary quantisation, affect the security of a TinyML model deployed on the Raspberry Pi 4B (RPi 4B) and Raspberry Pi 5 (RPi 5).

The working pipeline is split across two hardware environments:
1. Google Colaboratory with a T4 GPU: model training (unquantised Full-Point 32 (FP32) model), three quantised variants (PTQ, QAT, binary), adversarial evaluation (Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD))
   
2. RPi 4B and RPi 5: hardware deployment, DoS testing, and inference time delay measurement

## 2. Github Repository Structure
```
tinyml_security/
|
|-- colab_notebook/ # Google Colab notebooks
|  |-- baseline_training.py # train the unquantised FP32 baseline CNN
|  |-- ptq.py # post-training quantisation
|  |-- qat.py # quantisation-aware training quantisation
|  |-- binary.py # binary quantisation
|  |-- adversarial_attack_evaluation.py # FGSM and PGD evaluation
|
|-- pi_scripts/ # the scripts on the RPis
|  |-- pi_model_utils.py # TFLite model loader utility file
|  |-- resource_monitor.py # cpu and memory logger
|  |-- run_dos_test.py # dos flood and sustained load testing
|  |-- measure_time_delay.py # inference time delay measurement
|  |-- demo_visual.py # visual inference demo
|  |-- run_demo.py # full demo scripts (4-part demo)
|
|-- models/ # trained and quantised model files
|  |-- baseline.tflite # unquantised FP32 baseline
|  |-- ptq_model.tflite # ptq int 8 model
|  |-- qat_model.tflite # qat int 8 model
|  |-- binary_model.tflite # binary float32 model
|__ Readme.md # this file showing the developer guide
```
## 3. Requirements being running any files

#### Google Colab

- All colab notebook run on Google Colab with a T4 GPU. The following libraries are used in each notebook:
```
tensorflow 2.20
keras 3.x
numpy
matplotlib
adversarial-robustness-toolbox (ART)
```
#### RPis

```
python 3.13
ai-edge-litert 1.0.1 # for the inference runtime
numpy
plutil
matplotlib
pillow
```
## 4. Google Colab Set up
4.1. Open the Google Colab: sign in using your Google account

4.2. Set runtime to T4 GPU

4.3. Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```
4.4. Set the project directory
```python
PROJECT_DIR = '/content/drive/Mydrive/tinyml_project'
```
4.5. Run the following files in order
   
|Notebook|Output|
|---|---|
|`baseline_training.py`|`baseline.keras`, `baseline.tflite`|
|`ptq.py`|`ptq_model.tflite`|
|`qat.py`|`qat_model.tflite`|
|`binary.py`|`binary_model.tflite`, `binary_packed.npz`|
|`adversarial.py`|ASR results|

4.6. Transfer TFLite model files to RPis

Saves all files locally, then copy them to RPis using `scp` command: 
```bash
scp ~/location_of_your_file_locally/baseline.tflite your_pi_address:~/tinyml_project/models/
scp ~/location_of_your_file_locally/ptq_model.tflite your_pi_address:~/tinyml_project/models/
scp ~/location_of_your_file_locally/qat_model.tflite your_pi_address:~/tinyml_project/models/
scp ~/location_of_your_file_locally/binary_model.tflite your_pi_address:~/tinyml_project/models/
```

Redo above `scp` commands with both RPi 4B and RPi 5 (Don't forget to change the address between 2 RPis)

## 5. Raspberry Pi Setup

These steps apply to both RPi 4B and RPi 5

5.1. Find the RPi's IP address

On your command prompt or terminal:
`arp-a` or `ssh admin@raspberrypi.local` (if you have costumed your hostname, use that instead)

5.2. SSH into the RPi
`ssh admin@raspberrypi.local`

5.3. Create the project directory
```bash
mkdir -p ~/tinyml_project/models
mkdir -p ~/tinyml_project/results
mkdir -p ~/tinyml_project/demo_outputs
```

5.4. Install required packages
```bash
pip3 install ai-edge-litert --break-system-packages
pip3 install numpy psutill matplotlib pillow --break-system-packages
```
5.5. Copy the scripts to RPis

From your command prompt/ terminal:
```bash
scp pi_scripts/pi_model_utils.py your_pi_address:~/tinyml_project/
scp pi_scripts/resource_monitor.py your_pi_address:~/tinyml_project/
scp pi_scripts/run_dos_test.py your_pi_address:~/tinyml_project/
scp pi_scripts/measure_time_delay.py your_pi_address:~/tinyml_project/
scp pi_scripts/demo_visual.py your_pi_address:~/tinyml_project/
scp pi_scripts/run_demo.py your_pi_address:~/tinyml_project/
```

5.6. Set up the VNC remote monitor

VNC is require in this project only for running the visual demo. Personally, I used Screen5:VNC Remote Desktop and RealVNC Connect Viewer -- either of them works fine.

```bash
# install VNC server and desktop
sudo apt-get install realvnc-vnc-server lightdm lightdm-gtk-greeter lxde --yes

# to fix lightdm configuration
sudo sed -i 's/user-session=rpd-labwc/user-session=LXDE/' /etc/lightdm/lightdm.conf
sudo sed -i 's/greeter-session=lightdm-gtk-greeter-labwc/greeter-session=lightdm-gtk-greeter/' /etc/lightdm/lightdm.conf
sudo sed -i 's/autologin-session=rpd-labwc/aitplogin-session=LXDE/' /etc/lightdm/lightdm.conf

# to start the service
sudo systemctl enable vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced
sudo systemctl start lightdm
```
Then, connect your monitor to each RPi's address.

## 6. Running the experiments

6.1. Measure the inference time delay

- ssh to RPi and run:
```bash
cd ~/tinyml_project
python3 measure_time_delay.py
```
and the results is saved to `results/pi_time_delay.csv`

The expected results:
```
Model    Avg(ms)    P95(ms)    Size(KB)
FP32      xx.xx      xx.xx      xx.xx
PTQ       xx.xx      xx.xx      xx.xx
QAT       xx.xx      xx.xx      xx.xx
Binary    xx.xx      xx.xx      xx.xx
```

6.2. Run DoS Flood Test and sustained load

Start the resource monitor in the background first then run the DoS test:
```bash
cd ~/tinyml_project

# start cpu and memory monitor in the background
python3 resource_monitor.py results/monitor_log.csv &
# Run DoS test
python3 run_dos_test.py
```
The results are saved to
```
results/dos_flood_results.csv
results/dos_sustained_results.csv
results/monitor_log.csv
```
**Note:** The full DoS run takes around 40-45 minutes to complete. Don't close the terminal while it's running.

6.3. Copy the results back to your local computer

Using this following `scp` command:
```bash
scp -r your_pi_hostname@your_pi_address:~tinyml_project/results/ ~/your_local_computer_location
```

## 7. Running the live demo

The demo scripts required VNC to be running, so please connect the VNC before proceeding. Use LXterminal on VNC to run the demo.

Option A - Visual Demo only (This run classifies one image from each of the CIFAR-10 classes using all 4 models and display the result's prediction)

Open LXterminal on Pi Desktop via VNC to run:
```bash
cd ~/tinyml_project
DISPLAY=:0 python3 demo_visual.py
```
Option B - Full demo (There are 4 parts)
```bash
cd ~/tinyml_project
DISPLAY=:0 python3 run_demo.py
```
This runs 4 parts: 
1. Model file size measurement

2. Inference time delay measurement

3. Visual inference demo

4. DoS test (*Note:* this is a short version of the full DoS run) 

## 8. Result file explanation
|File|Explanation|
|---|---|
|`pi_time_delay.csv`|avg delay, P95 delay, min delay, max delay per model|
|`dos_flood_results.csv`|avg delay per concurrency level|
|`dos_sustained_results.csv`|avg delay, P95 delay, no. of requests in 300s|
|`monitor_log.csv`|CPU usage(%), memory usage (%), memory_used_mb every 0.5 s|

## 9. Debugging/ Troubleshooting
#### ImportError: cannot import name 'ImageTk' from 'PIL'

To solve, try: `pip3 install pillow --upgrade --break-system-packages`

#### VNC shows blank screen or fails to connect

To solve, try:
```bash
sudo systemctl restart lightdm
sleep 5
sudo ls /tmp/.X11-unix/
```
If `X0` appears after running above code, try connecting again on VNC monitor.

#### If RPi is not found 
Try: `ping your_pi_address` if still not working, check the power connection.

#### If ai-edge-litert not found
Try to install: `pip3 install ai-edge-litert --break-system-packages` 

## Key parameters I used in this project
|Parameter|Value|
|---|---|
|Dataset|CIFAR-10 (60,000 images, 10 classes)|
|Train/ Val/ Test Split| 45,000/ 5,000/ 10,000 respectively|
|Random Seed| 42 (for all experiments)|
|PTQ calibration samples| 300 (from training set)|
|QAT fine-tune epochs| 15 (early stop patience 6)|
|Binary fine-tune epochs| 25 (early stop patience 8)|
|Fine-tune learning rate| 1e-4|
|Attack Tool| Adversarial Robustness Toolbox (ART)|
|Attack Sample Size| 1,000 images (seed 42)|
|Epsilon values (perturbed strengths)|0.01, 0.0314(8/255), 0.05, 0.1|
|PGD iterations| 20|
|PGD step size| epsilon/4|
|DoS concurrency levels| 1,5,10,25,50,100|
|Flood stage duration| 30 seconds per level|
|Sustained load duration| 300 seconds|

## End of Developer Guide
