# AD4HEP 2025 Tutorial

This repository contains a hands-on tutorial notebook for training and evaluating simple ML models (teacher/student “co-learning” style) on **MNIST**, and then **converting a trained Keras model to FPGA-friendly firmware using `hls4ml`**.

The tutorial is designed to be run end-to-end in the included notebook:
- `ad4hep_tutorial.ipynb`

---

## Repository contents

- **`ad4hep_tutorial.ipynb`**  
  Main tutorial notebook: environment setup, dataset loading, model construction, co-learning training, evaluation, and `hls4ml` conversion.

- **`plotting.py`**  
  Plotting + evaluation utilities used by the notebook (loss curves, ROC/PR curves, score distributions, etc.).

- **`requirements.txt`**  
  Pinned Python dependencies for the tutorial environment (notebook stack, ML libraries, and `hls4ml`).

---

## Quickstart

### 1) Create a virtual environment (Python 3.10 or 3.11 recommended)

~~~bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
~~~

### 2) Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### 3) Run the notebook

~~~bash
jupyter lab
~~~

Open `ad4hep_tutorial.ipynb` and run the cells top-to-bottom.

---

## What the notebook covers

1. **Environment setup**
   - Assumes Python 3.10/3.11 and a clean environment workflow.

2. **Dataset**
   - Downloads and preprocesses **MNIST** (handwritten digits) and constructs a “normal vs anomaly” setup by choosing a target digit.

3. **Model construction**
   - Builds multiple models of increasing complexity (intended for comparing parameter counts / performance).

4. **Co-learning training**
   - Trains a teacher/student setup (student learns to match teacher behavior and produce anomaly scores).

5. **Evaluation**
   - Generates common evaluation plots (ROC, PR, score distributions) and summary metrics.

6. **`hls4ml` conversion**
   - Converts a trained Keras model to an `hls4ml` project, compiles it, runs inference, and compares performance before/after conversion.

---

## Plotting & evaluation utilities (`plotting.py`)

The helper module includes:
- Loss curves (train/val)
- ROC and precision–recall curves
- Anomaly-score distributions
- Wasserstein / Earth-Mover’s Distance comparisons
## Notes

- If you want to run the full FPGA synthesis flow, you’ll typically need a supported HLS toolchain (e.g., Vivado/Vitis HLS) configured on your machine in addition to Python packages. The notebook’s `hls4ml` steps can still be useful for conversion + software-level validation without full synthesis.

---
