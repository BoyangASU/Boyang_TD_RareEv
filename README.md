# CNN + TD(λ) for RUL Prediction on CMAPSS

This repository demonstrates how to train a Convolutional Neural Network (CNN) feature extractor and a linear value head using **temporal-difference (TD) learning** methods for Remaining Useful Life (RUL) prediction. The NASA C-MAPSS dataset is used as the principal benchmark.

---

## Table of Contents

- [Overview](#overview)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

1. **Objective**  
   Predict the Remaining Useful Life (RUL) of an aircraft engine using a combination of CNN-based feature extraction and TD(λ) reinforcement learning updates.

2. **Dataset**  
   - The NASA [C-MAPSS dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) is used.
   - Place your data files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) in the `CMAPSSData/` folder.

3. **Approach**  
   - We employ a CNN for feature extraction and a linear layer for final RUL estimation.
   - **Manual TD(λ) updates**: No typical optimizer; parameters (partially or fully) are updated through TD error and eligibility traces.

---

## File Descriptions

### 1. `td_all_features.py`
- **Description**:  
  - A TD(λ) approach that **unfreezes all CNN parameters**, allowing both the CNN feature extractor and the linear head to be updated by the TD error.
  - Demonstrates manual TD updates for each window of data.

- **Key Points**:
  - CNN1D-based `FeatureExtractorCNN` that outputs a feature vector for each time window.
  - `LinearValueFunction` to produce a scalar RUL estimate.
  - Implements manual eligibility trace updates, computing TD error at each time step/window.

### 2. `td_all_features_exp.py`
- **Description**:  
  - Runs experiments across multiple `(lambda, gamma)` pairs using the **same approach as `td_all_features.py`** (i.e., all CNN parameters are trainable).
  - Evaluates performance (RMSE) of each configuration.
  - Outputs a **LaTeX-formatted table** with the RMSE for each `(lambda, gamma)` pair.

- **Key Points**:
  - Loops over a grid of λ and γ values.
  - Logs RMSE results, saves best models, and prints a table showing performance metrics.

### 3. `td_freeze.py`
- **Description**:  
  - A TD(λ) script that **freezes the CNN feature extractor** parameters and **only trains** the linear head.
  - Uses **PyTorch 2.0** `torch.func` (including `vmap` and `grad`) for **per-sample gradient** computation in a vectorized way.

- **Key Points**:
  - CNN parameters remain fixed (frozen), reducing computation and focusing on how effectively the linear head can learn.
  - Demonstrates advanced PyTorch functional APIs for more efficient batch-level updates.

### Additional Modules

- **`preprocessing.py`**  
  Contains methods to load, clean, and prepare CMAPSS data. Functions like `prepare_data()`, `process_input_data_with_targets()`, and `process_targets()` handle data preprocessing and windowing.

- **`train.py`**  
  Defines the `RULDataset` class for PyTorch’s `DataLoader`. It may also contain supporting models such as `CNN1D` for baseline comparisons.

---

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (for `torch.func` in `td_freeze.py`)
- **NumPy**, **pandas**, **scikit-learn**, and **tqdm**

Install dependencies (adjust versions as needed):

```bash
pip install torch numpy pandas scikit-learn tqdm
