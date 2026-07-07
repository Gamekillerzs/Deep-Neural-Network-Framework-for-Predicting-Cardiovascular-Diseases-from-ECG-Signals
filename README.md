# Deep Neural Network Framework for Predicting Cardiovascular Disease from ECG Signals

A deep neural network (DNN) for predicting cardiovascular disease from 12-lead ECG signals, comparing three optimizers — **Adam**, **AdaGrad**, and **AdaDelta** — on the PTB-XL+ dataset.

## Overview

Cardiovascular disease (CVD) is the leading cause of death worldwide, encompassing conditions like coronary artery disease, heart failure, and arrhythmias. ECGs are a standard diagnostic tool, but manual interpretation isn't always accurate. This project trains a DNN to classify ECG recordings into **Normal (NORM)**, **Myocardial Infarction (MI)**, and **Other** (ST/T change, conduction disturbance, and hypertrophy combined), then compares training with three different optimizers to identify which gives the best predictive performance.

## Dataset

- **PTB-XL+** — a comprehensive electrocardiographic feature dataset from [PhysioNet](https://physionet.org/), containing 21,799 clinical 12-lead ECGs from 18,869 patients, each 10 seconds long.
- As with the related CNN/RNN comparison work, the model is trained on **extracted ECG features** (RR interval, ST elevation, PR/PQ intervals, QRS duration, QT interval, and R/Q/P wave amplitudes across all 12 leads) rather than raw waveforms.

> Note: The dataset is not included in this repository. Update the data-loading paths in the notebook to point to your own copy of PTB-XL+ (and its extracted feature set) before running.

### Class distribution

| Class | % of dataset (raw) |
|---|---|
| Normal | 55.8% |
| Myocardial Infarction | 15.6% |
| ST/T Change | 14.8% |
| Conduction Disturbance | 10.5% |
| Hypertrophy | 3.3% |

ST/T Change, Conduction Disturbance, and Hypertrophy are merged into a single **Other** class after feature extraction, giving a 3-class split of roughly Normal 56%, MI 16%, Other 28%. Data is split 80% training / 10% validation / 10% test after preprocessing.

## Repository Contents

| File | Description |
|---|---|
| `DNN_for_ECG_Signals.ipynb` | Data loading, feature extraction/preprocessing, and the DNN model trained with three optimizers |

## Model Architecture

```
Dense(256, activation='sigmoid')
BatchNormalization()
Dense(128, activation='sigmoid')
BatchNormalization()
Dense(64, activation='sigmoid')
Dense(3, activation='sigmoid')   # output layer, one unit per class
```

- **Loss:** categorical cross-entropy
- **Optimizers compared:** Adam, AdaGrad, AdaDelta (three separate training runs, same architecture)
- **Training parameters:** batch size 32, 50 epochs

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab
- TensorFlow / Keras
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Usage

1. Download the PTB-XL+ dataset (and its extracted ECG feature set) from PhysioNet and update the relevant paths in the notebook.
2. Open `DNN_for_ECG_Signals.ipynb` in Jupyter.
3. Run the cells sequentially to preprocess the data and train/evaluate the DNN under each optimizer.

## Results

| Optimizer | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Adam** | **85%** | **85.33%** | **85.33%** | **85.90%** |
| AdaGrad | 71% | 71.33% | 71.66% | 71.33% |
| AdaDelta | 68% | 67.66% | 67.66% | 67.66% |

Adam clearly outperformed AdaGrad and AdaDelta across every metric, making it the recommended optimizer for this architecture and dataset. Training/validation curves (loss and accuracy) for all three optimizers are included in the notebook and the associated paper.

## Citation

If you use this work in your research, please cite:

```
Soni, T., Gupta, D., Uppal, M., Juneja, S., Gulzar, Y. and Ghafoor, K.Z., 2024.
Deep neural network framework for predicting cardiovascular diseases from ECG signals.
Recent Advances in Computer Science and Communications.
```

## License

Specify a license (e.g., MIT, Apache 2.0) for this repository if you intend to allow others to reuse the code. Check your publisher's terms for the article itself, since those may differ from any license you choose for the code.

## Contact

For questions or collaboration, feel free to open an issue on this repository.
