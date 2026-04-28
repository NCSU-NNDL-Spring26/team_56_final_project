# ECE525-ECG-Classification-Project

Predicting cardiac arrhymias from electrocardiogram (ECG) heartbeat signals using the MIT-BIH Arrhythmia Database. The project aligns with course expectations through exploring and analyzing real data, creating a random forest as the baseline model, and improving on the baseline model with a custom neural network. The neural network was trained and evaluated using practices learned from this course. 

**Dataset:** [MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)

---

## Clinical Motivation

Cardiovascular disease remains one of the leading causes of death worldwide. To help reduce mortality rates in patients, arrhythmias must be detected early. Using a neural network to detect early signs of arrhythmias can improve patient outcomes, reduce emergency cardiac events, and support clinicians through faster ECG interpretation. 

## Technical Motivation

Create a reproducible pipeline from raw ECG waveform records to heartbeat classification. Evaluate traditional machine learning baseline models (random forest) with a custom convolutional neural network (CNN), and evaluate the results using practices learned from this class.

---

## Repository Contents

|Folder / File | Description |
|--------------|-------------|
| 'data/data/' | MIT-BIH raw ECG signal files and annotations |
| 'Baseline_Model/' | Random Forest Baseline Model |
| 'Neural_Network/' | PyTorch CNN Implementation |
|  'ReadME.md' | Project Overview and Setup Instructions |

---

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ECG-Arrhythmia-Classification-Team-56
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch wfdb
```

### 3. Run the Models

### Baseline Model (Random Forest)

Open the notebook inside:

'Baseline_Model/'

Run all cells to reproduce Random Forest baseline model results.

### Neural Network

Navigate to:

'Neural_Network/'

Then run:

```bash
python train.py
```

## Methodology Review

### Random Forest

Traditional machine learning model trained using extracted heartbeat features:

- Mean
- Standard Deviation
- Minimum Amplitude
- Maximum Amplitude
- Peak-to-Peak Range
- RMS Value
- Signal Energy

#### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Neural Network Model (1D CNN)

Raw ECG beat windows of 400 samples are extracted around the annotated beats and normalized individually.

#### CNN Architecture

```text
Conv1D(16) → BatchNorm → ReLU → MaxPool
Conv1D(32) → BatchNorm → ReLU → MaxPool
Conv1D(64) → BatchNorm → ReLU → MaxPool
Flatten → Dense(128) → Dropout → Output(5 Classes)
```

#### Target Classes

- Normal
- SVEB (Supraventricular Ectopic Beat)
- VEB (Ventricular Ectopic Beat)
- Fusion Beat
- Unclassifiable

#### Training Details

- Adam Optimizer
- Weighted Cross-Entropy Loss
- GPU Supports (CUDA if available)
- Reproducible Random Seed
