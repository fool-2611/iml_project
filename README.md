# Multimodal Human Activity Recognition Pipeline

An end-to-end Machine Learning pipeline consolidating data exploration, tensor reconstruction, domain-aware signal despiking, subject-independent validation, physics-based feature engineering, model benchmarking, and test inference into a single cohesive Jupyter Notebook (`iml_project_complete.ipynb`).

---

## Project Overview

The objective is to classify 12 distinct physical human activities from 23-channel multimodal physiological and kinematic sensor streams sampled at 50 Hz (5-second windows of 250 time steps per sample). Sensors include ECG (chest) and 3-axis accelerometer, gyroscope, and magnetometer streams across three body placements: chest, wrist, and ankle.

The pipeline addresses several critical real-world signal processing and machine learning challenges:
1. **Missing Data & Formatting**: Reconstructing raw flattened feature representations into 3D time-series tensors `(N, 250, 23)` and repairing `9999.0` sentinel dropout bursts with linear within-window interpolation.
2. **Duplication & Permutation Tests**: Identifying and removing 88 duplicate rows (44 exact pairs) and proving temporal independence across windows ($p = 0.788$).
3. **Domain-Specific Despiking**: Resolving 5,736 single-sample electrical spikes (>5.8 mV) on ECG leads while preserving valid cardiac QRS complexes, aligning test/train standard deviation ratios from 0.33 to 0.71.
4. **Leakage-Free Validation**: Exposing that standard random K-Fold cross-validation suffers from severe optimistic subject leakage (~100% artificial accuracy). Using unsupervised posture clustering and Hungarian bipartite matching to recover subject boundaries and establish strict 5-fold GroupKFold validation.
5. **Feature Engineering**: Extracting 581 features spanning time-domain moments, Hann-windowed FFT spectral energy bands, 3D kinematic magnitudes, inter-axis correlations, gravitational tilt angles, and heart rate variability (HRV).
6. **Model Benchmarking**: Systematically comparing 7 model families under strict group CV. Random Forest achieves **98.58% Out-of-Fold Accuracy** (98.66% ± 2.39%).

---

## Pipeline Architecture

The entire workflow is structured into 6 sequential phases in `iml_project_complete.ipynb`:

| Phase | Description | Key Output Artifact |
|---|---|---|
| **Phase 1: Exploratory Data Analysis** | Raw schema inspection, class balance audit, token/sentinel checks, temporal sequence independence permutation tests. | Visualizations in `figures/01_*.png` |
| **Phase 2: Data Cleaning & Reshaping** | Sentinel replacement, 3D tensor reshaping `(635, 250, 23)`, exact deduplication, bounded linear interpolation. | `outputs/clean_data.npz` |
| **Phase 3: Residual Filtering & Despiking** | Running median ($k=6.0$) on inertial channels, ECG-specific despiker preserving QRS peaks, test-train scale alignment. | `outputs/clean_data_v2.npz` |
| **Phase 4: Subject Discovery & Validation** | Posture clustering ($k=5$), Hungarian bipartite cross-activity matching, leak-free 5-fold GroupKFold split. | `outputs/splits.npz` |
| **Phase 5: Feature Engineering** | 581 domain features (time-domain moments, Hann FFT bands, 3D norms, tilt angles, axis correlations, ECG HR/HRV). | `outputs/features.npz` |
| **Phase 6: Benchmarking & Submission** | Model comparisons, feature importances, sensor ablation, confusion matrix, and test prediction generation. | `outputs/submission.csv` |

---

## Model Performance Benchmark

Models evaluated under leak-free 5-Fold Group Cross-Validation:

| Model Architecture | OOF Accuracy | Mean Fold Accuracy (± Std) |
|---|---|---|
| **Random Forest (Best)** | **98.58%** | **98.66% (± 2.39%)** |
| Extra Trees | 98.43% | 98.51% (± 2.52%) |
| LightGBM | 97.95% | 98.03% (± 2.80%) |
| Ensemble (Soft Voting) | 98.43% | 98.51% (± 2.45%) |
| Gradient Boosting | 97.17% | 97.28% (± 3.12%) |
| Support Vector Classifier (RBF) | 96.69% | 96.79% (± 3.41%) |
| Logistic Regression | 94.33% | 94.41% (± 4.05%) |

---

## Repository Structure

```
.
├── iml_project_complete.ipynb    # Consolidated master Jupyter Notebook (122 cells)
├── README.md                     # Project documentation & pipeline guide
├── requirements.txt              # Pinned Python package versions
├── data/                         # Sensor datasets
│   ├── train.csv                 # Labeled training set (679 raw rows)
│   ├── test.csv                  # Unlabeled test set (403 raw rows)
│   └── sample_submission.csv     # Sample submission format
├── outputs/                      # Generated pipeline artifacts
│   ├── clean_data.npz            # Interpolated 3D tensor arrays
│   ├── clean_data_v2.npz         # Despiked 3D tensor arrays
│   ├── splits.npz                # GroupKFold fold indices
│   ├── features.npz              # 581 engineered features & labels
│   └── submission.csv            # Final test predictions (403 rows, 100% valid)
└── figures/                      # High-resolution visualizations generated by notebook
    ├── 01_*.png                  # EDA & signal corruption profiles
    ├── 02_*.png                  # Cleaning & tensor verification plots
    ├── 03_*.png                  # Despiking comparisons & ECG audit
    ├── 04_*.png                  # Posture clustering & validation fold heatmaps
    ├── 05_*.png                  # Feature family & correlation heatmaps
    └── 06_*.png                  # Model benchmark, confusion matrix & ablation
```

---

## Setup & Execution

### Prerequisites
Python 3.10+ with standard scientific computing libraries. All exact dependencies are listed in `requirements.txt`.

### Installation
```bash
pip install -r requirements.txt
```

### Running the Notebook
Execute the notebook end-to-end to regenerate all artifacts and figures:
```bash
jupyter nbconvert --to notebook --execute --inplace iml_project_complete.ipynb
```
The notebook executes cleanly from top to bottom with zero external script dependencies.
