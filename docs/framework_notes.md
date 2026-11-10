# Pipeline Framework Architecture

This document outlines the end-to-end framework architecture derived from the pipeline methodology, capturing the phase sequence, inputs, outputs, validation checkpoints, and file hand-offs.

---

## 1. Pipeline Overview & Phase Sequence

The framework follows a modular, phase-based execution pipeline where each phase executes an isolated data science responsibility, validates its intermediate artifacts with assertions and visualizations, and persists clean files consumed by subsequent phases:

```
[Raw Data: train.csv, test.csv, sample_submission.csv]
       │
       ▼
Phase 1: Exploration ──► Informs cleaning decisions & sentinel identification
       │
       ▼
Phase 2: Cleaning ──► outputs/clean_data.npz
       │
       ▼
Phase 2b: Residual Noise Check ──► outputs/clean_data_v2.npz
       │
       ├──► Phase 3: Validation Strategy ──► outputs/splits.npz
       │                                             │
       ▼                                             │
Phase 4: Feature Extraction ──► outputs/features.npz  │
       │                                             │
       └──────────────────────┬──────────────────────┘
                              ▼
Phase 5: Model Comparison & Submission ──► outputs/submission.csv
       │
       ▼
Phase 6: Project Synthesis & Reporting (docs/report_notes.md)
```

---

## 2. Phase Contract & Hand-off Specification

### Phase 1: Data Exploration
- **Inputs:** `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.
- **Core Operations:**
  - Verify dataset dimensions, identifier uniqueness, and schema symmetry between train and test.
  - Audit raw column formats, time-major ordering, channel counts, and step counts per window.
  - Scan for unparsed non-numeric string tokens, NaN patterns, and numerical sentinels.
  - Profile target label distributions and majority-class baseline benchmarks.
  - Compute per-channel summary statistics (mean, spread, extrema) and evaluate train-test distributional drift.
  - Visualize raw signals across activity classes to assess domain properties.
- **Outputs:** In-memory diagnostic profiling that dictates the cleaning rules for Phase 2. No disk file generated.

---

### Phase 2: Systematic Cleaning & Reshaping
- **Inputs:** `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.
- **Core Operations:**
  - Parse and normalize text tokens, malformed decimal strings, and explicit sentinels into numerical missing values.
  - Deduplicate training records (verifying label consistency across identical input sequences).
  - Reshape 1D tabular records into 3D multivariate window tensors `(n_windows, n_time_steps, n_channels)`.
  - Impute missing / corrupted samples strictly within-window using local temporal interpolation.
  - Enforce zero leakage: no cross-window or dataset-wide statistical aggregation.
  - Validate that all missing values, non-numeric tokens, and sentinels are eliminated.
- **Output Artifact:** `outputs/clean_data.npz`
  - Keys: `X_train`, `y_train`, `train_ids`, `X_test`, `test_ids`, `channels`.
- **Downstream Consumers:** Phase 2b (or Phase 3 and Phase 4).

---

### Phase 2b: Residual Noise & Spike Assessment
- **Inputs:** `outputs/clean_data.npz`.
- **Core Operations:**
  - Evaluate residual high-frequency artifacts or unphysical discontinuities that survive standard linear cleaning.
  - Conduct channel-specific physics/physiology audits before applying filters (e.g. preserving genuine physiological ECG spikes while filtering mechanical accelerometer impulse noise).
  - Apply adaptive, robust local residual filtering (e.g., median filtering and MAD-based thresholding) strictly within individual windows.
  - Compare flagging rates between train and test sets to prevent distortion of clean test distributions.
- **Output Artifact:** `outputs/clean_data_v2.npz`
  - Keys: `X_train`, `y_train`, `train_ids`, `X_test`, `test_ids`, `channels`.
- **Downstream Consumers:** Phase 3 and Phase 4.

---

### Phase 3: Validation Strategy
- **Inputs:** `outputs/clean_data_v2.npz` (or `clean_data.npz`).
- **Core Operations:**
  - Determine whether natural group/subject identifiers exist in the data.
  - Since test data originates from unseen subjects, establish a cross-validation partitioning scheme that mimics unseen-subject generalization.
  - If subject IDs are absent, develop an honest proxy partition (e.g. clustering subject-level posture/biomechanical signatures globally across windows) ensuring complete class representation across all folds.
  - Audit validation difficulty: compare proxy-group cross-validation against standard random stratified K-Fold.
  - Enforce independence: verify that fold definitions do not leak target-dependent feature sets.
- **Output Artifact:** `outputs/splits.npz`
  - Keys: `folds`, `scheme`.
- **Downstream Consumers:** Phase 5.

---

### Phase 4: Feature Extraction
- **Inputs:** `outputs/clean_data_v2.npz`.
- **Core Operations:**
  - Transform raw multivariate time-series windows into fixed-dimension tabular feature vectors.
  - Engineer feature families grounded in domain physics:
    - Time-domain statistics (moments, extrema, quantiles, peak counts).
    - Frequency-domain spectral features (spectral energy bands, dominant frequencies, spectral entropy).
    - Biomechanical sensor kinematics (3D magnitude vectors, inter-axis correlations).
    - Posture and dynamic acceleration separation (low-pass gravity decomposition vs dynamic motion).
  - Enforce zero leakage: feature computation is strictly row-wise / window-wise; no global scaling or fit-based transformations are applied here.
  - Audit feature vectors for finiteness, non-constant variance, and schema alignment between train and test.
- **Output Artifact:** `outputs/features.npz`
  - Keys: `F_train`, `F_test`, `y_train`, `names`.
- **Downstream Consumers:** Phase 5.

---

### Phase 5: Model Exploration, Selection, & Submission
- **Inputs:** `outputs/features.npz`, `outputs/splits.npz`, `outputs/clean_data_v2.npz`.
- **Core Operations:**
  - Construct a comparative benchmark suite across model classes:
    - Baseline (Dummy / Majority Class)
    - Linear regularized models (Logistic Regression)
    - Distance-based models (k-Nearest Neighbors)
    - Non-linear kernel machines (Support Vector Classifier)
    - Tree ensembles (Random Forest, Gradient Boosting / LightGBM)
  - Encapsulate data preprocessing (e.g., standard scaling, variance thresholding) inside cross-validation pipelines to guarantee zero validation leakage.
  - Perform out-of-fold cross-validation using the exact fold assignments from Phase 3.
  - Rigorously inspect any outlier model performance for data leakage or optimistic evaluation bias.
  - Analyze performance via cross-validated confusion matrices, classification reports, and feature importance.
  - Retrain the selected best model on the complete training set and predict on `F_test`.
  - Validate submission formatting (exact row count, exact column headers `id,target`, non-empty predictions, valid class names).
- **Output Artifact:** `outputs/submission.csv`
  - Required format: `id,target`.

---

### Phase 6: Synthesis & Report Documentation
- **Inputs:** Outputs and metrics logged across all phases.
- **Core Operations:**
  - Synthesize empirical findings, pipeline design choices, validation results, known limitations, and critical risks into `docs/report_notes.md`.
  - Guarantee that every quantitative claim reflects verified, executed outputs.

---

## 3. Core Framework Rules & Leakage Guards

1. **Isolation of Phase Boundaries:** Each phase reads from disk and writes to disk, ensuring that all processing steps are reproducible and decoupled.
2. **Strict Per-Window Cleaning:** Preprocessing in Phases 2 and 2b must never borrow information across windows, subjects, or time steps outside the active window.
3. **Pipeline-Encapsulated Scaling:** Any data-driven transformation (mean subtraction, standard scaling, dimensionality reduction) must be placed inside `sklearn.pipeline.Pipeline` or fitted solely on training folds during cross-validation.
4. **Validation Integrity:** The validation scheme must test generalization across independent recording units (e.g., unseen subjects), matching competition test conditions.
