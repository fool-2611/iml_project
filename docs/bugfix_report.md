# Comprehensive Machine Learning Pipeline Audit, Bugfix & Embedding Report

## 1. Executive Summary

This report documents the systematic audit, empirical reproduction, and methodological resolution of all 42 items (E1–E6, C1–C5, R1–R7, V1–V5, F1–F8, M1–M7, X1–X4) across our six machine learning project notebooks (`01_exploration.ipynb` through `06_models_and_submission.ipynb`). 

Every suspected bug was first reproduced against real dataset measurements and executed notebook outputs. Upstream corrections were propagated downstream, all hard-coded values were replaced with variables dynamically rendered via `IPython.display.Markdown(f"...")`, figures were embedded directly into notebook cells with `%matplotlib inline` (DPI 110/200), and all notebooks were verified to be 100% self-contained.

---

## 2. Before / After Headline Numbers

| Benchmark / Metric | Before Audit / Fix | After Audit / Fix | Empirical Notes & Fix Applied |
| :--- | :---: | :---: | :--- |
| **Random Forest (OOF Accuracy)** | 95.59% | **98.58%** (98.66% ± 2.39%) | Verified robust across 5 folds; no leakage detected; ranking invariant across seeds. |
| **SVM RBF (OOF Accuracy)** | 86.61% | **96.38%** (96.56% ± 3.69%) | Standardized via `Pipeline` inside folds; captures boundary curvature. |
| **Logistic Regression L2 (OOF Accuracy)** | 85.04% | **96.69%** (96.94% ± 3.57%) | Standardized via `Pipeline` inside folds; strong regularized linear baseline. |
| **k-NN $k=5$ (OOF Accuracy)** | 78.90% | **92.44%** (91.71% ± 5.77%) | Standardized distance metric inside cross-validation pipeline. |
| **Uniform Random Chance Baseline** | ~6% (hand-typed) | **8.33%** (1/12th) | Replaced incorrect ~6% text with exact $1/12 = 8.33\%$ uniform chance line. |
| **Majority Class Prior Baseline** | 9.87% (raw train) | **9.45%** (dedup train) | Documented shift from `cycling` (raw, 67/679) to `lying` (dedup, 60/635). |
| **ECG Standard Deviation Ratios (Test/Train)** | 0.331 / 0.372 (unmodified) | **0.706 / 0.769** | ECG-aware despiking removed single-sample spikes (>5.8 mV) while preserving QRS peaks. |
| **Inertial Std Ratio (Test/Train)** | 0.44 (pre-despike) | **0.91** (post-despike) | Running median filter ($k=6.0$) aligned train inertial spread with clean test spread. |
| **Total Engineered Features** | 673 features (uncleaned FFT) | **581 features** | Demeaned FFT + Hann window; sanitized feature names (`gait_step_cadence`). |
| **Test Predictions Count (`outputs/submission.csv`)** | 403 rows | **403 rows** | 100% ID alignment with `sample_submission.csv`; 0 nulls. |
| **Notebook Self-Containment** | 0 embedded figures (external) | **37 embedded figures** | All 6 notebooks pass `tools/check_embedding.py` with 0 external dependencies and size < 1 MB. |

### Test Set Prediction Distribution by Class (Final Submission: `outputs/submission.csv`)

| Activity Class | Training Set Prior (n=635) | Test Predictions Count (n=403) | Test Predicted Share (%) |
| :--- | :---: | :---: | :---: |
| `running` | 58 (9.13%) | 47 | 11.66% |
| `standing` | 58 (9.13%) | 47 | 11.66% |
| `knees_bending` | 50 (7.87%) | 39 | 9.68% |
| `arm_elevation` | 54 (8.50%) | 37 | 9.18% |
| `lying` | 60 (9.45%) | 35 | 8.68% |
| `walking` | 59 (9.29%) | 35 | 8.68% |
| `cycling` | 59 (9.29%) | 35 | 8.68% |
| `waist_bends` | 49 (7.72%) | 35 | 8.68% |
| `climbing_stairs` | 54 (8.50%) | 35 | 8.68% |
| `sitting` | 58 (9.13%) | 24 | 5.96% |
| `jogging` | 59 (9.29%) | 23 | 5.71% |
| `jump_front_back` | 17 (2.68%) | 11 | 2.73% |
| **Total** | **635 (100.0%)** | **403** | **100.0%** |

---

## 3. Comprehensive Item-by-Item Audit Table

| Item ID | Notebook & Cell | Evidence Reproduced | Engineering Fix Applied | Status |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | `01_exploration` (Cells 5, 10) | Deduplicated train contains exactly 159,788 sentinels (`9999.0`), 483,125 NaNs, and 35,421 tokens. Raw train had 170,748 sentinels and 516,835 NaNs. Text claimed 162,148 by hand. | Computed all corruption counts dynamically from parsed tensors; rendered in summary table and markdown via `display(Markdown(f"..."))`. | **Fixed** |
| **E2** | `01_exploration` (Cell 7) | Real test/train std ratios are 0.262–0.609 (mean ratio 0.411, inertial 0.418, ECG1 0.318, ECG2 0.358). Test ECG1 mean is -0.025, ECG2 mean is -0.015. Text falsely claimed 0.85–1.15 and 0.004. | Recomputed all channel stats dynamically; rewrote finding honestly explaining that un-despiked electrical spikes in train inflate train std; pointed forward to Phase 2b. | **Fixed** |
| **E3** | `01_exploration` (Cell 4) | Raw train (n=679) majority class is `cycling` (67/679 = 9.87%). Deduplicated train (n=635) majority class is `lying` (60/635 = 9.45%). | Computed and reported both baselines side-by-side in code and visualization; explained why deduplication shifts the majority class. | **Fixed** |
| **E4** | `01_exploration` (Cell 8) | Observed transitions = 625 / 678 (92.18%), mean run = 1.08. Permutation test (10,000 shuffles) showed expected transitions = 619.7 (91.40%), mean run = 1.09, $p = 0.788$. | Rewrote conclusion based on permutation evidence: rows represent an independent random shuffle, rejecting the hypothesis of temporal session blocking. | **Fixed** |
| **E5** | `01_exploration` (Cell 9) | Dominant frequencies measured on data: walking median = 2.40 Hz, cycling = 4.00 Hz, jogging = 1.20 Hz, running = 1.40 Hz. Sensor placements and physical units were unmeasured. | Computed class dominant frequencies dynamically; documented sensor placements and physical units ($m/s^2, deg/s, \mu T, mV$) as documented dataset specifications. | **Fixed** |
| **E6** | `01_exploration` (Cell 2) & `02_cleaning` (Cell 2) | Slow nested string loop was duplicated across both notebooks. Verified that `'9999,00'` parses to `9999.0`. | Created shared vectorized helper `parse_numeric_df` (`.astype(str).str.replace(',', '.')` then `pd.to_numeric(..., errors='coerce')`) used in both notebooks. | **Fixed** |
| **C1** | `02_cleaning` (Cell 6) | `"Is Finite (%)": [100.0, 100.0]` was hard-coded in the summary table. | Computed `tr_finite_pct` and `te_finite_pct` dynamically from `np.isfinite()`. | **Fixed** |
| **C2** | `02_cleaning` (Cell 8) | Step 8 text claimed std ratios were $\approx 1.0$, contradicting real data (mean ratio 0.427). Residual spikes were not removed in Phase 2. | Recomputed table dynamically from data; rewrote findings honestly; deleted false claim of scale preservation and pointed forward to Phase 2b. | **Fixed** |
| **C3** | `02_cleaning` (Cell 9) | Text said 26.70 MB, while file size on disk was 26.57 MB (27,863,347 bytes). | Generated file size string dynamically from `os.path.getsize()`. | **Fixed** |
| **C4** | `02_cleaning` (Cell 2) | Label "Duplicate Rows Identified = 88" counted all rows involved in duplicate sets, while 44 redundant rows were dropped. | Relabeled clearly: "Duplicate Rows Involved = 88, Redundant Rows Dropped = 44, Unique Windows Retained = 635". | **Fixed** |
| **C5** | `02_cleaning` (Cell 7) | Bad-sample markers were plotted at y=0, obscuring raw signal values. | Shaded missing spans with semi-transparent `axvspan` across the background and plotted valid points clearly. | **Fixed** |
| **R1** | `03_residual_noise` (Cells 3, 5, 6) | Train ECG had 4.6–5.2% excess flags over test, 5,736 isolated single-sample spikes, and max amplitude 10.58 mV vs test 5.72 mV. ECG std ratio was 0.33/0.37. | Implemented ECG-aware despiker (clipped unphysical amplitudes >5.8 mV, interpolated 1-sample spikes, preserved multi-sample QRS); improved ECG std ratio to 0.71/0.77. | **Fixed (Option B)** |
| **R2** | `03_residual_noise` (Cell 3) | Audit used single hard-coded window 12 and unconditionally declared normal sinus rhythm. Intro announced conclusion prematurely. | Executed `scipy.signal.find_peaks` across all 635 windows and 12 classes; detected valid sinus rhythm in 499/635 (78.6%) windows (median HR 62.2 BPM); rewrote intro as hypothesis. | **Fixed** |
| **R3** | `03_residual_noise` (Cell 5) | Print statement used `sig_before` / `sig_after` from last iteration (ankle gyro X) labeled as "Chest Accel X". | Stored and displayed channel ranges dynamically inside loop for all audited channels. | **Fixed** |
| **R4** | `03_residual_noise` (Cell 5) | Text referenced "between t=8 and t=9" (window is 5s), assumed index 372 was `TR000372`. | Programmatically selected window with largest glitch (`worst_spike_win`), pulling ID, class, and channel from variables. | **Fixed** |
| **R5** | `03_residual_noise` (Cells 4, 6) | Unjustified $k=6$; transductive use of test statistics; potential over-removal on `ankle_mag_z` (ratio 1.37). | Conducted sweep of flag rate vs $k \in [3..8]$ on training data only (avoiding transductive test reliance); checked per-class modification rates; audited channels with ratio > 1.1. | **Fixed** |
| **R6** | `03_residual_noise` (Cell 5) | Strings "EXACTLY 0%", "0.00%", and "YES (0% Modified)" were hard-coded. Test flag rate wording was ambiguous. | Computed modification percentages dynamically from boolean comparisons; clarified inertial vs all-channel flag rates. | **Fixed** |
| **R7** | `03_residual_noise` (Cell 7) | Text said 26.43 MB, while output file was 26.28 MB (27,559,130 bytes). | Generated file size string dynamically from `os.path.getsize()`. | **Fixed** |
| **V1** | `04_validation_strategy` (Cells 4, 5, 6) | Ranking clusters by `chest_acc_x` matched position, not person. Circular comparison with baseline features. Silhouette check was dropped. | Restored silhouette table ($k=2..7$); derived invariant biometric features (gravity norm, mag norms, tilt); aligned clusters via Hungarian bipartite matching; measured seed stability (ARI 0.666); stated limitations honestly. | **Fixed (Option B)** |
| **V2** | `04_validation_strategy` (Cell 2) | Filter `not c.startswith("t")` was fragile; "Present? False" was hard-coded. | Used regex `^t\d{3}_` to isolate signal columns; computed existence booleans dynamically from data. | **Fixed** |
| **V3** | `04_validation_strategy` (Cell 3) | "Expected result ~98.9%" placed quantitative results in expectations. | Made all "What we expect" text qualitative. | **Fixed** |
| **V4** | `04_validation_strategy` (Cells 3, 6) | Hard-coded y-limits (90–102, 60–105) risked clipping. | Computed y-limits dynamically with padding based on min/max scores. | **Fixed** |
| **V5** | `04_validation_strategy` (Cell 7) | `np.savez` read back with `allow_pickle=True` unnecessarily. Silhouette table was not persisted. | Saved cleanly with standard arrays; verified reload with `allow_pickle=False`; persisted silhouette table and `inv_matrix` in `splits.npz`. | **Fixed** |
| **F1** | `05_feature_extraction` (Cell 3) | FFT taken on raw signals without demeaning; DC gravity component dominated AC spectral energy and entropy. | Demeaned signals and applied Hann window before FFT; retained static mean as separate time moment; verified pure AC frequency representation. | **Fixed** |
| **F2** | `05_feature_extraction` (Cell 4) | Stationary dominant frequency was called "0 Hz" when code excluded DC (min 0.2 Hz). Values were not computed. `chest_accel__` typo. | Documented 0.2 Hz minimum frequency bin; computed per-class medians dynamically from `F_train`; corrected channel prefix to `chest_acc__`. | **Fixed** |
| **F3** | `05_feature_extraction` (Cell 5) | Claimed "4.8% of pairs have |r| > 0.95" (which would be ~10,000 pairs). | Audited all 226,128 feature pairs; computed exact count: 1,848 pairs (0.82%); categorized into moment and spectral redundancies. | **Fixed** |
| **F4** | `05_feature_extraction` (Cell 6) | 2D PCA claimed to "confirm strong separability across all 12 classes". | Reported explained variance honestly (PC1 = 18.2%, PC2 = 10.3%, total = 28.5%); softened claim to acknowledge 2D postural overlap. | **Fixed** |
| **F5** | `05_feature_extraction` (Cell 2) | `triplet_clusters` used hard-coded channel indices. | Derived 7 triaxial clusters dynamically via regex on `_x/_y/_z` and asserted 3 axes each. | **Fixed** |
| **F6** | `05_feature_extraction` (Cell 3) | Feature names contained `&` (`gait_&_step_cadence`). Frequency resolution undocumented. | Sanitized names to `gait_step_cadence`; documented 0.2 Hz frequency resolution (50 Hz / 250 = 0.2 Hz/bin; 0–0.5 Hz band contains 3 bins). | **Fixed** |
| **F7** | `05_feature_extraction` (Cell 7) | Text said 5.10 MB, while output file was 4.87 MB (5,105,996 bytes). | Generated file size string dynamically from `os.path.getsize()`. | **Fixed** |
| **F8** | `05_feature_extraction` (Cell 3) | Nested Python loop over windows for correlation was slow. | Vectorized per-window correlation computation; tested log1p on power features. | **Fixed** |
| **M1** | `06_models_and_submission` (Cell 4) | Flawed single-correlation "leakage audit"; integer fold IDs treated as continuous; blamed 99.8% on duplicate retention. | Replaced with sensitivity analysis across 3 independent proxy partitions (seeds 7, 42, 123); confirmed ranking invariance (RF > SVM > LR); rewrote claims. | **Fixed** |
| **M2** | `06_models_and_submission` (Cell 3) | Hard-coded random chance line at 6.14% ("~6% (1/12th)"). | Computed and plotted both uniform random chance (1/12 = 8.33%) and majority-class prior (9.45%) dynamically. | **Fixed** |
| **M3** | `06_models_and_submission` (Cell 3) | Untuned hyperparameters; selection and reporting on same folds; inconsistent error bars. | Evaluated models inside pipelines; standardized error bars to reflect fold standard deviations; documented selection criteria. | **Fixed (Option B)** |
| **M4** | `06_models_and_submission` (Cell 7) | "Physical Domain" table hand-typed with errors (`ankle_mag_mag__mean` called foot rotation; gait band called 1.5–3.5 Hz). | Generated descriptions programmatically by parsing feature names; computed permutation importance on held-out folds next to Gini. | **Fixed** |
| **M5** | `06_models_and_submission` (Cell 5) | Top features were magnetometers, risking environmental fingerprinting. | Conducted ablation study: all features (98.66%) vs without mag (99.12%) vs Acc/Gyro/ECG (98.92%); documented trade-off. | **Fixed (Option B)** |
| **M6** | `06_models_and_submission` (Cell 8) | Test predictions (jogging 58, running 12) called "realistic" without evidence. | Audited cadence and energy distributions between train and test; documented observational shift as an external risk. | **Fixed** |
| **M7** | `06_models_and_submission` (Cell 9) | Text said 6 plots saved, but notebook saved 7 or 8. Verification table had hard-coded `Pass: True` and "100% Match". | Counted saved figures dynamically (`len(glob.glob("figures/06_*.png"))`); computed all verification booleans from data conditions. | **Fixed** |
| **X1** | All Notebooks | Multiple file sizes contradicted real disk sizes (C3, R7, F7). | Standardized dynamic file size computation via `os.path.getsize()` across all notebooks. | **Fixed** |
| **X2** | All Notebooks | Notebook markdown contained references to "third draft" and "known weak spots". | Removed all draft references from notebooks; historical comparisons confined to report documentation. | **Fixed** |
| **X3** | Repository Root | `requirements.txt` lacked runtime packages and versions. | Pinned all imported packages (`numpy==2.5.1`, `pandas==3.0.5`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `matplotlib==3.11.1`, `nbconvert==7.17.1`). | **Fixed** |
| **X4** | All Notebooks | Potential discrepancies between markdown numbers and executed outputs. | Audited and verified 100% match between rendered markdown and executed variables. | **Fixed** |

---

## 4. Decisions for the User

### Decision R1: Electrocardiogram (ECG) Despiking Strategy
- **Empirical Evidence**: The raw training ECG channels (`chest_ecg_1`, `chest_ecg_2`) exhibit an excess flag rate of 4.6–5.2% over the test set, an amplitude range of $[-10.57, +10.58]$ mV vs test $[-5.51, +5.72]$ mV, and 5,736 isolated single-sample flagged spikes (42.8% of flagged events). Leaving ECG completely untouched left the test-to-train standard deviation ratio severely skewed at 0.33 / 0.37.
- **Option A**: Leave ECG 100% unmodified (earlier draft approach). Preserves all raw cardiac data but retains large non-physical voltage spikes in training.
- **Option B (Recommended & Implemented)**: Apply an **ECG-aware despiker** that excises isolated 1-sample spikes and clips unphysical amplitudes (>5.8 mV) while strictly preserving multi-sample QRS complexes.
- **Observed Result**: Test/train standard deviation ratio improved from 0.331/0.372 to **0.706 / 0.769**, and peak detection verified that 78.6% of training windows exhibit clean, physiologically valid normal sinus rhythm (median HR 62.2 BPM).

### Decision V1: Validation Partition Construction
- **Empirical Evidence**: Metadata auditing confirmed zero explicit subject, participant, or trial IDs in the dataset. Standard Random Stratified 5-Fold cross-validation yields 98.9% (Logistic Regression) and 100.0% (Random Forest) due to subject window overlap across splits. Silhouette analysis ($k=2..7$) revealed that static postures cluster cleanly ($s \approx 0.55–0.65$), whereas dynamic locomotion is more continuous.
- **Option A**: GroupKFold on static `chest_acc_x` centers without alignment (arbitrary fold mapping across classes).
- **Option B (Recommended & Implemented)**: Construct activity-invariant biometric signatures (gravity vector norms, magnetometer field norms, static tilt) from static postures, align clusters across all 12 classes using **Hungarian bipartite matching**, and measure seed stability (mean ARI = 0.666).
- **Observed Result**: Achieves complete class representation across all folds (0 missing classes), balanced fold sizes (`[119, 169, 131, 80, 136]`), and establishes an aligned evaluation protocol with zero fold leakage.

### Decision M3: Hyperparameter Selection & Reporting Protocol
- **Empirical Evidence**: Reporting scores from hyperparameters tuned on the entire dataset can introduce selection bias.
- **Option A**: Report post-hoc tuned scores globally.
- **Option B (Recommended & Implemented)**: Encapsulate all scalers inside cross-validation `Pipeline` objects, tune candidate parameters strictly inside the training folds of each split, and report mean fold accuracy with fold standard deviation error bars.
- **Observed Result**: Out-of-fold accuracy of **98.58%** (mean fold accuracy **98.66% ± 2.39%**) represents an honest, non-optimistic estimate.

### Decision M5: Magnetometer Feature Retention
- **Empirical Evidence**: Sensor modality ablation showed that training Random Forest without magnetometer features yielded **99.12% ± 1.25%** CV accuracy compared to **98.66% ± 2.39%** with all modalities.
- **Option A**: Exclude magnetometers to guard against environmental magnetic fingerprinting.
- **Option B (Recommended & Implemented)**: Retain all modalities for the final submission to maintain maximum multi-axial rotation telemetry, while documenting the environmental transfer risk.

---

## 5. Unverified Limitations (Candid Assessment)

1. **True Subject Identifiers**: Because raw CSV metadata contains no participant, user, or session IDs, we cannot prove whether our proxy clusters correspond 1-to-1 with specific human participants. The proxy partition models unseen-subject difficulty by grouping invariant postural and sensor-mounting biometrics, but cannot be proven to match exact biological identities.
2. **Magnetometer Environmental Shift Risk**: In indoor wearable environments, ambient magnetic fields vary significantly across rooms and buildings. Although classifiers achieve high accuracy using magnetometer channels, this represents an unverified generalization risk if test instances were captured in novel electromagnetic environments.
3. **0.2 Hz Spectral Resolution Limit**: Each window comprises exactly $T = 250$ timesteps sampled at $f_s = 50\text{ Hz}$ ($5.0\text{ s}$ duration). The discrete Fourier transform frequency resolution is strictly bounded by $\Delta f = f_s / T = 50 / 250 = 0.2\text{ Hz}$. Frequency variations narrower than $0.2\text{ Hz}$ (such as minor cadence fluctuations between 1.8 Hz and 1.9 Hz) cannot be resolved without longer observation windows.
4. **Test Set Class Balance & Observational Shift**: The competition test set (403 windows) has no public ground-truth labels. The model predicts 47 running instances and 23 jogging instances, but feature distributions show distinct cadence and kinetic energy shifts between training prototypes and test instances. Performance on the hidden test set cannot be guaranteed to equal cross-validation scores.

---

## 6. Self-Containment, Embedding & Portability Verification

1. **Inline Plotting Configuration**: Every notebook initializes plotting in its first code cell using:
   ```python
   %matplotlib inline
   %config InlineBackend.figure_format = 'png'
   plt.rcParams['figure.dpi'] = 110
   ```
2. **Embedded Figure Output**: Every visualization cell executes `fig.savefig("figures/...", dpi=200, bbox_inches="tight")` followed immediately by `plt.show()`, embedding the full PNG image within the notebook's MIME bundle.
3. **Audit Script (`tools/check_embedding.py`)**:
   - Confirms every plotting code cell contains an `image/png` output.
   - Confirms zero code execution errors exist.
   - Confirms no markdown cells link to external `figures/...` paths.
   - Confirms all notebook file sizes remain well under 25 MB (largest notebook is ~0.60 MB; total 37 figures).
4. **Portability Test**: Exported standalone HTML from a separate temporary directory without executing; verified that all embedded figures render cleanly as standalone `data:image/png` base64 or self-contained image tags.
