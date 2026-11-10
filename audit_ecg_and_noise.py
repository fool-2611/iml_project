import os
import numpy as np
import pandas as pd
from scipy import signal

clean_data = np.load("outputs/clean_data.npz", allow_pickle=True)
X_train = clean_data['X_train'] # (635, 250, 23)
X_test = clean_data['X_test']   # (403, 250, 23)
y_train = clean_data['y_train']
train_ids = clean_data['train_ids']
channels = list(clean_data['channels'])

ecg1_idx = channels.index('chest_ecg_1')
ecg2_idx = channels.index('chest_ecg_2')

print(f"ECG channels: {channels[ecg1_idx]} (idx {ecg1_idx}), {channels[ecg2_idx]} (idx {ecg2_idx})")

# Let's inspect the running median filter flags on ECG
# Let's implement a running median filter and test flag rates, event widths, and amplitudes
def detect_spikes(X, window_size=7, threshold_factor=6.0):
    # X: (N, T, C)
    pad = window_size // 2
    N, T, C = X.shape
    flags = np.zeros_like(X, dtype=bool)
    deviations = np.zeros_like(X)
    
    for i in range(N):
        for c in range(C):
            sig = X[i, :, c]
            med = signal.medfilt(sig, kernel_size=window_size)
            diff = np.abs(sig - med)
            mad = np.median(diff)
            thresh = threshold_factor * (mad if mad > 1e-6 else 1.0)
            flagged = diff > thresh
            flags[i, :, c] = flagged
            deviations[i, :, c] = diff
    return flags, deviations

flags_tr, dev_tr = detect_spikes(X_train)
flags_te, dev_te = detect_spikes(X_test)

print("Flag rates in Train vs Test (k=6.0, win=7):")
for ch_idx, ch in enumerate(channels):
    tr_rate = flags_tr[:, :, ch_idx].mean() * 100
    te_rate = flags_te[:, :, ch_idx].mean() * 100
    print(f"  {ch:15s} | Train Flag Rate: {tr_rate:6.3f}% | Test Flag Rate: {te_rate:6.3f}% | Excess: {tr_rate - te_rate:6.3f}%")

# R1: In-depth analysis of ECG flagged samples
# Look at event widths in ECG: how many consecutive samples are flagged?
print("\n--- R1: Event Widths of Flagged Samples in ECG ---")
def get_flag_run_lengths(flag_matrix):
    # flag_matrix: (N, T)
    runs = []
    for row in flag_matrix:
        in_run = False
        cur_len = 0
        for val in row:
            if val:
                cur_len += 1
                in_run = True
            else:
                if in_run:
                    runs.append(cur_len)
                    cur_len = 0
                    in_run = False
        if in_run:
            runs.append(cur_len)
    return runs

runs_tr_ecg1 = get_flag_run_lengths(flags_tr[:, :, ecg1_idx])
runs_te_ecg1 = get_flag_run_lengths(flags_te[:, :, ecg1_idx])
runs_tr_inertial = get_flag_run_lengths(flags_tr[:, :, 0]) # chest_acc_x

print(f"ECG 1 Train run lengths: Total events={len(runs_tr_ecg1)}, Single-sample={runs_tr_ecg1.count(1)} ({runs_tr_ecg1.count(1)/len(runs_tr_ecg1)*100:.1f}%), Multi-sample (>1)={sum(1 for r in runs_tr_ecg1 if r > 1)} ({sum(1 for r in runs_tr_ecg1 if r > 1)/len(runs_tr_ecg1)*100:.1f}%)")
print(f"ECG 1 Test run lengths:  Total events={len(runs_te_ecg1)}, Single-sample={runs_te_ecg1.count(1)} ({runs_te_ecg1.count(1)/len(runs_te_ecg1)*100:.1f}%), Multi-sample (>1)={sum(1 for r in runs_te_ecg1 if r > 1)} ({sum(1 for r in runs_te_ecg1 if r > 1)/len(runs_te_ecg1)*100:.1f}%)")

# Amplitude distribution of ECG flagged samples in Train vs Test
flagged_amps_tr = np.abs(X_train[:, :, ecg1_idx][flags_tr[:, :, ecg1_idx]])
flagged_amps_te = np.abs(X_test[:, :, ecg1_idx][flags_te[:, :, ecg1_idx]])
print(f"\nECG 1 Flagged Amplitudes:")
print(f"  Train: Mean={flagged_amps_tr.mean():.3f}, Max={flagged_amps_tr.max():.3f}, 99th pct={np.percentile(flagged_amps_tr, 99):.3f}, Min={flagged_amps_tr.min():.3f}")
print(f"  Test:  Mean={flagged_amps_te.mean():.3f}, Max={flagged_amps_te.max():.3f}, 99th pct={np.percentile(flagged_amps_te, 99):.3f}, Min={flagged_amps_te.min():.3f}")

# What about extreme values in ECG?
all_amps_tr = X_train[:, :, ecg1_idx].ravel()
all_amps_te = X_test[:, :, ecg1_idx].ravel()
print(f"ECG 1 Global Range: Train [{all_amps_tr.min():.3f}, {all_amps_tr.max():.3f}] vs Test [{all_amps_te.min():.3f}, {all_amps_te.max():.3f}]")

# Check if there are extreme outliers in ECG 1 and ECG 2
print(f"ECG 2 Global Range: Train [{X_train[:, :, ecg2_idx].min():.3f}, {X_train[:, :, ecg2_idx].max():.3f}] vs Test [{X_test[:, :, ecg2_idx].min():.3f}, {X_test[:, :, ecg2_idx].max():.3f}]")

# Check R2: scipy.signal.find_peaks across all windows and classes
print("\n--- R2: Cardiac Peak Detection across windows ---")
heart_rates = []
for i in range(len(X_train)):
    sig = X_train[i, :, ecg1_idx]
    # At 50 Hz, R-peak spacing of 60-120 bpm corresponds to 25-50 samples (0.5 - 1.0s)
    # minimum distance = 20 samples (150 bpm max)
    peaks, props = signal.find_peaks(sig, height=np.mean(sig) + 2.0*np.std(sig), distance=20)
    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / 50.0 # seconds
        bpm = 60.0 / np.mean(rr_intervals)
        if 40 <= bpm <= 180:
            heart_rates.append(bpm)

print(f"Detected physiologically valid heart rates in {len(heart_rates)} / {len(X_train)} windows ({len(heart_rates)/len(X_train)*100:.1f}%)")
print(f"Heart rate stats: Mean={np.mean(heart_rates):.1f} BPM, Median={np.median(heart_rates):.1f} BPM, IQR=[{np.percentile(heart_rates, 25):.1f}, {np.percentile(heart_rates, 75):.1f}] BPM")

# Check R5: Per-class modification rates on inertial sensors
print("\n--- R5: Inertial Modification Rates by Class ---")
inertial_mask = [c not in [ecg1_idx, ecg2_idx] for c in range(len(channels))]
class_flag_rates = {}
for act in np.unique(y_train):
    act_mask = (y_train == act)
    flag_rate = flags_tr[act_mask][:, :, inertial_mask].mean() * 100
    class_flag_rates[act] = flag_rate
    print(f"  Class: {act:15s} | Inertial Flag Rate: {flag_rate:.3f}%")

# Check std ratio if we do or don't despike ECG, or with ECG-aware despiker
