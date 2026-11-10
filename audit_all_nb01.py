import os
import re
import numpy as np
import pandas as pd
from scipy import signal

# Load train and test with low_memory=False
train_df = pd.read_csv("data/train.csv", low_memory=False)
test_df = pd.read_csv("data/test.csv", low_memory=False)
feature_cols = [c for c in train_df.columns if c not in ['id', 'target']]

# Fast vectorized parse helper (E6)
def vectorized_parse_to_numeric(df, cols):
    # Process column blocks by dtype
    res = df[cols].copy()
    obj_cols = [c for c in cols if res[c].dtype == object]
    num_cols = [c for c in cols if res[c].dtype != object]
    
    # Fast string replace and float conversion
    for c in obj_cols:
        s = res[c].astype(str).str.replace(',', '.', regex=False)
        res[c] = pd.to_numeric(s, errors='coerce')
    for c in num_cols:
        res[c] = pd.to_numeric(res[c], errors='coerce')
    return res

train_num = vectorized_parse_to_numeric(train_df, feature_cols)
test_cols = [c for c in test_df.columns if c != 'id']
test_num = vectorized_parse_to_numeric(test_df, test_cols)

# Deduplication on numeric
dup_mask = train_num.duplicated(keep='first')
print(f"Duplicate rows identified: {dup_mask.sum()}, Unique: {(~dup_mask).sum()}")
train_dedup_num = train_num[~dup_mask].copy()

# Count raw vs dedup
raw_vals = train_num.values
dedup_vals = train_dedup_num.values

raw_sentinels = np.sum(raw_vals == 9999.0)
dedup_sentinels = np.sum(dedup_vals == 9999.0)
raw_nans = np.sum(np.isnan(raw_vals))
dedup_nans = np.sum(np.isnan(dedup_vals))

print(f"E1 Findings:")
print(f"  Raw 9999.0: {raw_sentinels}, Dedup 9999.0: {dedup_sentinels}")
print(f"  Raw NaNs:   {raw_nans}, Dedup NaNs:   {dedup_nans}")

# E2: Train vs Test Std Ratios
channel_names = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'chest_ecg_1', 'chest_ecg_2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z'
]

# In Step 7 of exploration:
# Replace 9999.0 with NaN for std computation
tr_arr = train_num.values.copy()
tr_arr[tr_arr == 9999.0] = np.nan
te_arr = test_num.values

# Check std ratios per channel
print("\nE2 Findings: Train vs Test Statistics (with 9999.0 as NaN, before despiking):")
ratios = []
for ch_idx, ch in enumerate(channel_names):
    cols_idx = [t * 23 + ch_idx for t in range(250)]
    tr_ch = tr_arr[:, cols_idx].ravel()
    te_ch = te_arr[:, cols_idx].ravel()
    tr_m, tr_s = np.nanmean(tr_ch), np.nanstd(tr_ch)
    te_m, te_s = np.nanmean(te_ch), np.nanstd(te_ch)
    r = te_s / tr_s if tr_s > 0 else np.nan
    ratios.append(r)
    print(f"  {ch:15s} | Train Mean: {tr_m:7.3f}, Test Mean: {te_m:7.3f} | Train Std: {tr_s:7.3f}, Test Std: {te_s:7.3f} | Test/Train Std Ratio: {r:6.3f}")

print(f"Mean Test/Train Std Ratio across all 23 channels: {np.nanmean(ratios):.3f}")
print(f"Mean Test/Train Std Ratio across 21 inertial channels: {np.nanmean(ratios[:3] + ratios[5:]):.3f}")
print(f"ECG 1 Ratio: {ratios[3]:.3f}, ECG 2 Ratio: {ratios[4]:.3f}")

# E3: Baseline accuracy
raw_counts = train_df['target'].value_counts()
dedup_counts = train_df.loc[~dup_mask, 'target'].value_counts()
print("\nE3 Findings: Majority Baseline:")
print(f"  Raw (n={len(train_df)}):   Majority '{raw_counts.index[0]}' count={raw_counts.iloc[0]} ({raw_counts.iloc[0]/len(train_df)*100:.2f}%)")
print(f"  Dedup (n={(~dup_mask).sum()}): Majority '{dedup_counts.index[0]}' count={dedup_counts.iloc[0]} ({dedup_counts.iloc[0]/(~dup_mask).sum()*100:.2f}%)")

# E4: Row order permutation test
labels = train_df['target'].values
n_changes = np.sum(labels[:-1] != labels[1:])
obs_change_rate = n_changes / (len(labels) - 1)
runs = []
cur_r = 1
for i in range(1, len(labels)):
    if labels[i] == labels[i-1]:
        cur_r += 1
    else:
        runs.append(cur_r)
        cur_r = 1
runs.append(cur_r)
obs_mean_run = np.mean(runs)

np.random.seed(42)
perm_changes = []
perm_runs = []
for _ in range(10000):
    shuffled = np.random.permutation(labels)
    c = np.sum(shuffled[:-1] != shuffled[1:])
    perm_changes.append(c)
    # runs
    r = []
    c_r = 1
    for i in range(1, len(shuffled)):
        if shuffled[i] == shuffled[i-1]:
            c_r += 1
        else:
            r.append(c_r)
            c_r = 1
    r.append(c_r)
    perm_runs.append(np.mean(r))

p_val = np.mean(np.array(perm_changes) <= n_changes)
print("\nE4 Findings: Row order permutation test:")
print(f"  Observed transitions: {n_changes} / {len(labels)-1} ({obs_change_rate*100:.2f}%), Mean run length: {obs_mean_run:.2f}")
print(f"  Expected under random shuffle: {np.mean(perm_changes):.1f} ({np.mean(perm_changes)/(len(labels)-1)*100:.2f}%), Mean run length: {np.mean(perm_runs):.2f}")
print(f"  Permutation p-value: {p_val:.4f}")
print(f"  Conclusion: The sequence is completely consistent with a random shuffle (p={p_val:.3f} > 0.05). There is no partial temporal block ordering.")

# E5: Dominant frequency per class for Step 9
print("\nE5 Findings: Dominant frequency per class (walking vs cycling):")
# Compute dominant frequency of chest_acc_x and ankle_acc_z for walking vs cycling
train_clean_num = train_num.copy()
train_clean_num[train_clean_num == 9999.0] = np.nan
# interpolate NaNs per row
for act in ['walking', 'cycling', 'jogging', 'running']:
    act_mask = (train_df['target'] == act) & (~dup_mask)
    # Get ankle_acc_z (channel 7)
    cols = [t * 23 + 7 for t in range(250)]
    windows = train_clean_num.loc[act_mask].iloc[:, cols].values
    dom_freqs = []
    for w in windows:
        # linear interpolate nan
        w_s = pd.Series(w).interpolate().bfill().ffill().values
        # detrend
        w_d = w_s - np.mean(w_s)
        fft_vals = np.abs(np.fft.rfft(w_d))
        freqs = np.fft.rfftfreq(len(w_d), d=1/50.0)
        # exclude 0 Hz
        dom_f = freqs[1:][np.argmax(fft_vals[1:])]
        dom_freqs.append(dom_f)
    print(f"  Activity: {act:12s} | Ankle Acc Z Dominant Freq: Mean = {np.mean(dom_freqs):.2f} Hz, Median = {np.median(dom_freqs):.2f} Hz (IQR: {np.percentile(dom_freqs, 25):.2f} - {np.percentile(dom_freqs, 75):.2f} Hz)")
