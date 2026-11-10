import os
import re
import numpy as np
import pandas as pd

# Load train and test data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("--- E1: Sentinel and Corruption Counts on Deduplicated Train ---")
# Drop duplicates
feature_cols = [c for c in train_df.columns if c not in ['id', 'target']]
train_dedup = train_df.drop_duplicates(subset=feature_cols).copy()
print(f"Raw train shape: {train_df.shape}, Deduplicated train shape: {train_dedup.shape}")

# Vectorized helper for E6
def parse_numeric_df(df, cols):
    # Vectorized string replace and to_numeric
    # First handle string columns or mixed types
    res = df[cols].copy()
    for c in cols:
        if res[c].dtype == object:
            res[c] = pd.to_numeric(res[c].astype(str).str.replace(",", ".", regex=False), errors='coerce')
        else:
            res[c] = pd.to_numeric(res[c], errors='coerce')
    return res

# Count in raw train vs deduplicated train
# Specifically count string tokens: '-', '?', 'unknown'
# Comma strings: contains ','
# Sentinel: 9999 or 9999.0 or '9999,00' or '9999.0'
raw_vals = train_df[feature_cols].values
dedup_vals = train_dedup[feature_cols].values

def audit_corruptions(arr):
    native_nans = 0
    tokens = 0
    commas = 0
    sentinels = 0
    for v in arr.ravel():
        if pd.isna(v):
            native_nans += 1
        elif isinstance(v, str):
            v_str = v.strip()
            if v_str in ['-', '?', 'unknown']:
                tokens += 1
            elif ',' in v_str:
                commas += 1
                v_clean = v_str.replace(',', '.')
                try:
                    vf = float(v_clean)
                    if vf == 9999.0:
                        sentinels += 1
                except:
                    pass
            else:
                try:
                    vf = float(v_str)
                    if vf == 9999.0:
                        sentinels += 1
                except:
                    pass
        elif isinstance(v, (int, float, np.number)):
            if v == 9999.0:
                sentinels += 1
    return native_nans, tokens, commas, sentinels

nan_raw, tok_raw, com_raw, sen_raw = audit_corruptions(raw_vals)
nan_dedup, tok_dedup, com_dedup, sen_dedup = audit_corruptions(dedup_vals)
print(f"Raw train:       NaN={nan_raw}, Tokens={tok_raw}, CommaDecimals={com_raw}, Sentinels={sen_raw}")
print(f"Deduplicated:    NaN={nan_dedup}, Tokens={tok_dedup}, CommaDecimals={com_dedup}, Sentinels={sen_dedup}")

print("\n--- E2: Train vs Test Std Ratios in 01_exploration ---")
# Parse test numeric
test_cols = [c for c in test_df.columns if c != 'id']
# Parse raw train and test
train_num = parse_numeric_df(train_df, feature_cols)
test_num = parse_numeric_df(test_df, test_cols)

# Group by channel
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

# Calculate std across all timesteps per channel for train (with sentinels replaced by NaN) and test
# In step 7 of exploration:
# Replace 9999.0 with NaN first
train_clean_vals = train_num.copy()
train_clean_vals[train_clean_vals == 9999.0] = np.nan

ratios = []
print(f"{'Channel':15s} | {'Train Mean':10s} | {'Test Mean':10s} | {'Train Std':10s} | {'Test Std':10s} | {'Test/Train Std':14s}")
print("-" * 80)
for ch in channel_names:
    ch_cols_tr = [f"{ch}_step_{s}" for s in range(250)]
    ch_cols_te = [f"{ch}_step_{s}" for s in range(250)]
    tr_vals = train_clean_vals[ch_cols_tr].values.ravel()
    te_vals = test_num[ch_cols_te].values.ravel()
    
    tr_m, tr_s = np.nanmean(tr_vals), np.nanstd(tr_vals)
    te_m, te_s = np.nanmean(te_vals), np.nanstd(te_vals)
    ratio = te_s / tr_s if tr_s > 0 else np.nan
    ratios.append(ratio)
    print(f"{ch:15s} | {tr_m:10.4f} | {te_m:10.4f} | {tr_s:10.4f} | {te_s:10.4f} | {ratio:14.4f}")
print(f"Overall Mean Test/Train Std Ratio: {np.nanmean(ratios):.4f}, Min: {np.nanmin(ratios):.4f}, Max: {np.nanmax(ratios):.4f}")

print("\n--- E3: Majority Baseline ---")
raw_vc = train_df['target'].value_counts()
dedup_vc = train_dedup['target'].value_counts()
print(f"Raw train majority class: {raw_vc.index[0]} with {raw_vc.iloc[0]} / {len(train_df)} = {raw_vc.iloc[0]/len(train_df)*100:.2f}%")
print(f"Deduplicated majority class: {dedup_vc.index[0]} with {dedup_vc.iloc[0]} / {len(train_dedup)} = {dedup_vc.iloc[0]/len(train_dedup)*100:.2f}%")

print("\n--- E4: Permutation Test on Row Order & Label Sequence ---")
labels = train_df['target'].values
n_changes = np.sum(labels[:-1] != labels[1:])
change_rate = n_changes / (len(labels) - 1)
# Compute runs
runs = []
cur_run = 1
for i in range(1, len(labels)):
    if labels[i] == labels[i-1]:
        cur_run += 1
    else:
        runs.append(cur_run)
        cur_run = 1
runs.append(cur_run)
mean_run = np.mean(runs)
print(f"Observed label transitions: {n_changes} / {len(labels)-1} (transition rate: {change_rate*100:.2f}%)")
print(f"Observed mean run length: {mean_run:.2f}")

# Simulate under random shuffle
np.random.seed(42)
perm_changes = []
perm_runs = []
for _ in range(1000):
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

print(f"Under random permutation: Expected transitions = {np.mean(perm_changes):.2f} +/- {np.std(perm_changes):.2f} (rate: {np.mean(perm_changes)/(len(labels)-1)*100:.2f}%)")
print(f"Under random permutation: Expected mean run length = {np.mean(perm_runs):.2f} +/- {np.std(perm_runs):.2f}")
p_val = np.mean(np.array(perm_changes) <= n_changes)
print(f"Empirical p-value of observed transitions <= permuted: {p_val:.5f}")

# Check ID order
ids = train_df['id'].values
id_nums = [int(i.replace('TR', '')) for i in ids]
is_sorted = np.all(np.diff(id_nums) > 0)
print(f"Are train IDs sorted consecutively? {is_sorted} (min: {id_nums[0]}, max: {id_nums[-1]}, len: {len(id_nums)})")
