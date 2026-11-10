import os
import numpy as np
import pandas as pd

clean_data = np.load("outputs/clean_data.npz", allow_pickle=True)
X_train = clean_data['X_train'] # (635, 250, 23)
X_test = clean_data['X_test']   # (403, 250, 23)
y_train = clean_data['y_train']
channels = clean_data['channels']

print("--- C1: Compute Is Finite ---")
tr_finite_pct = (np.isfinite(X_train).sum() / X_train.size) * 100
te_finite_pct = (np.isfinite(X_test).sum() / X_test.size) * 100
print(f"Computed Is Finite (%): Train = {tr_finite_pct:.4f}%, Test = {te_finite_pct:.4f}%")

print("\n--- C2: Recompute Step 8 Train vs Test Statistics in Clean Data ---")
ratios = []
for ch_idx, ch in enumerate(channels):
    tr_vals = X_train[:, :, ch_idx].ravel()
    te_vals = X_test[:, :, ch_idx].ravel()
    tr_m, tr_s = np.mean(tr_vals), np.std(tr_vals)
    te_m, te_s = np.mean(te_vals), np.std(te_vals)
    r = te_s / tr_s if tr_s > 0 else np.nan
    ratios.append(r)
    print(f"  {ch:15s} | Train Mean: {tr_m:7.3f}, Test Mean: {te_m:7.3f} | Train Std: {tr_s:7.3f}, Test Std: {te_s:7.3f} | Ratio: {r:6.3f}")

print(f"Mean Test/Train Std Ratio: {np.mean(ratios):.3f}")
print("Notice that at this stage (Phase 2), large residual spikes still exist in X_train, so train std is ~2.5x test std (ratio ~0.44). It is NOT ~1.0 yet!")

print("\n--- C3: Clean Data File Size ---")
file_sz_bytes = os.path.getsize("outputs/clean_data.npz")
file_sz_mb = file_sz_bytes / (1024 * 1024)
print(f"File size of clean_data.npz: {file_sz_mb:.2f} MB ({file_sz_bytes:,} bytes)")
