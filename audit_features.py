import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

clean_data = np.load("outputs/clean_data_v2.npz", allow_pickle=True)
splits = np.load("outputs/splits.npz", allow_pickle=True)
X_train = clean_data['X_train'] # (635, 250, 23)
y_train = clean_data['y_train']
folds = splits['folds']
channels = list(clean_data['channels'])

# Let's extract a minimal set of FFT features with raw vs demeaned + Hann window
# Test on a representative channel: ankle_acc_z (idx 7)
ch_idx = 7

def extract_fft_raw(sig):
    # Old way: raw FFT without demeaning
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1/50.0)
    # DC bin is bin 0
    power = fft_vals**2
    energy = np.sum(power)
    # entropy
    psd_norm = power / (energy + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    dom_freq = freqs[1:][np.argmax(fft_vals[1:])]
    return [energy, entropy, dom_freq]

def extract_fft_demeaned_hann(sig):
    # New way: demean + Hann window
    sig_dm = sig - np.mean(sig)
    sig_w = sig_dm * np.hanning(len(sig))
    fft_vals = np.abs(np.fft.rfft(sig_w))
    freqs = np.fft.rfftfreq(len(sig), d=1/50.0)
    power = fft_vals**2
    energy = np.sum(power)
    psd_norm = power / (energy + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    dom_freq = freqs[1:][np.argmax(fft_vals[1:])]
    return [energy, entropy, dom_freq]

print("Comparing FFT extraction on sample signal (standing vs walking):")
stand_sig = X_train[y_train == 'standing'][0, :, ch_idx]
walk_sig = X_train[y_train == 'walking'][0, :, ch_idx]

print("Standing:")
print("  Raw FFT:      energy =", extract_fft_raw(stand_sig)[0], "entropy =", extract_fft_raw(stand_sig)[1])
print("  Demeaned FFT: energy =", extract_fft_demeaned_hann(stand_sig)[0], "entropy =", extract_fft_demeaned_hann(stand_sig)[1])
print("Walking:")
print("  Raw FFT:      energy =", extract_fft_raw(walk_sig)[0], "entropy =", extract_fft_raw(walk_sig)[1])
print("  Demeaned FFT: energy =", extract_fft_demeaned_hann(walk_sig)[0], "entropy =", extract_fft_demeaned_hann(walk_sig)[1])

# F5: Deriving triplet clusters via regex
import re
triplets = {}
for idx, ch in enumerate(channels):
    m = re.match(r'^(.*)_[xyz]$', ch)
    if m:
        base = m.group(1)
        if base not in triplets:
            triplets[base] = []
        triplets[base].append(idx)
print("\nTriplets derived via regex:")
for k, v in triplets.items():
    print(f"  {k:15s}: indices {v} -> {[channels[i] for i in v]}")
    assert len(v) == 3, f"Cluster {k} does not have 3 axes: {v}"

# F8: Fast vectorised correlation matrix per window
print("\nTesting vectorised per-window correlation:")
# We have 7 triplet clusters: each has 3 axes (x, y, z)
# Compute Pearson correlation r_xy, r_xz, r_yz for all windows at once
def compute_triplet_corrs_vectorized(X, triplet_indices):
    # X: (N, 250, C)
    # triplet_indices: list of 3 channel indices [ix, iy, iz]
    # Center each channel per window
    sub = X[:, :, triplet_indices] # (N, 250, 3)
    sub_centered = sub - sub.mean(axis=1, keepdims=True) # (N, 250, 3)
    # Covariance numerator: sum over time of (x - mx)*(y - my)
    # std: sqrt(sum(x - mx)^2)
    stds = np.sqrt(np.sum(sub_centered**2, axis=1)) # (N, 3)
    stds[stds < 1e-12] = 1e-12
    
    # r_01, r_02, r_12
    r01 = np.sum(sub_centered[:, :, 0] * sub_centered[:, :, 1], axis=1) / (stds[:, 0] * stds[:, 1])
    r02 = np.sum(sub_centered[:, :, 0] * sub_centered[:, :, 2], axis=1) / (stds[:, 0] * stds[:, 2])
    r12 = np.sum(sub_centered[:, :, 1] * sub_centered[:, :, 2], axis=1) / (stds[:, 1] * stds[:, 2])
    return np.column_stack([r01, r02, r12]) # (N, 3)

c_fast = compute_triplet_corrs_vectorized(X_train, triplets['chest_acc'])
print(f"Vectorized correlations shape: {c_fast.shape}, values range: [{c_fast.min():.3f}, {c_fast.max():.3f}]")
