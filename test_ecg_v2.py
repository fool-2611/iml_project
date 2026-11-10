import numpy as np
from scipy import signal

clean_data = np.load("outputs/clean_data.npz", allow_pickle=True)
X_train = clean_data['X_train'].copy()
X_test = clean_data['X_test']
channels = list(clean_data['channels'])
ecg1_idx = channels.index('chest_ecg_1')
ecg2_idx = channels.index('chest_ecg_2')

s_te1 = np.std(X_test[:, :, ecg1_idx])
s_te2 = np.std(X_test[:, :, ecg2_idx])

def despike_ecg_aware_v2(sig, max_phys_amp=5.8, kernel_size=7, threshold_mad=5.0):
    clean = sig.copy()
    # 1. Running median
    med = signal.medfilt(clean, kernel_size=kernel_size)
    diff = np.abs(clean - med)
    mad = np.median(diff)
    thresh = threshold_mad * (mad if mad > 1e-4 else 1.0)
    
    # 2. Flag anomalies: extreme amplitude or large impulse difference
    flagged = (np.abs(clean) > max_phys_amp) | (diff > thresh)
    
    # 3. An authentic R-peak at 50 Hz lasts 2-4 samples.
    # Single isolated spikes, or points exceeding physical max, are glitches.
    # Replace them with median filter value or linear interpolation
    n = len(clean)
    for i in range(n):
        if np.abs(clean[i]) > max_phys_amp:
            clean[i] = med[i]
        elif flagged[i]:
            # check if width of flag event is 1
            is_single = True
            if (i > 0 and flagged[i-1]) or (i < n - 1 and flagged[i+1]):
                is_single = False
            if is_single:
                clean[i] = med[i]
    return clean

X_tr_ecg_despiked = X_train.copy()
for i in range(len(X_train)):
    X_tr_ecg_despiked[i, :, ecg1_idx] = despike_ecg_aware_v2(X_train[i, :, ecg1_idx])
    X_tr_ecg_despiked[i, :, ecg2_idx] = despike_ecg_aware_v2(X_train[i, :, ecg2_idx])

s_tr1_post = np.std(X_tr_ecg_despiked[:, :, ecg1_idx])
s_tr2_post = np.std(X_tr_ecg_despiked[:, :, ecg2_idx])
print(f"After ECG-aware despike v2: ECG1 std ratio = {s_te1/s_tr1_post:.3f}, ECG2 std ratio = {s_te2/s_tr2_post:.3f}")
print(f"Train ECG1 max: {np.abs(X_tr_ecg_despiked[:, :, ecg1_idx]).max():.3f} vs Test ECG1 max: {np.abs(X_test[:, :, ecg1_idx]).max():.3f}")

# Check R-peak detection retention
beats_orig = 0
beats_despiked = 0
for i in range(len(X_train)):
    sig_o = X_train[i, :, ecg1_idx]
    sig_d = X_tr_ecg_despiked[i, :, ecg1_idx]
    p_o, _ = signal.find_peaks(sig_o, height=np.mean(sig_o) + 2.0*np.std(sig_o), distance=20)
    p_d, _ = signal.find_peaks(sig_d, height=np.mean(sig_d) + 2.0*np.std(sig_d), distance=20)
    beats_orig += len(p_o)
    beats_despiked += len(p_d)

print(f"Total detected cardiac peaks in Train: Raw = {beats_orig}, Post-Despiked = {beats_despiked} (Retention = {beats_despiked/beats_orig*100:.1f}%)")
