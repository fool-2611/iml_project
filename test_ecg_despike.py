import numpy as np
from scipy import signal

clean_data = np.load("outputs/clean_data.npz", allow_pickle=True)
X_train = clean_data['X_train'].copy()
X_test = clean_data['X_test']
channels = list(clean_data['channels'])
ecg1_idx = channels.index('chest_ecg_1')
ecg2_idx = channels.index('chest_ecg_2')

# Before: std ratios
s_tr1 = np.std(X_train[:, :, ecg1_idx])
s_te1 = np.std(X_test[:, :, ecg1_idx])
s_tr2 = np.std(X_train[:, :, ecg2_idx])
s_te2 = np.std(X_test[:, :, ecg2_idx])
print(f"Before ECG despike: ECG1 std ratio = {s_te1/s_tr1:.3f}, ECG2 std ratio = {s_te2/s_tr2:.3f}")

# Design ECG-aware despiker:
# 1. Physical saturation clipping: In test set, max ECG magnitude is < 6.0 mV.
# In train, glitches exceed 10.0 mV.
# 2. Single-sample impulse filter: If |sig[t] - med[t]| > k * MAD AND the event width is 1 (or |sig[t]| > 6.0),
# it is an isolated spike, NOT an authentic multi-sample QRS complex.
# Replace isolated spikes with median or linear interpolation from neighbors!

def despike_ecg_aware(sig, max_phys_amp=6.0, k=5.0):
    # sig: 1D array of 250 samples
    clean = sig.copy()
    med = signal.medfilt(clean, kernel_size=5)
    diff = np.abs(clean - med)
    mad = np.median(diff)
    thresh = k * (mad if mad > 1e-4 else 1.0)
    
    # Flag points
    flagged = (diff > thresh) | (np.abs(clean) > max_phys_amp)
    
    # Check width of flagged segments
    # An R-peak spans 2-4 samples. An injected glitch is 1 sample (or amplitude > max_phys_amp)
    n = len(sig)
    for i in range(n):
        if flagged[i]:
            # check if isolated or extreme amplitude
            is_extreme = np.abs(clean[i]) > max_phys_amp
            is_isolated = True
            if i > 0 and flagged[i-1]:
                is_isolated = False
            if i < n - 1 and flagged[i+1]:
                is_isolated = False
            
            if is_extreme or is_isolated:
                # Interpolate from neighbors
                left = clean[i-1] if i > 0 else med[i]
                right = clean[i+1] if i < n - 1 else med[i]
                clean[i] = 0.5 * (left + right)
    return clean

# Apply to X_train ECG
X_tr_ecg_despiked = X_train.copy()
for i in range(len(X_train)):
    X_tr_ecg_despiked[i, :, ecg1_idx] = despike_ecg_aware(X_train[i, :, ecg1_idx])
    X_tr_ecg_despiked[i, :, ecg2_idx] = despike_ecg_aware(X_train[i, :, ecg2_idx])

s_tr1_post = np.std(X_tr_ecg_despiked[:, :, ecg1_idx])
s_tr2_post = np.std(X_tr_ecg_despiked[:, :, ecg2_idx])
print(f"After ECG-aware despike: ECG1 std ratio = {s_te1/s_tr1_post:.3f}, ECG2 std ratio = {s_te2/s_tr2_post:.3f}")
print(f"Train ECG1 max: {np.abs(X_tr_ecg_despiked[:, :, ecg1_idx]).max():.3f} vs Test ECG1 max: {np.abs(X_test[:, :, ecg1_idx]).max():.3f}")
