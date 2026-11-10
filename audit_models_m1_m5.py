import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

feats = np.load('outputs/features.npz', allow_pickle=True)
splits = np.load('outputs/splits.npz', allow_pickle=True)
F_train, F_test, y_train = feats['F_train'], feats['F_test'], feats['y_train']
names = list(feats['names'])
folds = splits['folds']
classes = np.unique(y_train)

# M5: Ablation study on feature modalities
# 1. All features (673)
# 2. Without magnetometer features
# 3. Acc + Gyro + ECG only
mag_indices = [i for i, n in enumerate(names) if '_mag' in n]
non_mag_indices = [i for i, n in enumerate(names) if '_mag' not in n]
acc_gyro_ecg_indices = [i for i, n in enumerate(names) if any(k in n for k in ['_acc', '_gyro', '_ecg'])]

print(f"Total features: {len(names)}")
print(f"Non-magnetometer features: {len(non_mag_indices)} (removed {len(mag_indices)} mag features)")
print(f"Acc/Gyro/ECG features:     {len(acc_gyro_ecg_indices)}")

# Benchmark RF across ablations
ablations = {
    'All Features (673)': list(range(len(names))),
    'Without Magnetometer': non_mag_indices,
    'Acc + Gyro + ECG Only': acc_gyro_ecg_indices
}

for abl_name, feat_idx in ablations.items():
    X_abl = F_train[:, feat_idx]
    accs = []
    oof_pred = np.empty_like(y_train)
    for f in range(5):
        tr_m = (folds != f)
        va_m = (folds == f)
        rf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
        rf.fit(X_abl[tr_m], y_train[tr_m])
        p = rf.predict(X_abl[va_m])
        accs.append(accuracy_score(y_train[va_m], p))
        oof_pred[va_m] = p
    oof_acc = accuracy_score(y_train, oof_pred)
    print(f"Ablation [{abl_name:25s}] -> Mean CV: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%, OOF: {oof_acc*100:.2f}%")

# M1: Sensitivity analysis across multiple proxy partitions (from V1)
print("\n--- M1: Sensitivity Analysis across alternative proxy partitions ---")
# Test 3 different random seeds for Hungarian alignment
from audit_validation_v1 import cluster_and_align_hungarian

for s in [7, 42, 123]:
    alt_folds = cluster_and_align_hungarian(5, seed=s)
    accs_rf = []
    accs_lr = []
    accs_svm = []
    for f in range(5):
        tr_m = (alt_folds != f)
        va_m = (alt_folds == f)
        # RF
        rf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
        rf.fit(F_train[tr_m], y_train[tr_m])
        accs_rf.append(accuracy_score(y_train[va_m], rf.predict(F_train[va_m])))
        # LR
        lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, max_iter=2000, random_state=42))])
        lr.fit(F_train[tr_m], y_train[tr_m])
        accs_lr.append(accuracy_score(y_train[va_m], lr.predict(F_train[va_m])))
        # SVM
        svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=5.0, kernel='rbf', random_state=42))])
        svm.fit(F_train[tr_m], y_train[tr_m])
        accs_svm.append(accuracy_score(y_train[va_m], svm.predict(F_train[va_m])))
    print(f"Seed {s:3d} Partition | RF: {np.mean(accs_rf)*100:.2f}% | SVM: {np.mean(accs_svm)*100:.2f}% | LR: {np.mean(accs_lr)*100:.2f}% | Ranking Preserved: {np.mean(accs_rf) > np.mean(accs_svm) > np.mean(accs_lr)}")

# M6: Inspect running vs jogging feature distributions in train vs predicted test
rf_full = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
rf_full.fit(F_train, y_train)
y_test_pred = rf_full.predict(F_test)

# Find cadence and energy feature indices
cadence_idx = [i for i, n in enumerate(names) if 'ankle_acc_z__band' in n or 'cadence' in n][0]
energy_idx = [i for i, n in enumerate(names) if 'ankle_acc_z__energy' in n or 'ankle_acc_z__spec_energy' in n or 'ankle_acc_z__time_mad' in n][0]
print(f"\nFeature for cadence: {names[cadence_idx]}")
print(f"Feature for energy:  {names[energy_idx]}")

for act in ['jogging', 'running']:
    tr_vals_c = F_train[y_train == act, cadence_idx]
    te_vals_c = F_test[y_test_pred == act, cadence_idx]
    tr_vals_e = F_train[y_train == act, energy_idx]
    te_vals_e = F_test[y_test_pred == act, energy_idx]
    print(f"\nActivity: {act}")
    print(f"  Cadence: Train Mean={np.mean(tr_vals_c):.3f} vs Test Mean={np.mean(te_vals_c):.3f}")
    print(f"  Energy:  Train Mean={np.mean(tr_vals_e):.3f} vs Test Mean={np.mean(te_vals_e):.3f}")
