import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

clean_data = np.load("outputs/clean_data.npz", allow_pickle=True)
X_train = clean_data['X_train'] # (635, 250, 23)
y_train = clean_data['y_train']
channels = list(clean_data['channels'])
classes = np.unique(y_train)

# V1 (a): Silhouette analysis for k=2..7 per class
print("--- V1 (a): Silhouette Analysis for k=2..7 per class ---")
# Use the chest_acc_x or invariant features
chest_ax_idx = channels.index('chest_acc_x')
silhouette_results = {}

for act in classes:
    act_mask = (y_train == act)
    n_samples = act_mask.sum()
    # Feature for this class: window mean of chest_acc_x
    X_act = X_train[act_mask, :, chest_ax_idx].mean(axis=1, keepdims=True)
    # Standardize within class
    if np.std(X_act) > 1e-6:
        X_act_std = (X_act - np.mean(X_act)) / np.std(X_act)
    else:
        X_act_std = X_act
        
    act_scores = {}
    for k in range(2, 8):
        if k >= n_samples:
            act_scores[k] = np.nan
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_act_std)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_act_std, labels)
        else:
            score = np.nan
        act_scores[k] = score
    silhouette_results[act] = act_scores

sil_df = pd.DataFrame(silhouette_results).T
print(sil_df.round(3))

# V1 (b): Activity-invariant features for clustering
print("\n--- V1 (b): Constructing Activity-Invariant Signatures ---")
# Triaxial norms across time
# 1. Chest acc norm mean: norm = sqrt(ax^2 + ay^2 + az^2)
c_ax = channels.index('chest_acc_x')
c_ay = channels.index('chest_acc_y')
c_az = channels.index('chest_acc_z')
chest_norm = np.linalg.norm(X_train[:, :, [c_ax, c_ay, c_az]], axis=2).mean(axis=1) # (635,)

# 2. Ankle mag norm mean
a_mx = channels.index('ankle_mag_x')
a_my = channels.index('ankle_mag_y')
a_mz = channels.index('ankle_mag_z')
ankle_mag_norm = np.linalg.norm(X_train[:, :, [a_mx, a_my, a_mz]], axis=2).mean(axis=1)

# 3. Arm mag norm mean
ar_mx = channels.index('arm_mag_x')
ar_my = channels.index('arm_mag_y')
ar_mz = channels.index('arm_mag_z')
arm_mag_norm = np.linalg.norm(X_train[:, :, [ar_mx, ar_my, ar_mz]], axis=2).mean(axis=1)

# 4. Static gravity tilt (chest_acc_x mean)
chest_ax_mean = X_train[:, :, c_ax].mean(axis=1)

# Combined invariant signature
inv_features = np.column_stack([chest_ax_mean, chest_norm, ankle_mag_norm, arm_mag_norm])
print(f"Invariant feature matrix shape: {inv_features.shape}")

# Test static classes (lying, sitting, standing)
static_mask = np.isin(y_train, ['lying', 'sitting', 'standing'])
print(f"Static windows count: {static_mask.sum()}")

# Test Hungarian alignment across classes
def cluster_and_align_hungarian(n_clusters=5, seed=42):
    folds = np.zeros(len(y_train), dtype=int)
    
    # 1. Cluster on static classes to establish reference centroids
    ref_mask = (y_train == 'sitting')
    km_ref = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    ref_labels = km_ref.fit_predict(inv_features[ref_mask])
    ref_centers = km_ref.cluster_centers_
    folds[ref_mask] = ref_labels
    
    # 2. For each other class, cluster and match with Hungarian algorithm
    for act in classes:
        if act == 'sitting':
            continue
        act_mask = (y_train == act)
        n_act = act_mask.sum()
        k_act = min(n_clusters, n_act)
        
        km_act = KMeans(n_clusters=k_act, random_state=seed, n_init=10)
        c_labels = km_act.fit_predict(inv_features[act_mask])
        act_centers = km_act.cluster_centers_
        
        # Cost matrix: Euclidean distance between class centroids and reference centroids
        # cost[i, j] = dist(act_centers[i], ref_centers[j])
        cost = np.zeros((k_act, n_clusters))
        for i in range(k_act):
            for j in range(n_clusters):
                cost[i, j] = np.linalg.norm(act_centers[i] - ref_centers[j])
                
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {row_ind[m]: col_ind[m] for m in range(len(row_ind))}
        
        aligned_labels = np.array([mapping[l] for l in c_labels])
        folds[act_mask] = aligned_labels
    return folds

folds_hungarian = cluster_and_align_hungarian(5, seed=42)
print("Hungarian fold counts:", [np.sum(folds_hungarian == i) for i in range(5)])

# Check cross-class class representation
ct = pd.crosstab(pd.Series(folds_hungarian, name="Fold"), pd.Series(y_train, name="Class"))
print("\nHungarian Fold Class Coverage (zeros count):", (ct == 0).sum().sum())
print(ct)

# Test seed stability (Adjusted Rand Index across seeds)
ari_scores = []
for s in [0, 7, 42, 123, 999]:
    f_s = cluster_and_align_hungarian(5, seed=s)
    ari = adjusted_rand_score(folds_hungarian, f_s)
    ari_scores.append(ari)
print(f"\nSeed Stability (ARI vs seed 42): Mean = {np.mean(ari_scores):.3f}, Min = {np.min(ari_scores):.3f}")
