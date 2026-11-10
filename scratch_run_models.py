import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

feats = np.load('outputs/features.npz', allow_pickle=True)
splits = np.load('outputs/splits.npz', allow_pickle=True)
clean = np.load('outputs/clean_data_v2.npz', allow_pickle=True)

F_train, F_test, y_train = feats['F_train'], feats['F_test'], feats['y_train']
names = feats['names']
folds = splits['folds']
test_ids = clean['test_ids']
classes = np.unique(y_train)

models = {
    'Majority Baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression (L2)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=2000, random_state=42))
    ]),
    'k-Nearest Neighbors (k=5)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Support Vector Machine (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=5.0, kernel='rbf', random_state=42))
    ]),
    'Random Forest (300 trees)': RandomForestClassifier(
        n_estimators=300, max_depth=16, random_state=42, n_jobs=-1
    )
}

results = {}
oof_predictions = {}

for name, model in models.items():
    fold_accs = []
    oof_pred = np.empty_like(y_train)
    
    for fold_idx in range(5):
        val_mask = (folds == fold_idx)
        train_mask = ~val_mask
        
        X_tr, y_tr = F_train[train_mask], y_train[train_mask]
        X_va, y_va = F_train[val_mask], y_train[val_mask]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        fold_accs.append(acc)
        oof_pred[val_mask] = preds
        
    oof_acc = accuracy_score(y_train, oof_pred)
    results[name] = {
        'fold_accs': fold_accs,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'oof_acc': oof_acc
    }
    oof_predictions[name] = oof_pred
    print(f"{name:30s} | Mean: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f} | OOF: {oof_acc:.4f} | Folds: {[round(x, 4) for x in fold_accs]}")

# Leakage audit: Random Stratified 5-Fold comparison for Random Forest
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_random_accs = []
for tr_idx, va_idx in skf.split(F_train, y_train):
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(F_train[tr_idx], y_train[tr_idx])
    rf_random_accs.append(accuracy_score(y_train[va_idx], rf.predict(F_train[va_idx])))
print(f"\nRandom Forest under Random Stratified 5-Fold: Mean = {np.mean(rf_random_accs):.4f} +/- {np.std(rf_random_accs):.4f}")
print(f"Random Forest under Proxy-Subject 5-Fold:       Mean = {results['Random Forest (300 trees)']['mean_acc']:.4f} +/- {results['Random Forest (300 trees)']['std_acc']:.4f}")

# Classification report for best model (Random Forest)
rf_oof = oof_predictions['Random Forest (300 trees)']
print("\nClassification Report (Random Forest OOF):")
print(classification_report(y_train, rf_oof, digits=4))

# Confusion matrix errors
cm = confusion_matrix(y_train, rf_oof, labels=classes)
errors = np.sum(y_train != rf_oof)
print(f"Total misclassifications: {errors} / {len(y_train)} ({errors/len(y_train)*100:.2f}%)")

# Fit on all train data and predict test
final_rf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
final_rf.fit(F_train, y_train)
y_test_pred = final_rf.predict(F_test)

print("\nTrain class distribution vs Test prediction distribution:")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_dist = pd.Series(y_test_pred).value_counts(normalize=True).sort_index()
comp_df = pd.DataFrame({'Train Pct (%)': (train_dist*100).round(2), 'Test Pred Pct (%)': (test_dist*100).round(2)})
comp_df['Test Count'] = pd.Series(y_test_pred).value_counts().sort_index()
print(comp_df)

# Feature importances
importances = final_rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:25]
print("\nTop 10 Feature Importances:")
for i in range(10):
    print(f"  {i+1}. {names[top_idx[i]]}: {importances[top_idx[i]]:.5f}")
