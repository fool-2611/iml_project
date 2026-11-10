import os
import sys
import time
import nbformat as nbf
from nbclient import NotebookClient

notebooks = [
    "notebooks/01_exploration.ipynb",
    "notebooks/02_cleaning.ipynb",
    "notebooks/03_residual_noise.ipynb",
    "notebooks/04_validation_strategy.ipynb",
    "notebooks/05_feature_extraction.ipynb",
    "notebooks/06_models_and_submission.ipynb"
]

print("=" * 70)
print("STARTING FULL END-TO-END PIPELINE VERIFICATION FROM CLEAN STATE")
print("=" * 70)

overall_start = time.time()
errors = []

for nb_rel_path in notebooks:
    print(f"\n---> Executing {nb_rel_path} top-to-bottom...")
    start_t = time.time()
    
    with open(nb_rel_path) as f:
        nb = nbf.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    try:
        client.execute()
        with open(nb_rel_path, "w") as f:
            nbf.write(nb, f)
        elapsed = time.time() - start_t
        print(f"     [SUCCESS] Completed in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_t
        print(f"     [ERROR] Failed after {elapsed:.2f}s: {e}")
        errors.append((nb_rel_path, str(e)))

print("\n" + "=" * 70)
print(f"PIPELINE VERIFICATION COMPLETE IN {time.time() - overall_start:.2f}s")
if errors:
    print(f"FAILED NOTEBOOKS: {len(errors)}")
    for nb_path, err in errors:
        print(f" - {nb_path}: {err}")
    sys.exit(1)
else:
    print("ALL 6 NOTEBOOKS EXECUTED WITH ZERO ERRORS!")
print("=" * 70)

# Verify key artifacts
expected_outputs = [
    "outputs/clean_data.npz",
    "outputs/clean_data_v2.npz",
    "outputs/splits.npz",
    "outputs/features.npz",
    "outputs/submission.csv"
]

print("\nVerifying Generated Artifacts:")
for out_path in expected_outputs:
    exists = os.path.exists(out_path)
    sz = os.path.getsize(out_path) if exists else 0
    print(f"  - {out_path:30s}: {'EXISTS' if exists else 'MISSING'} ({sz:,} bytes)")
    assert exists, f"Missing {out_path}"
    assert sz > 0, f"Empty file: {out_path}"

# Check submission integrity
import pandas as pd
sub = pd.read_csv("outputs/submission.csv")
assert sub.shape == (403, 2), f"Bad submission shape: {sub.shape}"
assert list(sub.columns) == ['id', 'target'], f"Bad columns: {sub.columns}"
assert sub['target'].isnull().sum() == 0, "Submission contains nulls"
print(f"\nFinal Submission Status: 100% VALID ({len(sub)} rows, 0 nulls)")
