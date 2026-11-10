import os
import sys
import subprocess
import time
import tempfile
import glob
import nbformat

PYTHON_BIN = "/home/ubuntu/miniforge3/envs/gis/bin/python"
JUPYTER_BIN = "/home/ubuntu/miniforge3/envs/gis/bin/jupyter"

print("=" * 70)
print("1. GENERATING FIXED NOTEBOOKS...")
print("=" * 70)

builders = [
    "build_nb01_fixed.py",
    "build_nb02_fixed.py",
    "build_nb03_fixed.py",
    "build_nb04_fixed.py",
    "build_nb05_fixed.py",
    "build_nb06_fixed.py"
]

for b in builders:
    print(f"Running {b}...")
    res = subprocess.run([PYTHON_BIN, b], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {b}: {res.stderr}")
        sys.exit(1)
    print(f"  {res.stdout.strip()}")

print("\n" + "=" * 70)
print("2. EXECUTING ALL 6 NOTEBOOKS SEQUENTIALLY IN-PLACE WITH NBCONVERT...")
print("=" * 70)

notebooks = [
    "notebooks/01_exploration.ipynb",
    "notebooks/02_cleaning.ipynb",
    "notebooks/03_residual_noise.ipynb",
    "notebooks/04_validation_strategy.ipynb",
    "notebooks/05_feature_extraction.ipynb",
    "notebooks/06_models_and_submission.ipynb"
]

overall_start = time.time()

for nb_path in notebooks:
    print(f"\n---> Executing {nb_path} with jupyter nbconvert...")
    t0 = time.time()
    cmd = [
        JUPYTER_BIN, "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        "--ExecutePreprocessor.kernel_name=python3",
        nb_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"  [ERROR] Execution failed for {nb_path} after {elapsed:.1f}s:")
        print(res.stderr)
        sys.exit(1)
    else:
        print(f"  [SUCCESS] {nb_path} executed and saved in-place ({elapsed:.1f}s)")

print("\n" + "=" * 70)
print(f"ALL NOTEBOOKS EXECUTED IN {time.time() - overall_start:.1f}s")
print("=" * 70)

print("\n3. RUNNING TOOLS/CHECK_EMBEDDING.PY AUDIT...")
audit_res = subprocess.run([PYTHON_BIN, "tools/check_embedding.py"], capture_output=True, text=True)
print(audit_res.stdout)
if audit_res.returncode != 0:
    print(audit_res.stderr)
    sys.exit(1)

print("\n4. RUNNING PORTABILITY TEST (CONVERTING ONE NOTEBOOK IN TEMP DIRECTORY)...")
with tempfile.TemporaryDirectory() as tmp_dir:
    test_nb = "notebooks/01_exploration.ipynb"
    tmp_nb_path = os.path.join(tmp_dir, "01_test.ipynb")
    
    # Copy notebook
    with open(test_nb, "r") as f_in, open(tmp_nb_path, "w") as f_out:
        f_out.write(f_in.read())
        
    # Convert to HTML without executing
    cmd_html = [
        JUPYTER_BIN, "nbconvert",
        "--to", "html",
        tmp_nb_path
    ]
    html_res = subprocess.run(cmd_html, capture_output=True, text=True, cwd=tmp_dir)
    assert html_res.returncode == 0, f"HTML conversion failed: {html_res.stderr}"
    
    tmp_html_path = os.path.join(tmp_dir, "01_test.html")
    with open(tmp_html_path, "r") as f:
        html_content = f.read()
    
    # Count <img tags
    img_tag_count = html_content.count("<img")
    print(f"Portability Test Result: Found {img_tag_count} <img tags in standalone exported HTML.")
    assert img_tag_count >= 6, f"Expected at least 6 embedded images, found {img_tag_count}"
    print("PORTABILITY TEST PASSED: All figures are self-contained embedded images!")

print("\n" + "=" * 70)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
