#!/usr/bin/env python3
"""
tools/check_embedding.py

Audits executed Jupyter notebooks to ensure that:
1. Every code cell that generates a plot (calls savefig or plt.show) has an 'image/png' in its output MIME bundle.
2. No code cell contains an error output.
3. No markdown cell links to an external 'figures/...' path (ensuring self-containment).
4. Each notebook's file size is under 25 MB.
"""

import sys
import os
import glob
import re
import nbformat

def check_notebook(nb_path):
    print(f"\nChecking: {nb_path}")
    nb = nbformat.read(nb_path, as_version=4)
    file_size_mb = os.path.getsize(nb_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 25.0:
        print(f"  [FAIL] File size exceeds 25 MB limit ({file_size_mb:.2f} MB)")
        return False, 0
    
    fig_count = 0
    errors = []
    
    for idx, cell in enumerate(nb.cells):
        # Check markdown cells for external figures links
        if cell.cell_type == 'markdown':
            # Look for markdown image or link syntax pointing to figures/
            if re.search(r'!?\[.*?\]\((\.\./)?figures/', cell.source):
                errors.append(f"Cell {idx} (markdown): Contains external reference to figures/")
        
        # Check code cells
        elif cell.cell_type == 'code':
            # Check for execution errors
            for out in cell.outputs:
                if out.output_type == 'error':
                    errors.append(f"Cell {idx} (code): Execution error '{out.get('ename')}: {out.get('evalue')}'")
            
            # Check if cell plots
            calls_plot = 'plt.show(' in cell.source or 'savefig(' in cell.source
            if calls_plot:
                has_png = False
                for out in cell.outputs:
                    data = out.get('data', {})
                    if 'image/png' in data:
                        has_png = True
                        fig_count += 1
                if not has_png:
                    errors.append(f"Cell {idx} (code): Calls plotting functions but has no 'image/png' output")

    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        return False, fig_count
    else:
        print(f"  [PASS] Embedded figures: {fig_count}, Errors: 0, Self-contained: True")
        return True, fig_count

def main():
    nb_dir = sys.argv[1] if len(sys.argv) > 1 else "notebooks"
    nb_files = sorted(glob.glob(os.path.join(nb_dir, "*.ipynb")))
    if not nb_files:
        print(f"No notebooks found in {nb_dir}")
        sys.exit(1)
        
    all_passed = True
    total_figs = 0
    for nb_file in nb_files:
        passed, count = check_notebook(nb_file)
        total_figs += count
        if not passed:
            all_passed = False
            
    print("\n" + "=" * 50)
    print(f"Total embedded figures across all notebooks: {total_figs}")
    if all_passed:
        print("ALL NOTEBOOKS PASSED SELF-CONTAINMENT & EMBEDDING AUDIT")
        sys.exit(0)
    else:
        print("SOME NOTEBOOKS FAILED THE AUDIT")
        sys.exit(1)

if __name__ == "__main__":
    main()
