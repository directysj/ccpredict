"""Compare the arrays dumped by notebook_reference.py and package_run.py.

    python tests/parity/compare.py NOTEBOOK.npz PACKAGE.npz
Exit code 1 if any array differs.
"""
from __future__ import annotations

import sys

import numpy as np


def main(ref_path: str, pkg_path: str) -> int:
    ref, pkg = np.load(ref_path), np.load(pkg_path)
    good = set(ref["indgood_models"].tolist())
    failures = 0
    print(f"{'array':<24}{'shape':>16}  result")
    for key in ref.files:
        a = ref[key]
        if key[:2] in ("X_", "y_") and key[2:].isdigit() and int(key[2:]) not in good:
            ok = a.shape[0] == 0 and key not in pkg.files
            print(f"{key:<24}{str(a.shape):>16}  {'empty in notebook, absent in package (QC-failed FOV)' if ok else 'MISMATCH'}")
            failures += not ok
            continue
        if key not in pkg.files:
            print(f"{key:<24}{str(a.shape):>16}  MISSING in package")
            failures += 1
            continue
        b = pkg[key]
        if a.shape != b.shape:
            print(f"{key:<24}{str(a.shape):>16}  SHAPE {b.shape}")
            failures += 1
            continue
        if np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            print(f"{key:<24}{str(a.shape):>16}  identical")
        elif a.dtype.kind == "f" and np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
            # RandomForestRegressor.predict with n_jobs>1 sums tree outputs in
            # thread order, so even one fitted model differs in the last bits
            # between calls. Anything above 1e-12 still counts as a mismatch.
            diff = np.nanmax(np.abs(a - b))
            print(f"{key:<24}{str(a.shape):>16}  equal to {diff:.1e} (parallel float summation order)")
        else:
            diff = np.nanmax(np.abs(a.astype(float) - b.astype(float))) if a.size else 0.0
            print(f"{key:<24}{str(a.shape):>16}  DIFFERS, max |diff| = {diff:.3e}")
            failures += 1
    print(f"\n{failures} mismatches")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
