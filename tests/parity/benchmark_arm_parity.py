"""Check that the benchmark's published_insample arm reproduces the notebook.

Uses the fixture and settings of notebook_reference.py (seed 123, 4 epochs,
20 trees, RFECV floor 254) on the notebook's own split (A2_1-3 -> B2_4).

    python tests/parity/benchmark_arm_parity.py FIXTURE_DIR REF.npz OUT_DIR
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main(fixture: str, ref_path: str, out_dir: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    logging.basicConfig(level=logging.WARNING)

    from ccpredict.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(scheme="fixed", arms=("published_insample",), run_split_optimism=False,
                          output_dir=out_dir, n_boot=20)
    p = cfg.pipeline
    p.paths.traj_dir, p.paths.vae_dir = fixture, os.path.join(fixture, "o2vae")
    p.paths.reporter_dir, p.paths.celltraj_path = os.path.join(fixture, "reporter"), fixture
    p.classifier.epochs, p.classifier.seed, p.classifier.verbose = 4, 123, 0
    p.regressor.n_estimators, p.regressor.min_features_to_select = 20, 254
    run_benchmark(cfg)

    ref = np.load(ref_path)
    got = np.load(os.path.join(out_dir, "predictions", "fixed_B2_4__published_insample__B2_4.npz"))
    checks = {
        "y_test": (ref["y_test"], got["y_true"], 0.0),
        "class probabilities": (ref["y_pred_nn"], got["proba"], 0.0),
        "forest predictions": (ref["y_pred_RFreg"], got["y_pred"], 1e-12),
    }
    bad = 0
    for name, (a, b, tol) in checks.items():
        same = a.shape == b.shape and (np.array_equal(a, b) if tol == 0 else np.allclose(a, b, rtol=0, atol=tol))
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
        print(f"{name:<22} {'OK' if same else 'DIFFERS'}  max|diff| {diff:.1e}")
        bad += not same
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
