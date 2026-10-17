"""Check that the benchmark's published_insample arm reproduces the notebook.

Uses the fixture and settings of notebook_reference.py (seed 123, 4 epochs,
20 trees, RFECV floor 254) on the notebook's own split (A2_1-3 -> B2_4).

    python tests/parity/benchmark_arm_parity.py FIXTURE_DIR REF.npz OUT_DIR

Stage 1 is PyTorch, so it cannot reproduce the notebook's Keras probabilities
bit for bit. The CNN is therefore replaced by a stand-in that checks it is
handed exactly the notebook's scaled training and test matrices and returns
the notebook's probabilities; everything around it (data preparation, split,
trimming, scaling, stacking, RFECV forest, scoring) is then checked exactly.
torch_vs_keras.py covers the CNN itself.
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
    logging.basicConfig(level=logging.WARNING)

    import ccpredict.models
    from ccpredict.benchmark import BenchmarkConfig, run_benchmark

    ref = np.load(ref_path)
    seen = {}

    class NotebookStageOne:
        """Stand-in for StageOneClassifier that replays the notebook's stage 1."""

        def __init__(self, cfg, num_classes, steps=1):
            self.history = {"loss": list(ref["history_loss"]), "val_loss": list(ref["history_val_loss"])}

        def fit(self, X, y_class, groups=None):
            seen["X_train"] = np.array_equal(X.astype(np.float32), ref["X_train_nn"].astype(np.float32))
            seen["y_train_class"] = np.array_equal(y_class, ref["y_train_class"])
            return self

        def predict_proba(self, X):
            seen["X_test"] = np.array_equal(X.astype(np.float32), ref["X_test_nn"].astype(np.float32))
            return ref["y_pred_nn"]

    ccpredict.models.StageOneClassifier = NotebookStageOne

    cfg = BenchmarkConfig(scheme="fixed", arms=("published_insample",), run_split_optimism=False,
                          output_dir=out_dir, n_boot=20)
    p = cfg.pipeline
    # The notebook's label semantics, pinned (the package defaults differ now).
    p.features.label_frame, p.features.include_last_window = "start_plus_trajl", False
    p.paths.traj_dir, p.paths.vae_dir = fixture, os.path.join(fixture, "o2vae")
    p.paths.reporter_dir, p.paths.celltraj_path = os.path.join(fixture, "reporter"), fixture
    p.classifier.epochs, p.classifier.seed, p.classifier.verbose = 4, 123, 0
    p.regressor.n_estimators, p.regressor.min_features_to_select = 20, 254
    run_benchmark(cfg)

    got = np.load(os.path.join(out_dir, "predictions", "fixed_B2_4__published_insample__B2_4.npz"))
    checks = {
        "y_test": (ref["y_test"], got["y_true"], 0.0),
        "forest predictions": (ref["y_pred_RFreg"], got["y_pred"], 1e-12),
    }
    bad = 0
    for name in ("X_train", "y_train_class", "X_test"):
        print(f"{'stage-1 input ' + name:<22} {'OK' if seen.get(name) else 'DIFFERS'}")
        bad += not seen.get(name)
    for name, (a, b, tol) in checks.items():
        same = a.shape == b.shape and (np.array_equal(a, b) if tol == 0 else np.allclose(a, b, rtol=0, atol=tol))
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
        print(f"{name:<22} {'OK' if same else 'DIFFERS'}  max|diff| {diff:.1e}")
        bad += not same
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
