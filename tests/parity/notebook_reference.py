"""Execute the notebook's own code cells on the synthetic fixture and dump its arrays.

Only these edits are made to the notebook source, each asserted to match once:
  - data paths -> fixture directory
  - epochs=50 -> --epochs (training time)
  - a keras seed is set right before create_model (the notebook sets none)
  - forest size and RFECV floor -> --n-estimators / --min-features (runtime)
The last cell (multi-GPU MirroredStrategy, marked untested in the notebook) is skipped.

    python tests/parity/notebook_reference.py NOTEBOOK FIXTURE_DIR OUT.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import types


def notebook_code(nb_path: str, fixture: str, epochs: int, seed: int,
                  n_estimators: int, min_features: int) -> list[str]:
    with open(nb_path) as fh:
        nb = json.load(fh)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cells = [c for c in cells if "MirroredStrategy" not in c]

    edits = [
        ("sys.path.append('/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj')",
         f"sys.path.append({fixture!r})"),
        ("pathSet = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/'",
         f"pathSet = {fixture + '/'!r}"),
        ("f'../o2vae/features_vae", f"f'{fixture}/o2vae/features_vae"),
        ("file_path = '/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/cellCycInfoRawReporterImgs/'",
         f"file_path = {fixture + '/reporter/'!r}"),
        ("epochs=50,", f"epochs={epochs},"),
        ("model_nn = create_model(input_shape, num_classes)",
         f"keras.utils.set_random_seed({seed}); model_nn = create_model(input_shape, num_classes)"),
        ("n_estimators=100,", f"n_estimators={n_estimators},"),
        ("min_features_to_select = 50,", f"min_features_to_select = {min_features},"),
    ]
    source = "\n#<<cell>>\n".join(cells)
    for old, new in edits:
        count = source.count(old)
        assert count == 1, f"expected exactly one match for {old!r}, found {count}"
        source = source.replace(old, new)
    return source.split("\n#<<cell>>\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("fixture")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-estimators", type=int, default=20)
    ap.add_argument("--min-features", type=int, default=254)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    sys.modules["umap"] = types.ModuleType("umap")  # imported by the notebook, never used
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import keras  # noqa: F401  (used by the injected seed line)

    ns: dict = {"keras": keras}
    for code in notebook_code(args.notebook, args.fixture, args.epochs, args.seed,
                              args.n_estimators, args.min_features):
        exec(compile(code, "notebook", "exec"), ns)

    import numpy as np

    out = {
        "indgood_models": ns["indgood_models"],
        "cellnumber_stdSet": ns["cellnumber_stdSet"],
        "original_indices_train": ns["original_indices_train"],
        "y_train_trim": ns["y_train_trim"],
        "X_train_nn": ns["X_train_nn"],
        "X_test_nn": ns["X_test_nn"],
        "y_test": ns["y_test"],
        "y_train_class": ns["y_train_class"],
        "y_test_class": ns["y_test_class"],
        "X_train_resampled": ns["X_train_resampled"],
        "y_train_resampled": ns["y_train_resampled"],
        "y_pred_nn": ns["y_pred_nn"],
        "predicted_classes": ns["predicted_classes"],
        "clf_metrics": np.array([ns["accuracy"], ns["precision"], ns["recall"], ns["f1"]]),
        "conf_matrix": ns["conf_matrix"],
        "history_loss": np.array(ns["history_nn"].history["loss"]),
        "history_val_loss": np.array(ns["history_nn"].history["val_loss"]),
        "y_pred_RFreg": ns["y_pred_RFreg"],
        "rfecv_support": ns["model_RFreg_fitted"].support_,
        "reg_metrics": np.array([ns["mse_testRF"], ns["R_testRF"], ns["R2_testRF"]]),
    }
    for i, (Xi, yi) in enumerate(zip(ns["X"], ns["y"])):
        out[f"X_{i}"], out[f"y_{i}"] = Xi, yi
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
