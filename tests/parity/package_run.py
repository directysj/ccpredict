"""Run the package on the synthetic fixture with settings matching notebook_reference.py.

    python tests/parity/package_run.py FIXTURE_DIR OUT.npz [--epochs 4 --seed 123 ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-estimators", type=int, default=20)
    ap.add_argument("--min-features", type=int, default=254)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from ccpredict import CellCyclePipeline, PipelineConfig
    from ccpredict.preprocessing import make_oversampler

    cfg = PipelineConfig()
    cfg.paths.traj_dir = args.fixture
    cfg.paths.vae_dir = os.path.join(args.fixture, "o2vae")
    cfg.paths.reporter_dir = os.path.join(args.fixture, "reporter")
    cfg.paths.celltraj_path = args.fixture
    cfg.paths.output_dir = args.output_dir
    cfg.classifier.epochs = args.epochs
    cfg.classifier.seed = args.seed
    cfg.regressor.n_estimators = args.n_estimators
    cfg.regressor.min_features_to_select = args.min_features

    pipe = CellCyclePipeline(cfg)
    res = pipe.run(make_figures=args.output_dir is not None)
    tt, co, ro = res.train_test, res.classifier_output, res.regressor_output
    X_res, y_res = make_oversampler(cfg.classifier.oversampler, cfg.classifier.oversampling_strategy,
                                    cfg.classifier.oversampling_seed).fit_resample(tt.X_train, co.y_train_class)

    m = co.metrics
    fov_index = {f: i for i, f in enumerate(cfg.fovs)}
    out = {
        "indgood_models": np.array([fov_index[f] for f in res.data.good_fovs]),
        "cellnumber_stdSet": np.array([res.data.cell_count_noise[f] for f in cfg.fovs]),
        "original_indices_train": tt.train_rows,
        "y_train_trim": tt.y_train,
        "X_train_nn": tt.X_train,
        "X_test_nn": tt.X_test,
        "y_test": tt.y_test,
        "y_train_class": co.y_train_class,
        "y_test_class": co.y_test_class,
        "X_train_resampled": X_res,
        "y_train_resampled": y_res,
        "y_pred_nn": co.proba_test,
        "predicted_classes": co.predicted_test,
        "clf_metrics": np.array([m["accuracy"], m["precision_weighted"], m["recall_weighted"], m["f1_weighted"]]),
        "conf_matrix": np.array(m["confusion_matrix_true_normalized"]),
        "history_loss": np.array(res.classifier.history["loss"]),
        "history_val_loss": np.array(res.classifier.history["val_loss"]),
        "y_pred_RFreg": ro.y_pred,
        "rfecv_support": res.regressor.rfecv.support_,
        "reg_metrics": np.array([ro.metrics["mse"], ro.metrics["pearson_r"], ro.metrics["r2"]]),
    }
    for f, lab in res.data.fovs.items():
        out[f"X_{fov_index[f]}"], out[f"y_{fov_index[f]}"] = lab.X, lab.y
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
