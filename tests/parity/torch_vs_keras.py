"""Statistical equivalence of the PyTorch stage 1 and the notebook's Keras stage 1.

    python tests/parity/torch_vs_keras.py BENCH_CONFIG.json OUT_DIR [--seeds 10] [...]

Both frameworks train the same CNN with the same protocol (oversampling,
keras_tail validation by default, early stopping) on the same data; only the
seed changes between runs. Dropout masks, shuffling and float kernels differ
between frameworks, so single runs cannot match. What must match is the
distribution over seeds. For each metric the report gives the mean and sd per
framework, the Welch 95% CI of the difference, a Mann-Whitney p-value and the
difference in units of the within-framework sd.

It also compares the predictions themselves: the correlation of seed-averaged
p(class 2) - p(class 0) between frameworks is set against the same correlation
between two halves of one framework's seeds. If the port behaves like Keras,
cross-framework agreement is as high as Keras agrees with itself.

Data: the benchmark's prepared (cached) data, the training replicate trimmed
as in the benchmark, then optionally subsampled by whole trajectories to
--max-train-rows so that 2 x seeds trainings fit in a reasonable time.
Results are written after every run (runs.csv), so an interrupted run keeps
what it has; rerunning skips the (framework, seed) pairs already in runs.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

log = logging.getLogger("torch_vs_keras")

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "log_loss", "spearman_score_vs_nc",
           "mean_p0", "mean_p1", "mean_p2", "epochs", "best_epoch", "final_val_loss", "best_val_loss",
           "seconds"]


def _data(cfg_path: str, args):
    from ccpredict.benchmark import BenchmarkConfig, column_selector, load_data, make_block, make_splits

    bcfg = BenchmarkConfig.from_json(cfg_path)
    fov_data, meta = load_data(bcfg)
    _, train_fovs, test_fovs = make_splits(bcfg, meta["good_fovs"])[0]
    if args.train_fovs:
        train_fovs = tuple(args.train_fovs.split(","))
    if args.test_fovs:
        test_fovs = tuple(args.test_fovs.split(","))
    s = bcfg.pipeline.split
    train = make_block(fov_data, train_fovs, (s.trim_bins, s.trim_target_density, s.trim_seed)
                       if bcfg.trim_training else None)
    test = make_block(fov_data, test_fovs)
    if args.max_train_rows and len(train) > args.max_train_rows:
        # Whole trajectories, so held-out-trajectory validation still makes sense.
        rng = np.random.default_rng(0)
        groups = train.groups
        keep, n = [], 0
        for g in rng.permutation(np.unique(groups)):
            rows = np.flatnonzero(groups == g)
            keep.append(rows)
            n += rows.size
            if n >= args.max_train_rows:
                break
        keep = np.sort(np.concatenate(keep))
        train = type(train)(train.X[keep], train.y[keep], train.fold[keep], train.traj[keep], train.fovs)
    cols = column_selector(meta)("vae")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(train.X[:, cols])
    clf_cfg = copy.deepcopy(bcfg.pipeline.classifier)
    clf_cfg.verbose = 0
    clf_cfg.validation = args.validation
    if args.epochs is not None:
        clf_cfg.epochs = args.epochs
    if args.patience is not None:
        clf_cfg.patience = args.patience
    return (scaler.transform(train.X[:, cols]), train.y, train.groups,
            scaler.transform(test.X[:, cols]), test.y, clf_cfg, train_fovs, test_fovs)


def _fit(framework: str, cfg, X, y_class, groups, num_classes):
    if framework == "torch":
        from ccpredict.models import StageOneClassifier

        clf = StageOneClassifier(cfg, num_classes).fit(X, y_class, groups)
        hist = dict(clf.history)
    else:
        from keras_reference import KerasStageOne

        clf = KerasStageOne(cfg, num_classes).fit(X, y_class, groups)
        hist = dict(clf.history)
        # The epoch Keras restored: replay its rule (min_delta, so not simply the argmin);
        # test_torch_port.py checks that this rule makes Keras' own decisions.
        from ccpredict.cnn import EarlyStopping

        stopper = EarlyStopping(cfg.patience, cfg.min_delta)
        for e, v in enumerate(hist["val_loss"]):
            stopper.update(e, v, lambda e=e: e)      # the snapshot must not be None
        hist["best_epoch"] = [stopper.best_epoch + 1]
    return clf, hist


def _metrics(proba, y_test, y_class_test, hist, seconds, num_classes):
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    pred = proba.argmax(1)
    p = np.clip(proba.astype(np.float64), 1e-12, 1)
    p /= p.sum(1, keepdims=True)
    out = {
        "accuracy": accuracy_score(y_class_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_class_test, pred),
        "f1_macro": f1_score(y_class_test, pred, labels=list(range(num_classes)), average="macro",
                             zero_division=0),
        "log_loss": log_loss(y_class_test, p, labels=list(range(num_classes))),
        "spearman_score_vs_nc": spearmanr(proba[:, -1] - proba[:, 0], y_test).statistic,
        "epochs": len(hist["val_loss"]),
        "best_epoch": hist["best_epoch"][0],
        "final_val_loss": hist["val_loss"][-1],
        "best_val_loss": min(hist["val_loss"]),
        "seconds": seconds,
    }
    for k in range(num_classes):
        out[f"mean_p{k}"] = float(proba[:, k].mean())
    return {k: float(v) for k, v in out.items()}


def _load_runs(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [{k: (v if k == "framework" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def _welch(a, b):
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    if se == 0:
        return d, d, d, 1.0
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    t = stats.t.ppf(0.975, df)
    p_mw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return d, d - t * se, d + t * se, p_mw


def write_report(out_dir, runs, proba_dir, y_test, info):
    seeds = {fw: sorted(int(r["seed"]) for r in runs if r["framework"] == fw) for fw in ("keras", "torch")}
    lines = ["# PyTorch vs Keras stage 1", "",
             f"Train FOVs {info['train_fovs']} ({info['n_train']} rows), test FOVs {info['test_fovs']} "
             f"({info['n_test']} rows); validation={info['validation']}, epochs<={info['epochs']}, "
             f"patience={info['patience']}, batch={info['batch_size']}.",
             f"Seeds: keras {len(seeds['keras'])}, torch {len(seeds['torch'])}.", "",
             "Difference = torch - keras. 'd/sd' is the difference over the pooled within-framework sd: "
             "|d/sd| well below 1 means the framework changes the result less than the seed does.", "",
             "| metric | keras mean ± sd | torch mean ± sd | difference [95% CI] | d/sd | Mann-Whitney p |",
             "|---|---|---|---|---|---|"]
    for m in METRICS:
        a = [r[m] for r in runs if r["framework"] == "keras"]
        b = [r[m] for r in runs if r["framework"] == "torch"]
        if len(a) < 2 or len(b) < 2:
            continue
        d, lo, hi, p = _welch(a, b)
        sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        lines.append(f"| {m} | {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f} | {np.mean(b):.4f} ± "
                     f"{np.std(b, ddof=1):.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{d / sd if sd > 0 else 0:+.2f} | {p:.3f} |")

    # Function-space agreement: seed-averaged score p2 - p0 on the test rows.
    def score(fw, ss):
        return np.mean([np.load(os.path.join(proba_dir, f"{fw}_{s}.npy")) for s in ss], 0)

    k, t = seeds["keras"], seeds["torch"]
    if len(k) >= 4 and len(t) >= 4:
        hk, ht = len(k) // 2, len(t) // 2
        sc = lambda P: P[:, -1] - P[:, 0]  # noqa: E731
        kk = np.corrcoef(sc(score("keras", k[:hk])), sc(score("keras", k[hk:])))[0, 1]
        tt = np.corrcoef(sc(score("torch", t[:ht])), sc(score("torch", t[ht:])))[0, 1]
        kt = np.mean([np.corrcoef(sc(score("keras", k[:hk])), sc(score("torch", t[ht:])))[0, 1],
                      np.corrcoef(sc(score("keras", k[hk:])), sc(score("torch", t[:ht])))[0, 1]])
        lines += ["", "## Do the two frameworks learn the same function?", "",
                  "Pearson r of the seed-averaged score p(class 2) - p(class 0) over the test rows, "
                  "each average over half of the seeds:", "",
                  f"- keras half vs keras half: {kk:.4f}",
                  f"- torch half vs torch half: {tt:.4f}",
                  f"- keras half vs torch half: {kt:.4f}", "",
                  "If the port were wrong in a way that matters, the last number would sit clearly "
                  "below the first two."]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="benchmark config JSON (data paths, QC, classifier settings)")
    ap.add_argument("out_dir")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frameworks", default="keras,torch")
    ap.add_argument("--train-fovs", default=None, help="comma-separated; default: the config's training replicate")
    ap.add_argument("--test-fovs", default=None, help="comma-separated; default: the config's test replicate")
    ap.add_argument("--max-train-rows", type=int, default=None, help="subsample whole trajectories")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--validation", default="keras_tail", choices=["keras_tail", "grouped_holdout"])
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for both frameworks")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if args.threads:
        import tensorflow as tf
        import torch

        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        torch.set_num_threads(args.threads)

    from ccpredict.preprocessing import discretize

    Xtr, ytr, groups, Xte, yte, cfg, train_fovs, test_fovs = _data(args.config, args)
    num_classes = len(cfg.class_boundaries) + 1
    ctr, cte = discretize(ytr, cfg.class_boundaries), discretize(yte, cfg.class_boundaries)
    os.makedirs(args.out_dir, exist_ok=True)
    proba_dir = os.path.join(args.out_dir, "proba")
    os.makedirs(proba_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "y_test.npy"), yte)
    info = {"train_fovs": list(train_fovs), "test_fovs": list(test_fovs), "n_train": int(ytr.size),
            "n_test": int(yte.size), "validation": cfg.validation, "epochs": cfg.epochs,
            "patience": cfg.patience, "batch_size": cfg.batch_size,
            "train_class_counts": np.bincount(ctr, minlength=num_classes).tolist(),
            "test_class_counts": np.bincount(cte, minlength=num_classes).tolist()}
    with open(os.path.join(args.out_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    log.info("data: %s", info)

    runs_path = os.path.join(args.out_dir, "runs.csv")
    runs = _load_runs(runs_path)
    done = {(r["framework"], int(r["seed"])) for r in runs}
    frameworks = args.frameworks.split(",")
    # Interleave frameworks so that a partial run is balanced.
    for seed in range(args.seeds):
        for fw in frameworks:
            if (fw, seed) in done:
                continue
            c = copy.deepcopy(cfg)
            c.seed = seed
            t0 = time.time()
            clf, hist = _fit(fw, c, Xtr, ctr, groups, num_classes)
            seconds = time.time() - t0
            proba = np.asarray(clf.predict_proba(Xte), np.float32)
            np.save(os.path.join(proba_dir, f"{fw}_{seed}.npy"), proba)
            with open(os.path.join(proba_dir, f"{fw}_{seed}_history.json"), "w") as fh:
                json.dump({k: [float(x) for x in v] for k, v in hist.items()}, fh)
            row = {"framework": fw, "seed": seed, **_metrics(proba, yte, cte, hist, seconds, num_classes)}
            runs.append(row)
            with open(runs_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["framework", "seed"] + METRICS)
                w.writeheader()
                w.writerows(runs)
            log.info("%s seed %d: acc %.4f bal %.4f logloss %.4f epochs %d best %d (%.0fs)", fw, seed,
                     row["accuracy"], row["balanced_accuracy"], row["log_loss"], row["epochs"],
                     row["best_epoch"], seconds)
            if fw == "keras":
                import keras

                keras.backend.clear_session()
    print(open(write_report(args.out_dir, runs, proba_dir, yte, info)).read())


if __name__ == "__main__":
    main()
