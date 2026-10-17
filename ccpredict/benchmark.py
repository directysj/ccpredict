"""Honest comparison of the published model with baselines and corrected protocols.

    python -m ccpredict.benchmark --write-config bench.json   # edit paths, then:
    python -m ccpredict.benchmark --config bench.json --output-dir results/

Default protocol (the paper's question): train on the EGF1 FOVs (A wells),
test on every EGF2 FOV (B wells) separately. Training labels are trimmed to a
flat histogram, as in the paper. Every arm sees the same rows.

Outputs in --output-dir:
    report.md          readable summary
    summary.csv        per arm: mean and sd over test FOVs
    results.csv        per arm x test FOV: all metrics
    results.json       the same, nested, with classifier metrics and run info
    diagnostics.json   data/protocol checks (trajectory overlap, label shift,
                       validation-slice composition, random-vs-grouped optimism)
    predictions/       per-row predictions and labels (optional)
    cache/             prepared features, so a rerun skips the slow loading
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from . import diagnostics as diag
from .arms import DEFAULT_ARMS, ArmResult, Block, SplitContext, make_arm
from .config import DEFAULT_FOV_REPLICATE, PipelineConfig, _from_dict
from .evaluation import as_jsonable, classification_metrics, regression_report
from .preprocessing import discretize, trim_by_histogram

log = logging.getLogger(__name__)
CACHE_VERSION = 1


@dataclass
class BenchmarkConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    scheme: Literal["cross_replicate", "lofo", "fixed"] = "cross_replicate"
    replicate_of: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FOV_REPLICATE))
    train_replicate: str = "EGF1"
    test_replicate: str = "EGF2"
    arms: Tuple[str, ...] = tuple(DEFAULT_ARMS)
    trim_training: bool = True
    # Seed for the CNN when pipeline.classifier.seed is None (the notebook set none).
    seed: int = 0
    n_boot: int = 200
    min_traj_len: int = 20
    run_split_optimism: bool = True
    optimism_trees: int = 50
    save_predictions: bool = True
    output_dir: str = "benchmark_results"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        with open(path) as fh:
            d = json.load(fh)
        pipeline = PipelineConfig.from_dict(d.pop("pipeline", {}))
        obj = _from_dict(cls, d)
        obj.pipeline = pipeline
        return obj

    def quick(self) -> "BenchmarkConfig":
        """Tiny settings for smoke tests; numbers from a quick run mean nothing."""
        q = copy.deepcopy(self)
        q.pipeline.classifier.epochs, q.pipeline.classifier.patience = 3, 2
        q.pipeline.classifier.batch_size = 256
        q.pipeline.net.epochs, q.pipeline.net.patience = 3, 2
        q.pipeline.regressor.n_estimators = 20
        q.pipeline.regressor.rfecv_step = 64
        q.n_boot, q.optimism_trees = 50, 10
        return q


# ----------------------------------------------------------------------------
# Data: prepare once, cache
# ----------------------------------------------------------------------------

def _needs_classical(arms) -> bool:
    return any("classical" in fs for a in arms for fs in make_arm(a).feature_sets)


def _cache_key(cfg: BenchmarkConfig, with_classical: bool) -> str:
    p = cfg.pipeline
    blob = json.dumps({"v": CACHE_VERSION, "fovs": p.fovs, "paths": dataclasses.asdict(p.paths),
                       "qc": dataclasses.asdict(p.qc), "features": dataclasses.asdict(p.features),
                       "classical": with_classical}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_data(cfg: BenchmarkConfig) -> tuple[Dict[str, dict], dict]:
    """Per FOV: X (float64), y, traj, label_frame; plus layout and QC metadata."""
    from .data import prepare_data

    with_classical = _needs_classical(cfg.arms)
    key = _cache_key(cfg, with_classical)
    cache_dir = os.path.join(cfg.output_dir, "cache")
    npz_path, meta_path = (os.path.join(cache_dir, f"prepared_{key}.{ext}") for ext in ("npz", "json"))

    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        p = copy.deepcopy(cfg.pipeline)
        if with_classical and p.features.handcrafted_from_col is None:
            p.features.handcrafted_from_col = 0
        log.info("preparing data (this is the slow part; cached afterwards)")
        data = prepare_data(p.fovs, p.paths, p.qc, p.features)
        os.makedirs(cache_dir, exist_ok=True)
        arrays = {}
        for fov, lab in data.fovs.items():
            # float32 storage; VAE embeddings are float32 already, so they round-trip exactly.
            arrays[f"{fov}__X"] = lab.X.astype(np.float32)
            arrays[f"{fov}__y"] = lab.y
            arrays[f"{fov}__traj"] = lab.snippets.traj_index
            arrays[f"{fov}__label_frame"] = lab.snippets.label_frame
        np.savez(npz_path, **arrays)
        trajl = p.features.trajl
        n_total = next(iter(data.fovs.values())).X.shape[1] if data.fovs else 0
        n_com = 3 * trajl if p.features.append_com else 0
        n_vae = p.features.latent_dim
        meta = {
            "good_fovs": data.good_fovs, "cell_count_noise": data.cell_count_noise,
            "trajl": trajl, "n_vae": n_vae, "n_classical": (n_total - n_com) // trajl - n_vae,
            "trajectory_overlap": {f: diag.trajectory_overlap(t) for f, t in data.trajectories.items()},
        }
        with open(meta_path, "w") as fh:
            json.dump(as_jsonable(meta), fh, indent=2)
    else:
        log.info("using cached prepared data %s", npz_path)

    with open(meta_path) as fh:
        meta = json.load(fh)
    z = np.load(npz_path)
    fov_data = {fov: {"X": z[f"{fov}__X"].astype(np.float64), "y": z[f"{fov}__y"],
                      "traj": z[f"{fov}__traj"], "label_frame": z[f"{fov}__label_frame"]}
                for fov in meta["good_fovs"]}
    return fov_data, meta


def column_selector(meta: dict):
    trajl, n_vae, n_cls = meta["trajl"], meta["n_vae"], meta["n_classical"]
    d = n_vae + n_cls

    def columns(feature_set: str) -> np.ndarray:
        per_cell = {"vae": np.arange(n_vae), "classical": n_vae + np.arange(n_cls),
                    "vae+classical": np.arange(d)}[feature_set]
        if per_cell.size == 0:
            raise ValueError(f"feature set {feature_set!r} has no columns in the prepared data")
        return (np.arange(trajl)[:, None] * d + per_cell[None, :]).ravel()

    return columns


# ----------------------------------------------------------------------------
# Splits and blocks
# ----------------------------------------------------------------------------

def make_splits(cfg: BenchmarkConfig, good: List[str]) -> List[Tuple[str, tuple, tuple]]:
    rep = cfg.replicate_of
    if cfg.scheme == "cross_replicate":
        train = tuple(f for f in good if rep.get(f) == cfg.train_replicate)
        test = tuple(f for f in good if rep.get(f) == cfg.test_replicate)
        if not train or not test:
            raise ValueError(f"no FOVs for {cfg.train_replicate} -> {cfg.test_replicate} among {good}")
        return [(f"{cfg.train_replicate}->{cfg.test_replicate}", train, test)]
    if cfg.scheme == "lofo":
        return [(f"lofo:{f}", tuple(g for g in good if g != f), (f,)) for f in good]
    if cfg.scheme == "fixed":
        s = cfg.pipeline.split
        return [(f"fixed:{s.test_fov}", tuple(s.train_fovs), (s.test_fov,))]
    raise ValueError(cfg.scheme)


def make_block(fov_data: Dict[str, dict], fovs: tuple, trim: Optional[tuple] = None) -> Block:
    X = np.concatenate([fov_data[f]["X"] for f in fovs])
    y = np.concatenate([fov_data[f]["y"] for f in fovs])
    fold = np.concatenate([np.full(fov_data[f]["y"].size, i) for i, f in enumerate(fovs)])
    traj = np.concatenate([fov_data[f]["traj"] for f in fovs])
    if trim is not None:
        rows = trim_by_histogram(y, *trim)
        X, y, fold, traj = X[rows], y[rows], fold[rows], traj[rows]
    return Block(X, y, fold, traj, tuple(fovs))


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

KEY_METRICS = ("pearson_r", "mse", "r2", "spearman", "mae_log", "r2_log", "mean_bias",
               "calibration_slope")


def _flat_row(split, arm, fov, rep, clf_metrics, seconds, info):
    row = {"split": split, "arm": arm, "test_fov": fov, "n": rep["n"]}
    row.update({k: rep[k] for k in KEY_METRICS})
    ci = rep.get("ci95", {})
    for k in ("pearson_r", "mse"):
        if k in ci:
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci[k]
    pt = rep.get("per_trajectory", {})
    row["traj_median_r"] = pt.get("median_r")
    row["traj_frac_r_gt_0.5"] = pt.get("frac_r_gt_0.5")
    if clf_metrics:
        for k in ("accuracy", "majority_class_accuracy", "balanced_accuracy", "f1_macro"):
            row[f"clf_{k}"] = clf_metrics[k]
    row["fit_seconds"] = round(seconds, 1)
    for k, v in info.items():
        if isinstance(v, (int, float, str)):
            row[f"info_{k}"] = v
    return row


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.output_dir, "config.json"))
    clf_cfg = copy.deepcopy(cfg.pipeline.classifier)
    if clf_cfg.seed is None:
        clf_cfg.seed = cfg.seed
    net_cfg = copy.deepcopy(cfg.pipeline.net)
    if net_cfg.seed is None:
        net_cfg.seed = cfg.seed
    boundaries = clf_cfg.class_boundaries
    trim = ((cfg.pipeline.split.trim_bins, cfg.pipeline.split.trim_target_density,
             cfg.pipeline.split.trim_seed) if cfg.trim_training else None)

    fov_data, meta = load_data(cfg)
    columns = column_selector(meta)
    splits = make_splits(cfg, meta["good_fovs"])

    diagnostics = {
        "good_fovs": meta["good_fovs"], "cell_count_noise": meta["cell_count_noise"],
        "trajectory_overlap": meta["trajectory_overlap"],
        "label_distribution": {f: diag.label_distribution(d["y"], boundaries) for f, d in fov_data.items()},
        "splits": {},
    }
    rows, nested = [], []
    for split_name, train_fovs, test_fovs in splits:
        log.info("split %s: train %s, test %s", split_name, train_fovs, test_fovs)
        train = make_block(fov_data, train_fovs, trim)
        tests = {f: make_block(fov_data, (f,)) for f in test_fovs}
        sd = {"train_fovs": train_fovs, "test_fovs": test_fovs,
              "train_label_distribution": diag.label_distribution(train.y, boundaries),
              "keras_tail_validation": diag.keras_tail_validation(
                  discretize(train.y, boundaries), clf_cfg.oversampler, clf_cfg.oversampling_strategy,
                  clf_cfg.oversampling_seed, clf_cfg.validation_fraction)}
        if cfg.run_split_optimism:
            cols = columns("vae")
            sd["split_optimism_rf_vae"] = diag.split_optimism(
                train.X[:, cols], train.y, train.fold, train.groups,
                {f: (b.X[:, cols], b.y) for f, b in tests.items()},
                cfg.pipeline.regressor, n_estimators=cfg.optimism_trees, seed=cfg.seed)
        diagnostics["splits"][split_name] = sd
        _write_json(os.path.join(cfg.output_dir, "diagnostics.json"), diagnostics)

        ctx = SplitContext(split_name, train, tests, columns, clf_cfg, cfg.pipeline.regressor,
                           net_cfg, trajl=meta["trajl"])
        for arm_name in cfg.arms:
            arm = make_arm(arm_name)
            log.info("[%s] arm %s", split_name, arm_name)
            t0 = time.time()
            try:
                results: Dict[str, ArmResult] = arm.run(ctx)
            except Exception as exc:  # keep going: one failing arm must not cost a long run
                log.error("arm %s failed: %s", arm_name, exc)
                nested.append({"split": split_name, "arm": arm_name, "error": traceback.format_exc()})
                continue
            seconds = time.time() - t0
            for fov, res in results.items():
                b = tests[fov]
                rep = regression_report(b.y, res.y_pred, b.traj, min_traj_len=cfg.min_traj_len,
                                        n_boot=cfg.n_boot, seed=cfg.seed)
                clf_m = None
                if res.proba is not None:
                    clf_m = classification_metrics(discretize(b.y, boundaries),
                                                   np.argmax(res.proba, axis=1), len(boundaries) + 1)
                rows.append(_flat_row(split_name, arm_name, fov, rep, clf_m, seconds, res.info))
                nested.append({"split": split_name, "arm": arm_name, "test_fov": fov, "regression": rep,
                               "classification": clf_m, "info": res.info, "fit_seconds": seconds})
                if cfg.save_predictions:
                    pdir = os.path.join(cfg.output_dir, "predictions")
                    os.makedirs(pdir, exist_ok=True)
                    safe = split_name.replace(">", "").replace(":", "_")
                    np.savez(os.path.join(pdir, f"{safe}__{arm_name}__{fov}.npz"), y_true=b.y,
                             y_pred=res.y_pred, traj=b.traj,
                             **({"proba": res.proba} if res.proba is not None else {}))
            _write_outputs(cfg, rows, nested)
            log.info("[%s] %s done in %.0f s: R %s", split_name, arm_name, seconds,
                     ", ".join(f"{f} {r['pearson_r']:.3f}" for f, r in
                               ((x["test_fov"], x) for x in rows[-len(results):])))
        del ctx
    _write_outputs(cfg, rows, nested)
    write_report(cfg, rows, diagnostics)
    return {"rows": rows, "results": nested, "diagnostics": diagnostics}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(as_jsonable(obj), fh, indent=2)


def summarize(rows: List[dict]) -> List[dict]:
    out = []
    by_arm: Dict[tuple, List[dict]] = {}
    for r in rows:
        by_arm.setdefault((r["split"], r["arm"]), []).append(r)
    for (split, arm), rs in by_arm.items():
        s = {"split": split, "arm": arm, "n_test_fovs": len(rs)}
        for k in ("pearson_r", "mse", "r2", "spearman", "mae_log", "traj_median_r", "clf_balanced_accuracy"):
            vals = np.array([r[k] for r in rs if r.get(k) is not None], float)
            if vals.size:
                s[f"{k}_mean"] = float(vals.mean())
                s[f"{k}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out.append(s)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_outputs(cfg, rows, nested):
    _write_csv(os.path.join(cfg.output_dir, "results.csv"), rows)
    _write_csv(os.path.join(cfg.output_dir, "summary.csv"), summarize(rows))
    _write_json(os.path.join(cfg.output_dir, "results.json"), nested)


def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(cfg: BenchmarkConfig, rows: List[dict], diagnostics: dict) -> str:
    lines = ["# ccpredict benchmark", "",
             f"Scheme: `{cfg.scheme}`; training labels trimmed: {cfg.trim_training}; "
             f"snippet length: {cfg.pipeline.features.trajl}; label frame: `{cfg.pipeline.features.label_frame}`; "
             f"CNN device: `{cfg.pipeline.classifier.device}`, batch {cfg.pipeline.classifier.batch_size}; "
             f"nn_* device: `{cfg.pipeline.net.device}`, batch {cfg.pipeline.net.batch_size}, "
             f"seed {cfg.pipeline.net.seed if cfg.pipeline.net.seed is not None else cfg.seed}; "
             f"RFECV step: {cfg.pipeline.regressor.rfecv_step}.",
             "", "R and MSE are on raw N/C (the paper's metrics). The 95% interval resamples whole "
             "trajectories. `*_insample` arms are fitted on the rows they are scored on.", ""]
    test_fovs = list(dict.fromkeys(r["test_fov"] for r in rows))
    hdr = "| arm | " + " | ".join(f"R {f} [95% CI]" for f in test_fovs) + " | mean R | mean MSE | mean R² | per-traj median R | fit s |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for s in summarize(rows):
        rs = {r["test_fov"]: r for r in rows if r["arm"] == s["arm"] and r["split"] == s["split"]}
        cells = []
        for f in test_fovs:
            r = rs.get(f)
            cells.append("–" if r is None else
                         f"{_fmt(r['pearson_r'])} [{_fmt(r.get('pearson_r_ci_lo'))}, {_fmt(r.get('pearson_r_ci_hi'))}]")
        secs = max(r["fit_seconds"] for r in rs.values())
        lines.append(f"| {s['arm']} | " + " | ".join(cells) +
                     f" | {_fmt(s.get('pearson_r_mean'))} | {_fmt(s.get('mse_mean'))} | {_fmt(s.get('r2_mean'))} | "
                     f"{_fmt(s.get('traj_median_r_mean'))} | {secs:.0f} |")
    clf_rows = [r for r in rows if r.get("clf_accuracy") is not None]
    if clf_rows:
        lines += ["", "## Stage-1 classifier on the test FOVs", "",
                  "| arm | test FOV | accuracy | majority-class accuracy | balanced accuracy | macro F1 |",
                  "|---|---|---|---|---|---|"]
        seen = set()
        for r in clf_rows:
            key = (r["split"], r["test_fov"], r.get("info_classifier_epochs"), r["arm"].endswith("groupedval"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {r['arm']} | {r['test_fov']} | {_fmt(r['clf_accuracy'])} | "
                         f"{_fmt(r['clf_majority_class_accuracy'])} | {_fmt(r['clf_balanced_accuracy'])} | "
                         f"{_fmt(r['clf_f1_macro'])} |")
    lines += ["", "## Diagnostics", ""]
    for split, sd in diagnostics.get("splits", {}).items():
        kt = sd["keras_tail_validation"]
        lines.append(f"- **{split}**, notebook validation slice: {kt['frac_validation_rows_that_are_duplicates']:.0%} "
                     f"of validation rows are oversampled duplicates; class mix {kt['validation_class_fractions']}.")
        opt = sd.get("split_optimism_rf_vae")
        if opt:
            lines.append("- **" + split + "**, forest R by validation scheme: " +
                         ", ".join(f"{k} {v:.3f}" for k, v in opt.items()) + ".")
        td = sd["train_label_distribution"]
        lines.append(f"- **{split}**, trimmed training labels: mean {td['mean']:.3f}, class fractions "
                     f"{td['class_fractions']}.")
    for f, d in diagnostics.get("label_distribution", {}).items():
        ov = diagnostics.get("trajectory_overlap", {}).get(f, {})
        lines.append(f"- {f}: n {d['n']}, mean N/C {d['mean']:.3f}, class fractions {d['class_fractions']}, "
                     f"cells in >1 trajectory {ov.get('frac_cells_in_more_than_one_trajectory', float('nan')):.1%}.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(cfg.output_dir, "report.md"), "w") as fh:
        fh.write(text)
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ccpredict.benchmark", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="benchmark JSON (see --write-config)")
    ap.add_argument("--write-config", metavar="PATH", help="write the default benchmark config and exit")
    ap.add_argument("--output-dir")
    ap.add_argument("--arms", help="comma-separated subset of arms")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args(argv)

    if args.write_config:
        BenchmarkConfig().to_json(args.write_config)
        print(f"wrote {args.write_config}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import matplotlib

    matplotlib.use("Agg")
    cfg = BenchmarkConfig.from_json(args.config) if args.config else BenchmarkConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.arms:
        cfg.arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.quick:
        cfg = cfg.quick()
    for a in cfg.arms:
        make_arm(a)  # fail early on a typo
    run_benchmark(cfg)
    print(open(os.path.join(cfg.output_dir, "report.md")).read())


if __name__ == "__main__":
    main()
