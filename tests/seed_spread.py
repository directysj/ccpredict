"""How much of an arm's R is the seed? Run one or more arms over several seeds.

    python tests/seed_spread.py BENCH_CONFIG.json OUT_DIR [--arms nn_mlp,nn_mlp_hist] [--seeds 5]

A benchmark run fits each arm once, so two arms differing by 0.02 in R may
differ only by their initialisation and batch order. This refits the same arms
on the same split with seeds 0..n-1 and reports mean +- sd of Pearson R per test
FOV, so a difference between arms can be compared with the spread within one.

Only the ``nn_*`` arms and the seed-dependent parts of the published arms are
affected; the LASSO and forest baselines are deterministic. Uses the
benchmark's own cached prepared data, split and scaling, so the numbers are
comparable with a benchmark report's.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

log = logging.getLogger("seed_spread")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--arms", default="nn_mlp")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from ccpredict.arms import SplitContext, make_arm
    from ccpredict.benchmark import (BenchmarkConfig, column_selector, load_data, make_block,
                                     make_splits)

    cfg = BenchmarkConfig.from_json(args.config)
    fov_data, meta = load_data(cfg)
    split_name, train_fovs, test_fovs = make_splits(cfg, meta["good_fovs"])[0]
    s = cfg.pipeline.split
    trim = (s.trim_bins, s.trim_target_density, s.trim_seed) if cfg.trim_training else None
    train = make_block(fov_data, train_fovs, trim)
    tests = {f: make_block(fov_data, (f,)) for f in test_fovs}
    columns = column_selector(meta)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    path = os.path.join(args.out_dir, "seed_spread.csv")
    for name in args.arms.split(","):
        for seed in range(args.seeds):
            net = copy.deepcopy(cfg.pipeline.net)
            net.seed = seed
            clf = copy.deepcopy(cfg.pipeline.classifier)
            clf.seed = seed
            # A fresh context per run: the cache holds the scaler, not the models.
            ctx = SplitContext(split_name, train, tests, columns, clf, cfg.pipeline.regressor,
                               net, trajl=cfg.pipeline.features.trajl)
            t0 = time.time()
            out = make_arm(name).run(ctx)
            for fov, res in out.items():
                r = float(np.corrcoef(res.y_pred, tests[fov].y)[0, 1])
                rows.append({"arm": name, "seed": seed, "test_fov": fov, "pearson_r": r,
                             "seconds": round(time.time() - t0, 1)})
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["arm", "seed", "test_fov", "pearson_r", "seconds"])
                w.writeheader()
                w.writerows(rows)
            log.info("%s seed %d: %s (%.0fs)", name, seed,
                     "  ".join(f"{r['test_fov']} {r['pearson_r']:.3f}"
                               for r in rows if r["arm"] == name and r["seed"] == seed),
                     time.time() - t0)

    lines = [f"# Seed spread ({split_name})", "",
             f"Seeds 0-{args.seeds - 1}, train {list(train_fovs)}, test {list(test_fovs)}.", "",
             "| arm | " + " | ".join(f"R {f}" for f in test_fovs) + " | mean R | s/run |",
             "|---|" + "---|" * (len(test_fovs) + 2)]
    for name in args.arms.split(","):
        mine = [r for r in rows if r["arm"] == name]
        cells, means = [], []
        for f in test_fovs:
            v = [r["pearson_r"] for r in mine if r["test_fov"] == f]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}")
            means.append(np.mean(v))
        per_seed = [np.mean([r["pearson_r"] for r in mine if r["seed"] == s])
                    for s in sorted({r["seed"] for r in mine})]
        secs = np.mean([r["seconds"] for r in mine])
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {np.mean(means):.3f} ± {np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0:.3f}"
                     + f" | {secs:.0f} |")
    report = os.path.join(args.out_dir, "seed_spread.md")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {report} and {path}")


if __name__ == "__main__":
    main()
