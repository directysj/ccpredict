"""Metrics, splits, diagnostics, simulator and a CPU-only benchmark smoke test.

None of these need TensorFlow. Run with: pytest tests/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccpredict import diagnostics as diag  # noqa: E402
from ccpredict.benchmark import BenchmarkConfig, column_selector, make_block, make_splits  # noqa: E402
from ccpredict.evaluation import (classification_metrics, per_trajectory_r,  # noqa: E402
                                  regression_report, trajectory_bootstrap)


# ---------------------------------------------------------------- metrics
def _toy(seed=0, n_traj=40, length=30):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_traj), length)
    y = np.exp(rng.normal(0.3, 0.3, groups.size))
    yhat = y * np.exp(rng.normal(0, 0.2, groups.size))
    return y, yhat, groups


def test_per_trajectory_r_matches_direct_computation():
    y, yhat, g = _toy()
    g = np.random.default_rng(1).permutation(np.unique(g))[g]  # unsorted group labels
    r = per_trajectory_r(y, yhat, g, min_len=20)
    direct = sorted(np.corrcoef(y[g == k], yhat[g == k])[0, 1] for k in np.unique(g))
    assert np.allclose(sorted(r), direct)
    assert per_trajectory_r(y, yhat, g, min_len=31).size == 0


def test_bootstrap_interval_brackets_the_estimate():
    y, yhat, g = _toy()
    ci = trajectory_bootstrap(y, yhat, g, n_boot=300, seed=0)
    r = np.corrcoef(y, yhat)[0, 1]
    mse = np.mean((y - yhat) ** 2)
    assert ci["pearson_r"][0] < r < ci["pearson_r"][1]
    assert ci["mse"][0] < mse < ci["mse"][1]
    assert ci["n_trajectories"] == 40


def test_trajectory_bootstrap_is_wider_than_row_bootstrap_for_correlated_rows():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 50)
    offset = rng.normal(0, 1, 30)[g]           # shared within a trajectory
    y = 1.5 + 0.3 * offset + 0.05 * rng.standard_normal(g.size)
    yhat = 1.5 + 0.3 * offset * rng.choice([1, -1], 30)[g] + 0.05 * rng.standard_normal(g.size)
    traj = trajectory_bootstrap(y, yhat, g, n_boot=300, seed=0)["pearson_r"]
    rows = trajectory_bootstrap(y, yhat, np.arange(g.size), n_boot=300, seed=0)["pearson_r"]
    assert traj[1] - traj[0] > 3 * (rows[1] - rows[0])


def test_regression_report_handles_constant_prediction():
    y, _, g = _toy()
    rep = regression_report(y, np.full(y.size, y.mean()), g, n_boot=20)
    assert np.isnan(rep["pearson_r"]) and np.isnan(rep["spearman"])
    assert rep["mse"] > 0 and "per_trajectory" not in rep


def test_regression_report_keys_and_log_metrics():
    y, yhat, g = _toy()
    rep = regression_report(y, yhat, g, n_boot=20)
    for k in ("pearson_r", "mse", "r2", "spearman", "mae_log", "r2_log", "calibration_slope", "ci95",
              "per_trajectory"):
        assert k in rep
    assert abs(rep["mae_log"] - np.mean(np.abs(np.log(y) - np.log(yhat)))) < 1e-12


def test_classification_metrics_majority_baseline():
    y_true = np.array([1] * 95 + [0] * 3 + [2] * 2)
    y_pred = np.ones(100, dtype=int)
    m = classification_metrics(y_true, y_pred, 3)
    assert m["accuracy"] == m["majority_class_accuracy"] == 0.95
    assert abs(m["balanced_accuracy"] - 1 / 3) < 1e-12


# ---------------------------------------------------------------- splits and blocks
FOVS = ["A2_1", "A2_2", "A2_3", "B2_1", "B2_3", "B2_4"]


def test_cross_replicate_and_lofo_splits_are_disjoint():
    cfg = BenchmarkConfig()
    (name, train, test), = make_splits(cfg, FOVS)
    assert train == ("A2_1", "A2_2", "A2_3") and test == ("B2_1", "B2_3", "B2_4")
    cfg.scheme = "lofo"
    splits = make_splits(cfg, FOVS)
    assert len(splits) == 6
    for _, tr, te in splits:
        assert not set(tr) & set(te) and len(tr) == 5
    cfg.scheme = "cross_replicate"
    assert make_splits(cfg, ["A2_1", "B2_4"])[0][1:] == (("A2_1",), ("B2_4",))


def test_block_groups_unique_across_fovs():
    fov_data = {f: {"X": np.zeros((5, 2)), "y": np.ones(5), "traj": np.arange(5)} for f in ("A", "B")}
    b = make_block(fov_data, ("A", "B"))
    assert np.unique(b.groups).size == 10 and b.fold.tolist() == [0] * 5 + [1] * 5


def test_column_selector_layout():
    cols = column_selector({"trajl": 2, "n_vae": 3, "n_classical": 2})
    assert cols("vae").tolist() == [0, 1, 2, 5, 6, 7]
    assert cols("classical").tolist() == [3, 4, 8, 9]
    assert cols("vae+classical").tolist() == list(range(10))


# ---------------------------------------------------------------- diagnostics
def test_keras_tail_validation_is_last_class_duplicates():
    y = np.array([1] * 1000 + [0] * 30 + [2] * 20)
    out = diag.keras_tail_validation(y, "random", "not majority", 42, 0.2)
    assert out["n_after_oversampling"] == 3000
    assert out["frac_validation_rows_that_are_duplicates"] == 1.0
    assert out["validation_class_fractions"] == [0.0, 0.0, 1.0]


def test_trajectory_overlap():
    assert diag.trajectory_overlap([np.arange(5), np.arange(5, 9)])["frac_cells_in_more_than_one_trajectory"] == 0
    assert diag.trajectory_overlap([np.arange(5), np.arange(3, 9)])["frac_cells_in_more_than_one_trajectory"] == 2 / 9


# ---------------------------------------------------------------- grouped validation guard
def test_grouped_holdout_requires_groups():
    pytest.importorskip("keras")
    from ccpredict.config import ClassifierConfig
    from ccpredict.models import StageOneClassifier

    cfg = ClassifierConfig(validation="grouped_holdout", epochs=1, verbose=0)
    with pytest.raises(ValueError, match="groups"):
        StageOneClassifier(cfg, 3).fit(np.random.rand(30, 8), np.repeat([0, 1, 2], 10))


# ---------------------------------------------------------------- compute device
def test_classifier_runs_on_cpu_by_default():
    from ccpredict.config import ClassifierConfig

    assert ClassifierConfig().device == "cpu"


def test_configure_tensorflow():
    tf = pytest.importorskip("tensorflow")
    from ccpredict.compute import configure_tensorflow

    with pytest.raises(ValueError):
        configure_tensorflow("tpu")
    assert "CPU" in configure_tensorflow("cpu")
    assert not tf.config.get_visible_devices("GPU")


def test_rfecv_runs_folds_serially_and_trees_in_parallel():
    from ccpredict.config import RegressorConfig
    from ccpredict.models import StageTwoRegressor

    reg = StageTwoRegressor(RegressorConfig())
    assert reg.rfecv.n_jobs == 1 and reg.rfecv.estimator.n_jobs == -1


# ---------------------------------------------------------------- simulator
@pytest.fixture(scope="module")
def tiny_sim(tmp_path_factory):
    from ccpredict.simulate import SimConfig, simulate, write_dataset

    out = str(tmp_path_factory.mktemp("sim"))
    cfg = SimConfig(scale=0.05, seed=3)
    sims = simulate(cfg)
    write_dataset(sims, cfg, out)
    return out, cfg, sims


def test_simulator_matches_target_distribution(tiny_sim):
    _, _, sims = tiny_sim
    nc = np.concatenate([s.nc_obs for s in sims.values()])
    q5, q50, q95 = np.nanpercentile(nc, [5, 50, 95])
    assert abs(q50 - 1.43) < 0.08 and abs(q5 - 1.02) < 0.08 and abs(q95 - 2.28) < 0.2


def test_simulator_trajectories_partition_detections(tiny_sim):
    _, _, sims = tiny_sim
    for s in sims.values():
        cells = np.concatenate(s.trajectories)
        assert np.array_equal(np.sort(cells), np.arange(s.cell_frame.size))
        for t in s.trajectories:
            assert np.all(np.diff(s.cell_frame[t]) > 0)  # forward in time


def test_simulator_reporter_file_aligned(tiny_sim):
    out, _, sims = tiny_sim
    from ccpredict.data import read_reporter_file

    rep = read_reporter_file(os.path.join(out, "reporter", "LI204601_G_B2_4.dat"))
    s = sims["B2_4"]
    assert len(rep) == len(s.trajectories)
    k = int(np.argmax([len(t) for t in s.trajectories]))
    t = s.trajectories[k]
    ok = ~np.isnan(s.nc_obs[t])
    assert np.array_equal(rep.frames[k], s.cell_frame[t][ok].astype(int))
    assert np.allclose(rep.n2c[k], s.nc_obs[t][ok])


# ---------------------------------------------------------------- benchmark smoke test (no TF)
def test_benchmark_runs_cpu_arms(tiny_sim, tmp_path):
    from ccpredict.benchmark import run_benchmark

    out, _, _ = tiny_sim
    cfg = BenchmarkConfig().quick()
    p = cfg.pipeline.paths
    p.traj_dir, p.vae_dir, p.reporter_dir, p.celltraj_path = out, os.path.join(out, "o2vae"), \
        os.path.join(out, "reporter"), out
    cfg.arms = ("constant", "lasso_classical", "lasso_vae", "rf_vae", "rf_vae_insample")
    cfg.output_dir = str(tmp_path / "bench")
    res = run_benchmark(cfg)
    arms = {r["arm"] for r in res["rows"]}
    assert arms == set(cfg.arms)
    assert {r["test_fov"] for r in res["rows"]} == {"B2_1", "B2_3", "B2_4"}
    for f in ("report.md", "summary.csv", "results.csv", "diagnostics.json", "config.json"):
        assert os.path.exists(os.path.join(cfg.output_dir, f))
    # A second run must reuse the cache and give the same numbers (forests with
    # n_jobs > 1 sum trees in thread order, so allow last-bit differences).
    res2 = run_benchmark(cfg)
    a = {(r["arm"], r["test_fov"]): r["mse"] for r in res["rows"]}
    b = {(r["arm"], r["test_fov"]): r["mse"] for r in res2["rows"]}
    assert a.keys() == b.keys()
    assert all(abs(a[k] - b[k]) <= 1e-12 * abs(a[k]) for k in a)
    # The config round-trips through JSON.
    cfg.to_json(str(tmp_path / "c.json"))
    assert BenchmarkConfig.from_json(str(tmp_path / "c.json")).to_dict() == cfg.to_dict()
