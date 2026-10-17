"""Metrics, splits, diagnostics, simulator and a CPU-only benchmark smoke test.

None of these need TensorFlow (the CNN tests need PyTorch). Run with: pytest tests/
"""
from __future__ import annotations

import dataclasses
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
    pytest.importorskip("torch")
    from ccpredict.config import ClassifierConfig
    from ccpredict.models import StageOneClassifier

    cfg = ClassifierConfig(validation="grouped_holdout", epochs=1, verbose=0)
    with pytest.raises(ValueError, match="groups"):
        StageOneClassifier(cfg, 3).fit(np.random.rand(30, 8), np.repeat([0, 1, 2], 10))


# ---------------------------------------------------------------- compute device
def test_classifier_runs_on_cpu_by_default():
    from ccpredict.config import ClassifierConfig

    assert ClassifierConfig().device == "cpu"


def test_select_device():
    pytest.importorskip("torch")
    from ccpredict.compute import select_device

    with pytest.raises(ValueError):
        select_device("tpu")
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cpu", "cuda", "mps")


def test_old_tensorflow_thread_keys_still_load():
    from ccpredict.config import ClassifierConfig, _from_dict

    cfg = _from_dict(ClassifierConfig, {"tf_intra_op_threads": 4, "tf_inter_op_threads": 2})
    assert cfg.num_threads == 4


def _tiny_problem(seed=0, n=600, d=32):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.1, 0.8, 0.1])
    X = rng.random((n, d)) + 0.3 * y[:, None]
    return X, y


def _quick_cfg(**kw):
    from ccpredict.config import ClassifierConfig

    return ClassifierConfig(epochs=3, patience=2, batch_size=64, verbose=0, seed=7, **kw)


def test_cnn_regularisers_follow_the_notebook():
    pytest.importorskip("torch")
    from ccpredict.config import ClassifierConfig
    from ccpredict.models import build_cnn_classifier

    m = build_cnn_classifier(32, 3, ClassifierConfig())
    kinds = [(type(layer).__name__, kind, lam) for layer, (kind, lam) in m._penalties]
    assert kinds == [("Conv1d", "l2", 0.01), ("Conv1d", "l1", 0.02),
                     ("Linear", "l2", 0.01), ("Linear", "l1", 0.02)]
    manual = sum(lam * (layer.weight.abs().sum() if kind == "l1" else (layer.weight ** 2).sum())
                 for layer, (kind, lam) in m._penalties)
    assert abs(manual.item() - m.penalty().item()) < 1e-6


@pytest.mark.parametrize("validation", ["keras_tail", "grouped_holdout"])
def test_classifier_is_deterministic_with_a_seed(validation, tmp_path):
    pytest.importorskip("torch")
    from ccpredict.models import StageOneClassifier

    X, y = _tiny_problem()
    groups = np.arange(len(y)) // 10
    a = StageOneClassifier(_quick_cfg(validation=validation), 3).fit(X, y, groups)
    b = StageOneClassifier(_quick_cfg(validation=validation), 3).fit(X, y, groups)
    pa, pb = a.predict_proba(X), b.predict_proba(X)
    assert pa.dtype == np.float32 and pa.shape == (len(y), 3)
    assert np.allclose(pa.sum(1), 1, atol=1e-5)
    assert np.array_equal(pa, pb)
    assert {"loss", "val_loss", "accuracy", "val_accuracy", "best_epoch"} <= set(a.history)

    a.save(str(tmp_path))
    c = StageOneClassifier.load(str(tmp_path), _quick_cfg(validation=validation))
    assert np.array_equal(pa, c.predict_proba(X))


# ---------------------------------------------------------------- one-stage neural regressor
def _ratio_problem(seed=0, n=800, d=32):
    """Positive, right-skewed target with a signal in the first few features."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.exp(0.3 * (X[:, :4].sum(1) - X[:, 4:8].sum(1)) + rng.normal(0, 0.15, n)) + 0.6
    return X, y, np.arange(n) // 10


def _net_cfg(**kw):
    from ccpredict.config import NetConfig

    return NetConfig(**{"epochs": 4, "patience": 2, "batch_size": 64, "verbose": 0, "seed": 5, **kw})


def test_net_config_defaults_depart_from_the_notebook_deliberately():
    from ccpredict.config import ClassifierConfig, NetConfig

    cfg, notebook = NetConfig(), ClassifierConfig()
    assert cfg.encoder == "mlp"                       # no convolution along the latent index
    assert cfg.norm_position == "before_activation" and cfg.conv_bias is False
    assert cfg.validation == "grouped_holdout"        # not the oversampled tail
    # Much lighter than the notebook's l2 0.01 / l1 0.02.
    assert max(lam for _, lam in cfg.regularizers) < max(lam for _, lam in notebook.regularizers) / 10


def test_pipeline_config_without_a_net_section_still_loads():
    from ccpredict.config import NetConfig, PipelineConfig

    cfg = PipelineConfig.from_dict({"fovs": ["A2_1"], "classifier": {"epochs": 2}})
    assert cfg.net == NetConfig() and cfg.classifier.epochs == 2


def test_benchmark_seed_reaches_the_nn_arms():
    """NetConfig.seed defaults to None so a run's seed is not silently ignored
    (it used to default to 0, which no BenchmarkConfig.seed could override)."""
    from ccpredict.benchmark import BenchmarkConfig
    from ccpredict.config import NetConfig

    assert NetConfig().seed is None
    cfg = BenchmarkConfig(seed=7)
    net = cfg.pipeline.net
    assert (net.seed if net.seed is not None else cfg.seed) == 7
    # An explicit seed still wins over the run's.
    cfg.pipeline.net.seed = 3
    assert (cfg.pipeline.net.seed if cfg.pipeline.net.seed is not None else cfg.seed) == 3


def test_metrics_never_ask_the_device_for_float64():
    """Apple's MPS backend has no float64, so a `.double()` anywhere in the
    training path kills every run with device='auto'/'gpu' on a Mac. Metrics are
    computed from numpy on the host instead; this forbids Tensor.double outright
    so the regression is caught on a CPU-only machine too."""
    torch = pytest.importorskip("torch")
    from ccpredict.models import DirectRegressor

    X, y, groups = _ratio_problem(n=200)
    original = torch.Tensor.double

    def refuse(self, *a, **k):
        raise AssertionError("Tensor.double() in the training path: MPS has no float64")

    torch.Tensor.double = refuse
    try:
        m = DirectRegressor(_net_cfg(head="histogram", n_bins=4)).fit(X, y, groups)
        assert np.all(np.isfinite(m.predict(X)))
        m = DirectRegressor(_net_cfg()).fit(X, y, groups)
        assert np.all(np.isfinite(m.predict(X)))
        assert all(np.isfinite(m.history["pearson_r"]))
    finally:
        torch.Tensor.double = original


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_fit_runs_on_an_accelerator_when_present(device):
    """The real check for the above, on whatever accelerator this machine has."""
    torch = pytest.importorskip("torch")
    available = {"mps": getattr(torch.backends, "mps", None) is not None
                 and torch.backends.mps.is_available(),
                 "cuda": torch.cuda.is_available()}
    if not available[device]:
        pytest.skip(f"no {device} on this machine")
    from ccpredict.models import DirectRegressor

    X, y, groups = _ratio_problem(n=300)
    for over in ({}, {"head": "histogram", "n_bins": 8}, {"encoder": "conv"}):
        m = DirectRegressor(_net_cfg(device="auto", **over)).fit(X, y, groups)
        assert m.device.type == device
        assert np.all(np.isfinite(m.predict(X)))


def test_no_regularisers_means_no_penalty():
    """regularizers=() is the natural way to say 'no penalty'; it used to raise
    ZeroDivisionError from the k % len(regs) rule."""
    pytest.importorskip("torch")
    from ccpredict.models import build_net

    net = build_net(32, 1, _net_cfg(regularizers=()))
    assert net._penalties == []
    assert net.penalty().item() == 0.0


def test_mlp_encoder_has_no_convolutions():
    pytest.importorskip("torch")
    from ccpredict.models import build_net

    mlp = build_net(32, 1, _net_cfg())
    assert len(mlp.convs) == 0 and len(mlp.norms) == 0
    assert mlp.denses[0].weight.shape[1] == 32          # features go straight to the dense stack
    conv = build_net(32, 1, _net_cfg(encoder="conv"))
    assert len(conv.convs) == 2 and all(c.bias is None for c in conv.convs)


def test_conv_bias_is_dead_when_batchnorm_comes_first():
    """Why conv_bias=False goes with norm_position='before_activation': BatchNorm
    subtracts the batch mean, so a per-channel constant added before it cannot
    change any output. With the notebook's order the activation sits in between,
    so the bias does something (except for filters that never leave the linear
    branch -- see tests/parity/test_torch_port.py)."""
    torch = pytest.importorskip("torch")
    from ccpredict.models import build_net

    X = torch.rand(64, 32)
    ratios = {}
    for position in ("before_activation", "after_activation"):
        net = build_net(32, 1, _net_cfg(encoder="conv", norm_position=position, conv_bias=True))
        net.train()
        net(X).square().mean().backward()
        conv = net.convs[0]
        # Against the weight gradients of the same layer, so this is scale-free.
        ratios[position] = (float(conv.bias.grad.abs().max())
                            / float(conv.weight.grad.abs().max()))
    assert ratios["before_activation"] < 1e-4        # float noise around an exact zero
    assert ratios["after_activation"] > 1e-2


def test_training_rejects_zero_epochs():
    """It used to run no epochs and then crash in load_state_dict(None)."""
    torch = pytest.importorskip("torch")
    from ccpredict.cnn import FeedForwardNet, train

    with pytest.raises(ValueError, match="epochs"):
        train(FeedForwardNet(8, 3, encoder="mlp"), np.zeros((4, 8)), np.zeros(4, int),
              np.zeros((4, 8)), np.zeros(4, int), epochs=0, batch_size=2, learning_rate=1e-3,
              patience=1, min_delta=0.0, device=torch.device("cpu"),
              rng=np.random.default_rng(0), verbose=0)


def test_calibration_slope_is_one_for_perfect_predictions():
    """Slope = cov(y, p) / var(p); both parts need the same ddof."""
    rng = np.random.default_rng(0)
    y = rng.normal(1.5, 0.3, 50)
    assert abs(regression_report(y, y.copy(), n_boot=5)["calibration_slope"] - 1.0) < 1e-12
    # Halving the spread of the predictions must double the slope, exactly.
    p = y.mean() + 0.5 * (y - y.mean())
    assert abs(regression_report(y, p, n_boot=5)["calibration_slope"] - 2.0) < 1e-12


def test_histogram_head_predicts_the_expected_label():
    torch = pytest.importorskip("torch")
    from ccpredict.cnn import HistogramObjective

    y = np.concatenate([np.full(100, 1.0), np.full(100, 2.0), np.full(100, 3.0)])
    obj = HistogramObjective.from_labels(y, n_bins=3)
    assert obj.out_features == 3
    assert np.allclose(obj.centers, [1.0, 2.0, 3.0])
    assert np.array_equal(obj.encode(np.array([1.0, 2.0, 3.0])), [0, 1, 2])
    # A confident logit returns that bin's mean label; a flat one returns the overall mean.
    confident = torch.tensor([[20.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
    assert np.allclose(obj.predict(confident), [1.0, 3.0], atol=1e-6)
    assert np.allclose(obj.predict(torch.zeros(1, 3)), [2.0], atol=1e-6)


def test_log_target_round_trips_and_rejects_non_positive_labels():
    pytest.importorskip("torch")
    from ccpredict.cnn import RegressionObjective

    obj = RegressionObjective("log")
    y = np.array([0.5, 1.0, 2.5])
    assert np.allclose(obj.decode(obj.encode(y)), y, atol=1e-6)
    with pytest.raises(ValueError):
        obj.encode(np.array([1.0, 0.0]))


def test_inverse_density_weights_favour_rare_labels():
    pytest.importorskip("torch")
    from ccpredict.cnn import inverse_density_weights

    y = np.concatenate([np.random.default_rng(0).normal(1.5, 0.05, 2000), [3.0, 3.1]])
    w = inverse_density_weights(y, clip=10.0)
    assert w.max() <= 10.0                            # nothing counts for more than 10 average rows
    assert w[-1] == 10.0 and w[-2] == 10.0            # the two rare high ratios are the ones clipped
    assert w[:2000].mean() < 1.0                      # the bulk is held below average weight
    # Within the bulk, rarer labels still get more weight than the mode.
    mode = np.argmin(np.abs(y[:2000] - 1.5))
    assert w[np.argmax(np.abs(y[:2000] - 1.5))] > w[mode]


@pytest.mark.parametrize("over", [{}, {"target_transform": "log"}, {"loss": "huber"},
                                  {"head": "histogram", "n_bins": 8},
                                  {"label_weighting": "inverse_density"},
                                  {"encoder": "conv"}])
def test_direct_regressor_is_deterministic_and_round_trips(over, tmp_path):
    pytest.importorskip("torch")
    from ccpredict.models import DirectRegressor

    X, y, groups = _ratio_problem()
    a = DirectRegressor(_net_cfg(**over)).fit(X, y, groups)
    b = DirectRegressor(_net_cfg(**over)).fit(X, y, groups)
    pa = a.predict(X)
    assert pa.shape == (len(y),)
    assert np.array_equal(pa, b.predict(X))
    assert np.all(np.isfinite(pa)) and pa.min() > 0
    assert {"loss", "val_loss", "best_epoch"} <= set(a.history)

    a.save(str(tmp_path))
    c = DirectRegressor.load(str(tmp_path))
    assert np.array_equal(pa, c.predict(X))
    assert c.cfg == a.cfg


def test_direct_regressor_starts_at_the_training_mean():
    """A regression head's output bias is the mean target, so an untrained net
    predicts the constant baseline instead of drifting towards it."""
    pytest.importorskip("torch")
    from ccpredict.models import DirectRegressor

    X, y, groups = _ratio_problem()
    m = DirectRegressor(_net_cfg(epochs=1, dropout=0.0)).fit(X, y, groups)
    assert abs(float(m.model.out.bias.item()) - y.mean()) < 0.35


def test_direct_regressor_beats_the_constant_baseline():
    pytest.importorskip("torch")
    from ccpredict.models import DirectRegressor

    X, y, groups = _ratio_problem()
    held = np.arange(len(y)) % 4 == 0
    m = DirectRegressor(_net_cfg(epochs=40, patience=10)).fit(X[~held], y[~held], groups[~held])
    pred = m.predict(X[held])
    mse = float(np.mean((pred - y[held]) ** 2))
    constant = float(np.mean((y[~held].mean() - y[held]) ** 2))
    assert mse < 0.7 * constant
    assert np.corrcoef(pred, y[held])[0, 1] > 0.5


def test_grouped_validation_requires_groups_but_random_rows_does_not():
    pytest.importorskip("torch")
    from ccpredict.models import DirectRegressor

    X, y, _ = _ratio_problem(n=200)
    with pytest.raises(ValueError, match="groups"):
        DirectRegressor(_net_cfg()).fit(X, y)
    DirectRegressor(_net_cfg(validation="random_rows")).fit(X, y)


def test_neural_arms_differ_only_by_their_overrides():
    from ccpredict.arms import NEURAL_ARMS, SplitContext, make_arm
    from ccpredict.config import ClassifierConfig, NetConfig, RegressorConfig

    ctx = SplitContext("s", None, {}, lambda f: None, ClassifierConfig(), RegressorConfig(),
                       NetConfig())
    base = NetConfig()
    for name, overrides in NEURAL_ARMS.items():
        cfg = make_arm(name).cfg(ctx)
        changed = {f.name for f in dataclasses.fields(cfg)
                   if getattr(cfg, f.name) != getattr(base, f.name)}
        assert changed == set(overrides), name
    with pytest.raises(KeyError):
        from ccpredict.arms import NeuralArm

        NeuralArm("bogus", no_such_field=1).cfg(ctx)


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


# ---------------------------------------------------------------- benchmark smoke test
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
