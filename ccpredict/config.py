"""Configuration for the cell-cycle prediction pipeline.

Every default reproduces ``vaeFeatCCinfoABwells-NNclfRFreg.ipynb`` exactly,
including its defects. Each defect that changes numbers has a switch here,
marked ``NOTEBOOK DEFECT`` in its comment; README.md explains them.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


# Paper: A wells = EGF1, B wells = EGF2, biological replicates of MCF10A + EGF.
DEFAULT_FOV_REPLICATE = {
    "A2_1": "EGF1", "A2_2": "EGF1", "A2_3": "EGF1",
    "B2_1": "EGF2", "B2_3": "EGF2", "B2_4": "EGF2",
}


@dataclass
class PathsConfig:
    """Where the inputs live. Mirrors pathSet / file_path / ../o2vae in the notebook."""

    traj_dir: str = "/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/"
    sys_name: str = "LI204601_P"
    vae_dir: str = "../o2vae"
    reporter_dir: str = "/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/cellCycInfoRawReporterImgs/"
    reporter_sys_name: str = "LI204601_G"
    # Directory holding jcTrajectory_CP.py; required to unpickle the tracked FOVs.
    celltraj_path: Optional[str] = "/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj"
    # Figures, metrics and fitted models are written here when set.
    output_dir: Optional[str] = None

    def model_file(self, fov: str) -> str:
        return os.path.join(self.traj_dir, f"{self.sys_name}_{fov}.obj")

    def vae_file(self, fov: str, latent_dim: int) -> str:
        # Notebook: f'../o2vae/features_vae{latent_dim}_{sysName[:-1]}{fovs[i]}.npz'
        return os.path.join(self.vae_dir, f"features_vae{latent_dim}_{self.sys_name[:-1]}{fov}.npz")

    def reporter_file(self, fov: str) -> str:
        return os.path.join(self.reporter_dir, f"{self.reporter_sys_name}_{fov}.dat")


@dataclass
class QCConfig:
    """Cell-count smoothness QC used to drop badly tracked FOVs (notebook In [8]-[10])."""

    n_frames: int = 193
    window_start_pct: float = 70.0
    window_end_pct: float = 98.5
    std_cut: float = 0.50
    smooth: float = 0.98
    n_smooth_points: int = 10000
    # NOTEBOOK DEFECT: the spline is evaluated on a 10,000-point grid and then
    # indexed with frame numbers 136..190, i.e. at frames ~2.6-3.7 instead of
    # 136-190. False reproduces that; True evaluates the spline on the frames.
    evaluate_on_frames: bool = False

    def window_frames(self):
        import math

        import numpy as np

        start = math.ceil(self.window_start_pct * self.n_frames / 100.0)
        end = math.ceil(self.window_end_pct * self.n_frames / 100.0)
        return np.arange(start, end).astype(int)


@dataclass
class FeatureConfig:
    """How per-cell features are turned into per-snippet feature vectors (In [15]-[18])."""

    latent_dim: int = 256
    trajl: int = 1
    # Commented-out notebook option: append handcrafted columns Xf[:, k:]
    # (k = 77 gave the cell-cell interaction features). None = VAE only.
    handcrafted_from_col: Optional[int] = None
    # Commented-out notebook option: append centre-of-mass features Xf_com.
    append_com: bool = False
    # The notebook used False, which drops the last window of every trajectory
    # (and every length-trajl trajectory entirely). With label_frame below set
    # to the notebook's value those rows had no label anyway; with
    # "snippet_end" they do, so keeping them adds ~2% more labelled rows.
    include_last_window: bool = True
    # The ground truth N/C belongs to the LAST snapshot of the snippet.
    # "snippet_end" reads the frame of that snapshot, which is the intended
    # semantics. The notebook read frame(start) + trajl, one frame (15 min)
    # past the window, so it fitted labels shifted into the future; that is a
    # defect, not a design choice, and costs 0.03-0.05 Pearson R on simulated
    # data. "start_plus_trajl" reproduces the notebook's off-by-one.
    label_frame: Literal["start_plus_trajl", "snippet_end"] = "snippet_end"


@dataclass
class SplitConfig:
    """Train/test FOVs and the histogram trimming of the training labels (In [22]-[23])."""

    train_fovs: Tuple[str, ...] = ("A2_1", "A2_2", "A2_3")
    test_fov: str = "B2_4"
    trim_bins: int = 50
    trim_target_density: int = 2000
    trim_seed: int = 42


@dataclass
class ForestConfig:
    """The random-forest baseline (`rf_vae`, `rf_classical`, and their `_insample` twins).

    The paper's stage-2 hyperparameters, kept because the forest is still the
    reference a one-stage network has to beat. No feature selection: the RFECV
    wrapper belonged to the two-stage model and went with it.
    """

    n_estimators: int = 100
    max_depth: Optional[int] = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    random_state: int = 42
    # Trees are built in parallel threads, which scales with cores.
    n_jobs: int = -1


@dataclass
class NetConfig:
    """A one-stage neural regressor of the N/C ratio: no class bins, no forest.

    This is not part of the notebook. It exists because the published pipeline
    turns a continuous, strongly skewed target into three bins of which the
    middle one holds ~97% of the rows, and then feeds the resulting class
    probabilities to a forest. Here the network predicts the ratio itself.

    The architecture switches below are deliberate departures from the
    notebook's, each isolated by its own benchmark arm:

    encoder="mlp"
        The notebook convolves a (3,) kernel along the *latent index* of the
        VAE embedding. That index has no ordering: dimension 7 is not a
        neighbour of dimension 8, so the kernel mixes three arbitrary
        dimensions and weight sharing across the 256 of them has no meaning.
        An MLP on the same features makes no such assumption.
    norm_position="before_activation"
        The notebook normalises after the activation. Normalising before it is
        the usual order and is what makes the conv bias provably redundant
        (BatchNorm subtracts the batch mean, so a per-channel constant is
        removed), hence conv_bias=False. Set "after_activation" to measure the
        notebook's order with everything else held fixed.
    regularizers
        The notebook's l2 0.01 / l1 0.02 on every layer gives a penalty of
        about 13 at initialisation against a cross-entropy of about 1.1, i.e.
        92% of the loss is the penalty. Default here is a single light l2.
    """

    # ---- the task ----
    head: Literal["regression", "histogram"] = "regression"
    # "identity" fits the raw ratio, "log" fits log(N/C) and exponentiates back.
    # The ratio is positive and right-skewed, so "log" makes the residuals more
    # nearly homoscedastic; metrics are always reported on the raw ratio.
    target_transform: Literal["identity", "log"] = "identity"
    loss: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0
    # head="histogram": cross-entropy over equal-count (quantile) bins of the
    # training labels, predicting the expectation sum_k p_k * mean(y in bin k).
    # Equal-count bins remove the class imbalance by construction; the
    # expectation keeps the prediction continuous.
    n_bins: int = 32
    # "inverse_density" weights each row by 1 / (density of its label), the
    # regression analogue of the notebook's minority-class oversampling.
    label_weighting: Literal["none", "inverse_density"] = "none"
    weight_clip: float = 10.0

    # ---- the network ----
    # See cnn.FeedForwardNet for what each encoder does: "mlp",
    # "conv" (the notebook's, along the flattened axis), "snapshot_shared",
    # "temporal_conv", "transformer".
    encoder: Literal["mlp", "conv", "snapshot_shared", "temporal_conv",
                     "transformer"] = "mlp"
    snapshot_hidden: Tuple[int, ...] = (64,)
    temporal_filters: Tuple[int, ...] = (64, 64)
    temporal_kernel: int = 3
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 128
    transformer_attention: Literal["self", "none"] = "self"
    pool: Literal["concat", "mean", "attention", "last"] = "concat"
    norm_position: Literal["before_activation", "after_activation"] = "before_activation"
    conv_bias: bool = False
    conv_filters: Tuple[int, ...] = (32, 64)
    dense_units: Tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    regularizers: Tuple[Tuple[str, float], ...] = (("l2", 1e-4),)

    # ---- training ----
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 128
    # Early stopping holds out whole trajectories ("grouped_holdout"); random
    # rows would put adjacent frames of one cell on both sides.
    validation: Literal["grouped_holdout", "random_rows"] = "grouped_holdout"
    validation_fraction: float = 0.2
    validation_seed: int = 42
    patience: int = 20
    min_delta: float = 1e-4
    # None = unseeded, as the notebook was. The benchmark fills it with
    # BenchmarkConfig.seed, so a run's seed reaches these arms too; set it here
    # to pin one arm independently of the run.
    seed: Optional[int] = None
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    num_threads: Optional[int] = None
    verbose: int = 1


@dataclass
class PipelineConfig:
    fovs: Tuple[str, ...] = ("A2_1", "A2_2", "A2_3", "B2_1", "B2_3", "B2_4")
    paths: PathsConfig = field(default_factory=PathsConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    net: NetConfig = field(default_factory=NetConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# Keys that older configs may contain: old name -> new name (None = dropped).
# The two-stage model was removed from this branch, so a config written before
# that carries a "classifier" block and stage-2 forest keys. They are dropped
# rather than erroring, so an old bench.json still runs the arms that remain.
_RENAMED_KEYS = {
    "ForestConfig": {"rfecv_n_jobs": None, "cv_splits": None, "cv_seed": None,
                     "rfecv_step": None, "min_features_to_select": None,
                     "scoring": None, "fit_on": None, "stage2_fraction": None,
                     "stage2_seed": None, "use_rfecv": None},
    "PipelineConfig": {"classifier": None, "regressor": "forest"},
}


def _from_dict(cls, d: dict):
    """Build a (nested) dataclass from a dict, turning JSON lists back into tuples."""
    d = dict(d)
    for old, new in _RENAMED_KEYS.get(cls.__name__, {}).items():
        if old in d:
            value = d.pop(old)
            if new is not None and new not in d:
                d[new] = value
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in d.items():
        default = getattr(cls(), name) if _has_defaults(cls) else None
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default), value)
        elif isinstance(value, list):
            kwargs[name] = _to_tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def _has_defaults(cls) -> bool:
    return all(
        f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        for f in dataclasses.fields(cls)
    )
