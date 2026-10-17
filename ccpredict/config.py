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
    # data. "start_plus_trajl" reproduces it and is what tests/parity pins.
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
class ClassifierConfig:
    """Stage 1: CNN+DNN classifier of binned N/C ratio (In [25], [27], [32]-[37]).

    Implemented in PyTorch with the notebook's Keras defaults reproduced
    (see cnn.py): Glorot init, BatchNorm momentum 0.99 / eps 1e-3, LeakyReLU
    slope 0.3, L1/L2 penalties added to the loss, Keras-style Adam (eps 1e-7).
    """

    class_boundaries: Tuple[float, ...] = (0.9, 2.7)
    # Which axis of the (trajl x nf) input the encoder works along; see
    # cnn.FeedForwardNet. "conv" is the notebook (a width-3 kernel along the
    # flattened vector, so almost entirely within one snapshot). The other three
    # are alternatives to compare it with, not defaults:
    #   "mlp"             no convolution, the flat vector into the dense layers
    #   "snapshot_shared" one MLP per snapshot, weights shared across time
    #   "temporal_conv"   Conv1d along time with nf input channels
    #   "transformer"     self-attention among the snapshots of the snippet
    encoder: Literal["conv", "mlp", "snapshot_shared", "temporal_conv",
                     "transformer"] = "conv"
    # The sequence encoders only; see cnn.FeedForwardNet.
    snapshot_hidden: Tuple[int, ...] = (64,)
    temporal_filters: Tuple[int, ...] = (64, 64)
    temporal_kernel: int = 3
    # encoder="transformer" only. transformer_dim must divide by transformer_heads.
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 128
    # "none" keeps the whole block (token projection, both LayerNorms, the
    # feed-forward sublayer) and makes only the attention sublayer the identity.
    # The ablation that prices attention itself rather than the block around it.
    transformer_attention: Literal["self", "none"] = "self"
    pool: Literal["concat", "mean", "attention", "last"] = "concat"
    conv_filters: Tuple[int, ...] = (32, 64)
    dense_units: Tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    # Regularisers alternate within each block: conv1 l2, conv2 l1, dense1 l2, dense2 l1.
    regularizers: Tuple[Tuple[str, float], ...] = (("l2", 0.01), ("l1", 0.02))
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 20
    validation_fraction: float = 0.2
    patience: int = 15
    min_delta: float = 1e-3
    oversampler: Literal["random", "smote", "kmeans_smote", "adasyn", "none"] = "random"
    oversampling_strategy: str = "not majority"
    oversampling_seed: int = 42
    # NOTEBOOK DEFECT: "keras_tail" oversamples first and then takes the last
    # 20% as validation, as Keras' validation_split did in the notebook.
    # Oversampled duplicates are appended at the end, grouped by class, so the
    # validation set is copies of minority-class training cells.
    # "grouped_holdout" holds out whole trajectories first and oversamples only
    # the training part.
    validation: Literal["keras_tail", "grouped_holdout"] = "keras_tail"
    # The notebook sets no seed, so its training is not reproducible. None keeps that.
    seed: Optional[int] = None
    # Where the CNN runs. "cpu"; "gpu" = CUDA, else Apple Metal (MPS), and fails
    # if neither exists; "auto" = CUDA > MPS > CPU. Default "cpu" while
    # developing on a laptop; switch to "auto" (or "gpu") for a GPU node.
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    # PyTorch CPU threads; None = PyTorch's default (all cores).
    num_threads: Optional[int] = None
    verbose: int = 1


@dataclass
class RegressorConfig:
    """Stage 2: RFECV random-forest regressor on [features, class probabilities] (In [42], [44] and the unexecuted cells after them)."""

    n_estimators: int = 100
    max_depth: Optional[int] = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    random_state: int = 42
    # Parallelism. n_jobs is used by every random forest (trees are built in
    # parallel threads, which scales with cores). rfecv_n_jobs is how many CV
    # folds RFECV runs at once; inside parallel folds each forest drops to one
    # thread, so with many cores it is faster to run folds one at a time (1)
    # and let the trees use all cores. The notebook used -1 for both; results
    # are the same either way up to float summation order.
    n_jobs: int = -1
    rfecv_n_jobs: Optional[int] = 1
    cv_splits: int = 5
    cv_seed: int = 42
    rfecv_step: int = 1
    min_features_to_select: int = 50
    scoring: str = "neg_mean_squared_error"
    # Where stage 2 gets its training rows, and whose stage-1 probabilities it
    # sees. Only the last two are honest about both at once:
    #   "test"           NOTEBOOK DEFECT: fitted on the test FOV and scored on
    #                    the same rows, so MSE/R/R^2 are training-set numbers.
    #   "train"          the trimmed training FOVs, but with stage 1's in-sample
    #                    probabilities, which are more confident than the ones it
    #                    will meet on a new FOV.
    #   "train_crossfit" out-of-fold probabilities, one stage 1 per held-out
    #                    training FOV: all the data, K extra classifier fits.
    #   "train_holdout"  one stage 1 on part of the training trajectories, stage 2
    #                    on the rest with that model's out-of-sample probabilities:
    #                    one classifier fit, less data for each stage.
    fit_on: Literal["test", "train", "train_crossfit", "train_holdout"] = "test"
    # fit_on="train_holdout": the share of training *trajectories* kept for
    # stage 2 (grouped, so no cell appears in both stages).
    stage2_fraction: float = 0.3
    stage2_seed: int = 42
    # False replaces RFECV by a single forest on all features (much faster).
    use_rfecv: bool = True


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
    # Same options as ClassifierConfig.encoder (see cnn.FeedForwardNet): "mlp",
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
    # None = unseeded, as ClassifierConfig.seed is. The benchmark fills it with
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
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    # Used only by the nn_* benchmark arms; the notebook pipeline ignores it.
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
_RENAMED_KEYS = {
    "ClassifierConfig": {"tf_intra_op_threads": "num_threads", "tf_inter_op_threads": None},
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
