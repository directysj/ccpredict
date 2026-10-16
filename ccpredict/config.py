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
    # NOTEBOOK DEFECT: range(traj_len - seg_length) drops the last window of
    # every trajectory (and every length-trajl trajectory entirely).
    include_last_window: bool = False
    # NOTEBOOK DEFECT: labels are read at frame(start) + trajl, one frame past
    # the snippet's last cell. "snippet_end" reads the frame of the last cell.
    label_frame: Literal["start_plus_trajl", "snippet_end"] = "start_plus_trajl"


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
    """Stage 1: CNN+DNN classifier of binned N/C ratio (In [25], [27], [32]-[37])."""

    class_boundaries: Tuple[float, ...] = (0.9, 2.7)
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
    # NOTEBOOK DEFECT: "keras_tail" oversamples first and then lets Keras take
    # the last 20% as validation. Oversampled duplicates are appended at the
    # end, grouped by class, so the validation set is copies of minority-class
    # training cells. "grouped_holdout" holds out whole trajectories first and
    # oversamples only the training part.
    validation: Literal["keras_tail", "grouped_holdout"] = "keras_tail"
    # The notebook sets no seed, so its training is not reproducible. None keeps that.
    seed: Optional[int] = None
    # Where the CNN runs. "cpu" hides every GPU from TensorFlow (CUDA and Apple
    # Metal alike); "gpu" requires one and fails if none is visible; "auto" lets
    # TensorFlow choose. Default "cpu" while developing on a laptop; switch to
    # "auto" (or "gpu") for runs on a GPU node.
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    # TensorFlow CPU thread pools; None = TensorFlow's default (all cores).
    tf_intra_op_threads: Optional[int] = None
    tf_inter_op_threads: Optional[int] = None
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
    # NOTEBOOK DEFECT: the regressor is fitted on the test FOV and then
    # evaluated on the same rows, so its MSE/R/R^2 are training-set numbers.
    # "train" fits on the trimmed training FOVs using the classifier's in-sample
    # probabilities (over-confident); "train_crossfit" uses out-of-fold
    # probabilities, one classifier per held-out training FOV (proper stacking).
    fit_on: Literal["test", "train", "train_crossfit"] = "test"
    # False replaces RFECV by a single forest on all features (much faster).
    use_rfecv: bool = True


@dataclass
class PipelineConfig:
    fovs: Tuple[str, ...] = ("A2_1", "A2_2", "A2_3", "B2_1", "B2_3", "B2_4")
    paths: PathsConfig = field(default_factory=PathsConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)

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


def _from_dict(cls, d: dict):
    """Build a (nested) dataclass from a dict, turning JSON lists back into tuples."""
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
