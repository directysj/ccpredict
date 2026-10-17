"""Stage 1 (CNN+DNN classifier) and stage 2 (RFECV random forest regressor).

Notebook In [27], [33]-[37] (stage 1) and In [42], [44] plus the unexecuted fit cells (stage 2).

``DirectRegressor`` is not from the notebook: it replaces both stages with one
network that predicts the N/C ratio itself (see NetConfig in config.py).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from typing import Optional

import numpy as np

from .config import ClassifierConfig, NetConfig, RegressorConfig
from .preprocessing import make_oversampler

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Stage 1 (PyTorch; network and training loop in cnn.py)
# ----------------------------------------------------------------------------

def build_cnn_classifier(input_dim: int, num_classes: int, cfg: ClassifierConfig,
                         steps: int = 1):
    """Stage 1 as a PyTorch module (see cnn.FeedForwardNet for every Keras equivalence).

    With cfg.encoder="conv" (the notebook) the input (d,) is treated as a
    length-d sequence, so the (3,)-kernel convolution runs along the feature
    index; for VAE latents that index has no ordering, so "neighbouring"
    dimensions are arbitrary. ``steps`` is the snippet length (trajl), used by
    the encoders that read the input as a sequence of snapshots.
    """
    from .cnn import FeedForwardNet

    return FeedForwardNet(input_dim, num_classes, cfg.conv_filters, cfg.dense_units,
                          cfg.dropout, cfg.regularizers, encoder=cfg.encoder, steps=steps,
                          snapshot_hidden=cfg.snapshot_hidden,
                          temporal_filters=cfg.temporal_filters,
                          temporal_kernel=cfg.temporal_kernel, pool=cfg.pool)


class StageOneClassifier:
    """Oversample, train the CNN+DNN with early stopping, predict class probabilities."""

    def __init__(self, cfg: ClassifierConfig, num_classes: int, steps: int = 1):
        self.cfg = cfg
        self.num_classes = num_classes
        self.steps = steps          # snippet length (trajl); only the sequence encoders use it
        self.model = None
        self.device = None
        self._fitted_input_dim: Optional[int] = None
        self.history: dict = {}

    def _oversample(self, X, y):
        sampler = make_oversampler(self.cfg.oversampler, self.cfg.oversampling_strategy,
                                   self.cfg.oversampling_seed)
        return (X, y) if sampler is None else sampler.fit_resample(X, y)

    def _configure_device(self) -> None:
        from .compute import select_device

        self.device = select_device(self.cfg.device, self.cfg.num_threads)

    def fit(self, X: np.ndarray, y_class: np.ndarray,
            groups: Optional[np.ndarray] = None) -> "StageOneClassifier":
        """Train on (X, y_class). ``groups`` (trajectory ids) is required for grouped_holdout."""
        import torch

        from .cnn import train

        cfg = self.cfg
        self._configure_device()
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        if cfg.validation == "keras_tail":
            # Notebook behaviour: oversample, then take the LAST validation_fraction
            # of the arrays as Keras' validation_split did. The oversampler
            # appends its duplicates at the end, grouped by class, so validation
            # = copies of minority-class training rows; val_loss then drives
            # early stopping and the restored weights.
            log.warning("validation='keras_tail': validation rows are oversampled duplicates "
                        "of training rows (notebook behaviour)")
            X_res, y_res = self._oversample(X, y_class)
            split_at = int(math.floor(len(y_res) * (1.0 - cfg.validation_fraction)))
            X_tr, y_tr, X_val, y_val = X_res[:split_at], y_res[:split_at], X_res[split_at:], y_res[split_at:]
        elif cfg.validation == "grouped_holdout":
            # Hold out whole trajectories, then oversample only the training part.
            # Splitting rows at random would put adjacent, near-identical frames
            # of one cell on both sides and make val_loss optimistic.
            if groups is None:
                raise ValueError("validation='grouped_holdout' needs groups (trajectory ids per row)")
            from sklearn.model_selection import GroupShuffleSplit

            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=cfg.validation_fraction,
                                            random_state=cfg.oversampling_seed).split(X, y_class, groups))
            X_tr, y_tr = self._oversample(X[tr], y_class[tr])
            X_val, y_val = X[va], y_class[va]
        else:
            raise ValueError(f"unknown validation {cfg.validation!r}")

        self._fitted_input_dim = int(X.shape[1])
        self.model = build_cnn_classifier(X.shape[1], self.num_classes, cfg, self.steps)
        self.history = train(self.model, X_tr, y_tr, X_val, y_val, epochs=cfg.epochs,
                             batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                             patience=cfg.patience, min_delta=cfg.min_delta, device=self.device,
                             rng=rng, verbose=cfg.verbose)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from .cnn import predict_proba

        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        return predict_proba(self.model, X, self.device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, directory: str) -> None:
        import torch

        os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self._fitted_input_dim,
                    "num_classes": self.num_classes, "steps": self.steps},
                   os.path.join(directory, "stage1_classifier.pt"))
        with open(os.path.join(directory, "stage1_history.json"), "w") as fh:
            json.dump({"num_classes": self.num_classes, "steps": self.steps,
                       "history": self.history}, fh, indent=2)

    @classmethod
    def load(cls, directory: str, cfg: ClassifierConfig) -> "StageOneClassifier":
        import torch

        with open(os.path.join(directory, "stage1_history.json")) as fh:
            meta = json.load(fh)
        obj = cls(cfg, meta["num_classes"], meta.get("steps", 1))
        obj._configure_device()
        ckpt = torch.load(os.path.join(directory, "stage1_classifier.pt"), map_location=obj.device)
        obj.model = build_cnn_classifier(ckpt["input_dim"], meta["num_classes"], cfg,
                                         ckpt.get("steps", 1)).to(obj.device)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj._fitted_input_dim = ckpt["input_dim"]
        obj.history = meta["history"]
        return obj


# ----------------------------------------------------------------------------
# One-stage neural regressor (not in the notebook; see NetConfig)
# ----------------------------------------------------------------------------

def build_net(input_dim: int, out_features: int, cfg: NetConfig, steps: int = 1):
    from .cnn import FeedForwardNet

    return FeedForwardNet(input_dim, out_features, cfg.conv_filters, cfg.dense_units,
                          cfg.dropout, cfg.regularizers, encoder=cfg.encoder,
                          norm_position=cfg.norm_position, conv_bias=cfg.conv_bias,
                          steps=steps, snapshot_hidden=cfg.snapshot_hidden,
                          temporal_filters=cfg.temporal_filters,
                          temporal_kernel=cfg.temporal_kernel, pool=cfg.pool)


class DirectRegressor:
    """Predicts the N/C ratio with one network: no class bins, no forest.

    Replaces both notebook stages. Early stopping holds out whole trajectories,
    so ``groups`` is required unless cfg.validation == "random_rows". For a
    regression head the output bias starts at the mean training target, so
    epoch 0 is the constant-prediction baseline and nothing is learned by
    drifting towards the mean.
    """

    def __init__(self, cfg: NetConfig, steps: int = 1):
        self.cfg = cfg
        self.steps = steps          # snippet length (trajl); only the sequence encoders use it
        self.model = None
        self.device = None
        self.objective = None
        self._fitted_input_dim: Optional[int] = None
        self.history: dict = {}

    def _objective(self, y_train: np.ndarray):
        from .cnn import HistogramObjective, RegressionObjective

        cfg = self.cfg
        if cfg.head == "regression":
            return RegressionObjective(cfg.target_transform, cfg.loss, cfg.huber_delta)
        if cfg.head == "histogram":
            # Bins come from the training rows only, never the validation rows.
            return HistogramObjective.from_labels(y_train, cfg.n_bins)
        raise ValueError(f"unknown head {cfg.head!r}")

    def _split(self, X, y, groups):
        cfg = self.cfg
        if cfg.validation == "grouped_holdout":
            if groups is None:
                raise ValueError("validation='grouped_holdout' needs groups (trajectory ids per row)")
            from sklearn.model_selection import GroupShuffleSplit

            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=cfg.validation_fraction,
                                            random_state=cfg.validation_seed).split(X, y, groups))
            return tr, va
        if cfg.validation == "random_rows":
            # Optimistic for this data (adjacent frames of one cell land on both
            # sides); here only to measure that effect.
            rng = np.random.default_rng(cfg.validation_seed)
            order = rng.permutation(len(y))
            cut = int(math.floor(len(y) * (1.0 - cfg.validation_fraction)))
            return order[:cut], order[cut:]
        raise ValueError(f"unknown validation {cfg.validation!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> "DirectRegressor":
        import torch

        from .cnn import inverse_density_weights, train

        cfg = self.cfg
        from .compute import select_device

        self.device = select_device(cfg.device, cfg.num_threads)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        tr, va = self._split(X, y, groups)
        self.objective = self._objective(y[tr])
        t_tr, t_va = self.objective.encode(y[tr]), self.objective.encode(y[va])
        weights = None
        if cfg.label_weighting == "inverse_density":
            weights = inverse_density_weights(y[tr], clip=cfg.weight_clip)
        elif cfg.label_weighting != "none":
            raise ValueError(f"unknown label_weighting {cfg.label_weighting!r}")

        self._fitted_input_dim = int(X.shape[1])
        self.model = build_net(X.shape[1], self.objective.out_features, cfg, self.steps)
        if cfg.head == "regression":
            with torch.no_grad():
                self.model.out.bias.fill_(float(np.mean(t_tr)))
        log.info("direct regressor: %s encoder, %s head, %d train rows, %d validation rows (%s)",
                 cfg.encoder, cfg.head, tr.size, va.size, cfg.validation)
        self.history = train(self.model, X[tr], t_tr, X[va], t_va, epochs=cfg.epochs,
                             batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                             patience=cfg.patience, min_delta=cfg.min_delta, device=self.device,
                             rng=rng, verbose=cfg.verbose, objective=self.objective,
                             weights=weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from .cnn import predict

        if self.model is None:
            raise RuntimeError("regressor is not fitted")
        return predict(self.model, X, self.device, self.objective)

    def save(self, directory: str) -> None:
        import torch

        from .cnn import HistogramObjective

        os.makedirs(directory, exist_ok=True)
        state = {"state_dict": self.model.state_dict(), "input_dim": self._fitted_input_dim,
                 "out_features": self.objective.out_features, "steps": self.steps}
        if isinstance(self.objective, HistogramObjective):
            state["edges"] = self.objective.edges
            state["centers"] = self.objective.centers
        torch.save(state, os.path.join(directory, "direct_regressor.pt"))
        with open(os.path.join(directory, "direct_regressor_history.json"), "w") as fh:
            json.dump({"config": dataclasses.asdict(self.cfg), "steps": self.steps,
                       "history": self.history}, fh, indent=2)

    @classmethod
    def load(cls, directory: str, cfg: Optional[NetConfig] = None) -> "DirectRegressor":
        import torch

        from .compute import select_device
        from .config import _from_dict

        with open(os.path.join(directory, "direct_regressor_history.json")) as fh:
            meta = json.load(fh)
        obj = cls(cfg if cfg is not None else _from_dict(NetConfig, meta["config"]),
                  meta.get("steps", 1))
        obj.device = select_device(obj.cfg.device, obj.cfg.num_threads)
        ckpt = torch.load(os.path.join(directory, "direct_regressor.pt"), map_location=obj.device,
                          weights_only=False)
        if "edges" in ckpt:
            from .cnn import HistogramObjective

            obj.objective = HistogramObjective(ckpt["edges"], ckpt["centers"])
        else:
            obj.objective = obj._objective(np.zeros(1))
        obj.steps = ckpt.get("steps", obj.steps)
        obj.model = build_net(ckpt["input_dim"], ckpt["out_features"], obj.cfg,
                              obj.steps).to(obj.device)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj._fitted_input_dim = ckpt["input_dim"]
        obj.history = meta["history"]
        return obj


def out_of_fold_proba(X: np.ndarray, y_class: np.ndarray, folds: np.ndarray,
                      cfg: ClassifierConfig, num_classes: int,
                      groups: Optional[np.ndarray] = None, steps: int = 1) -> np.ndarray:
    """Class probabilities for every row from a classifier that never saw that row's fold.

    With folds = FOV of each training row this is leave-one-FOV-out
    cross-fitting: the stage-2 model is then trained on probabilities that are
    as (un)confident as the ones it will see on new FOVs.
    """
    proba = np.zeros((X.shape[0], num_classes))
    for k in np.unique(folds):
        held = folds == k
        log.info("out-of-fold classifier: holding out fold %s (%d rows)", k, held.sum())
        clf = StageOneClassifier(cfg, num_classes, steps).fit(
            X[~held], y_class[~held], None if groups is None else groups[~held])
        proba[held] = clf.predict_proba(X[held])
    return proba


# ----------------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------------

def stack_features(X: np.ndarray, class_proba: np.ndarray) -> np.ndarray:
    """[scaled features | stage-1 class probabilities] (In [42])."""
    return np.concatenate((X, class_proba), axis=1)


def stage2_split(groups: np.ndarray, fraction: float, seed: int):
    """Split rows into (stage 1, stage 2) by whole trajectories.

    For fit_on="train_holdout": stage 1 trains on the first part, stage 2 on the
    second, so the probabilities stage 2 learns from are out-of-sample for stage 1
    and no cell contributes to both stages.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if groups is None:
        raise ValueError("fit_on='train_holdout' needs groups (trajectory ids per row)")
    if not 0 < fraction < 1:
        raise ValueError(f"stage2_fraction must be strictly between 0 and 1, got {fraction}")
    return next(GroupShuffleSplit(n_splits=1, test_size=fraction,
                                  random_state=seed).split(np.zeros(len(groups)), None, groups))


class StageTwoRegressor:
    """MinMaxScaler -> RFECV(RandomForestRegressor), as in In [44].

    The scaler is redundant (inputs are already in [0, 1] or close, and trees
    are invariant to monotone rescaling) but kept for identical numbers.
    RFECV with step=1 from 259 features down to 50 refits a 5-fold forest
    210 times; that is where the runtime goes. Folds run one at a time by
    default (rfecv_n_jobs=1) so each forest can use every core.
    """

    def __init__(self, cfg: RegressorConfig):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        self.cfg = cfg
        forest = RandomForestRegressor(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features, n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbose=0)
        if not cfg.use_rfecv:
            self.pipeline = Pipeline([("scaler", MinMaxScaler()), ("forest", forest)])
            return
        cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.cv_seed)
        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("rfecv", RFECV(estimator=forest, step=cfg.rfecv_step, cv=cv, scoring=cfg.scoring,
                            min_features_to_select=cfg.min_features_to_select, n_jobs=cfg.rfecv_n_jobs)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StageTwoRegressor":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    @property
    def rfecv(self):
        """The fitted RFECV step, or None when use_rfecv is False."""
        return self.pipeline.named_steps.get("rfecv")

    @property
    def n_features_used(self) -> int:
        if self.rfecv is not None:
            return int(self.rfecv.n_features_)
        return int(self.pipeline.named_steps["forest"].n_features_in_)

    def save(self, directory: str) -> None:
        import joblib

        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(directory, "stage2_regressor.joblib"))

    @classmethod
    def load(cls, directory: str, cfg: Optional[RegressorConfig] = None) -> "StageTwoRegressor":
        import joblib

        obj = cls(cfg or RegressorConfig())
        obj.pipeline = joblib.load(os.path.join(directory, "stage2_regressor.joblib"))
        return obj
