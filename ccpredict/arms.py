"""Models compared by the benchmark ("arms"). Each arm is fitted once per split and scored on
every test FOV of that split.

    constant                      training-set mean (anything must beat this)
    lasso_classical               LASSO on handcrafted features       (paper: R 0.510)
    lasso_vae                     LASSO on VAE embeddings              (paper: R 0.531)
    rf_vae                        random forest on VAE, fit on training FOVs  (paper: R 0.539)
    rf_vae_insample               same forest, fit AND scored on the test FOV (diagnostic)
    published_insample            CNN probabilities + RFECV forest exactly as in the notebook:
                                  forest fit and scored on the test FOV (paper: R 0.755)
    published_crossfit            same CNN, forest fit on training FOVs with out-of-fold CNN
                                  probabilities, scored on the test FOVs
    published_crossfit_groupedval as published_crossfit, but the CNN's early stopping uses held-out
                                  trajectories instead of the oversampled tail
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import ClassifierConfig, RegressorConfig
from .preprocessing import discretize

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Data blocks
# ----------------------------------------------------------------------------

@dataclass
class Block:
    """Rows of one or more FOVs: features, labels and trajectory bookkeeping."""

    X: np.ndarray
    y: np.ndarray
    fold: np.ndarray     # index of the FOV within the block's FOV list
    traj: np.ndarray     # trajectory index within that FOV
    fovs: tuple

    @property
    def groups(self) -> np.ndarray:
        from .preprocessing import trajectory_groups
        return trajectory_groups(self.fold, self.traj)

    def __len__(self) -> int:
        return self.y.size


@dataclass
class ArmResult:
    y_pred: np.ndarray
    proba: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)


@dataclass
class SplitContext:
    """What an arm sees: the training block, the test blocks and shared caches."""

    split_name: str
    train: Block                   # trimmed if the benchmark trims
    tests: Dict[str, Block]
    columns: Callable[[str], np.ndarray]
    classifier_cfg: ClassifierConfig
    regressor_cfg: RegressorConfig
    cache: dict = field(default_factory=dict)

    def X_train(self, feature_set: str) -> np.ndarray:
        return self.train.X[:, self.columns(feature_set)]

    def X_test(self, fov: str, feature_set: str) -> np.ndarray:
        return self.tests[fov].X[:, self.columns(feature_set)]


class Arm:
    name: str = "arm"
    feature_sets: tuple = ("vae",)

    def run(self, ctx: SplitContext) -> Dict[str, ArmResult]:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

class ConstantArm(Arm):
    name = "constant"
    feature_sets = ()

    def run(self, ctx):
        mean = float(ctx.train.y.mean())
        return {f: ArmResult(np.full(len(b), mean), info={"value": mean}) for f, b in ctx.tests.items()}


def _lasso_alphas(X: np.ndarray, y: np.ndarray, n: int = 30) -> np.ndarray:
    """Explicit alpha grid, so the code runs on old and new scikit-learn alike."""
    alpha_max = np.max(np.abs(X.T @ (y - y.mean()))) / X.shape[0]
    return np.logspace(np.log10(alpha_max * 1e-4), np.log10(alpha_max), n)[::-1]


class LassoArm(Arm):
    """StandardScaler + LassoCV, alpha chosen by trajectory-grouped 5-fold CV."""

    def __init__(self, feature_set: str):
        self.feature_set = feature_set
        self.feature_sets = (feature_set,)
        self.name = f"lasso_{feature_set.replace('+', '_')}"

    def run(self, ctx):
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(ctx.X_train(self.feature_set))
        cv = list(GroupKFold(n_splits=5).split(X, ctx.train.y, ctx.train.groups))
        model = LassoCV(alphas=_lasso_alphas(X, ctx.train.y), cv=cv, n_jobs=-1, max_iter=5000)
        model.fit(X, ctx.train.y)
        info = {"alpha": float(model.alpha_), "n_nonzero": int(np.sum(model.coef_ != 0))}
        return {f: ArmResult(model.predict(scaler.transform(ctx.X_test(f, self.feature_set))), info=info)
                for f in ctx.tests}


def _forest(cfg: RegressorConfig):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features, n_jobs=cfg.n_jobs, random_state=cfg.random_state)


class ForestArm(Arm):
    """The paper's forest hyperparameters, no feature selection.

    fit_on="train": fit on the training FOVs, score on each test FOV.
    fit_on="test":  fit on each test FOV and score the same rows (in-sample).
    """

    def __init__(self, feature_set: str = "vae", fit_on: str = "train"):
        self.feature_set, self.fit_on = feature_set, fit_on
        self.feature_sets = (feature_set,)
        self.name = f"rf_{feature_set.replace('+', '_')}" + ("_insample" if fit_on == "test" else "")

    def run(self, ctx):
        out = {}
        if self.fit_on == "train":
            model = _forest(ctx.regressor_cfg).fit(ctx.X_train(self.feature_set), ctx.train.y)
            for f in ctx.tests:
                out[f] = ArmResult(model.predict(ctx.X_test(f, self.feature_set)))
        else:
            for f, b in ctx.tests.items():
                X = ctx.X_test(f, self.feature_set)
                out[f] = ArmResult(_forest(ctx.regressor_cfg).fit(X, b.y).predict(X))
        return out


# ----------------------------------------------------------------------------
# The published two-stage model
# ----------------------------------------------------------------------------

class PublishedArm(Arm):
    """Notebook stage 1 (CNN on MinMax-scaled VAE features) + stage 2 (RFECV forest).

    The MinMax scaler is fit on the (trimmed) training block, as in the notebook.
    The stage-1 classifier is shared between arms with the same validation
    mode through ctx.cache, so the CNN is trained once per mode and split.
    """

    feature_sets = ("vae",)

    def __init__(self, fit_on: str, validation: str = "keras_tail"):
        if fit_on not in ("test", "train_crossfit"):
            raise ValueError(fit_on)
        self.fit_on, self.validation = fit_on, validation
        self.name = {"test": "published_insample", "train_crossfit": "published_crossfit"}[fit_on]
        if validation != "keras_tail":
            self.name += "_groupedval"

    def _scaled(self, ctx):
        key = ("minmax",)
        if key not in ctx.cache:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler().fit(ctx.X_train("vae"))
            ctx.cache[key] = (scaler.transform(ctx.X_train("vae")),
                              {f: scaler.transform(ctx.X_test(f, "vae")) for f in ctx.tests})
        return ctx.cache[key]

    def _clf_cfg(self, ctx) -> ClassifierConfig:
        cfg = copy.deepcopy(ctx.classifier_cfg)
        cfg.validation = self.validation
        return cfg

    def _classifier(self, ctx):
        from .models import StageOneClassifier

        key = ("clf", self.validation)
        if key not in ctx.cache:
            Xs, _ = self._scaled(ctx)
            cfg = self._clf_cfg(ctx)
            y_class = discretize(ctx.train.y, cfg.class_boundaries)
            t0 = time.time()
            clf = StageOneClassifier(cfg, len(cfg.class_boundaries) + 1).fit(Xs, y_class, ctx.train.groups)
            ctx.cache[key] = (clf, time.time() - t0)
        return ctx.cache[key]

    def _oof(self, ctx):
        from .models import out_of_fold_proba

        key = ("oof", self.validation)
        if key not in ctx.cache:
            Xs, _ = self._scaled(ctx)
            cfg = self._clf_cfg(ctx)
            y_class = discretize(ctx.train.y, cfg.class_boundaries)
            ctx.cache[key] = out_of_fold_proba(Xs, y_class, ctx.train.fold, cfg,
                                               len(cfg.class_boundaries) + 1, groups=ctx.train.groups)
        return ctx.cache[key]

    def run(self, ctx):
        from .models import StageTwoRegressor, stack_features

        Xs_train, Xs_test = self._scaled(ctx)
        clf, clf_seconds = self._classifier(ctx)
        info = {"classifier_epochs": len(clf.history.get("loss", [])),
                "classifier_seconds": round(clf_seconds, 1)}
        if "val_loss" in clf.history and clf.history["val_loss"]:
            info["classifier_best_epoch"] = int(np.argmin(clf.history["val_loss"])) + 1
        proba = {f: clf.predict_proba(Xs_test[f]) for f in ctx.tests}

        out = {}
        if self.fit_on == "test":
            for f, b in ctx.tests.items():
                Z = stack_features(Xs_test[f], proba[f])
                reg = StageTwoRegressor(ctx.regressor_cfg).fit(Z, b.y)
                out[f] = ArmResult(reg.predict(Z), proba[f], dict(info, n_features=reg.n_features_used))
        else:
            reg = StageTwoRegressor(ctx.regressor_cfg).fit(stack_features(Xs_train, self._oof(ctx)), ctx.train.y)
            for f in ctx.tests:
                out[f] = ArmResult(reg.predict(stack_features(Xs_test[f], proba[f])), proba[f],
                                   dict(info, n_features=reg.n_features_used))
        return out


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

ARMS: Dict[str, Callable[[], Arm]] = {
    "constant": ConstantArm,
    "lasso_classical": lambda: LassoArm("classical"),
    "lasso_vae": lambda: LassoArm("vae"),
    "rf_vae": lambda: ForestArm("vae", "train"),
    "rf_vae_insample": lambda: ForestArm("vae", "test"),
    "published_insample": lambda: PublishedArm("test", "keras_tail"),
    "published_crossfit": lambda: PublishedArm("train_crossfit", "keras_tail"),
    "published_crossfit_groupedval": lambda: PublishedArm("train_crossfit", "grouped_holdout"),
}

DEFAULT_ARMS: List[str] = list(ARMS)


def make_arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(ARMS)}")
    return ARMS[name]()
