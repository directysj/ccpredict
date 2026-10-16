"""Stage 1 (CNN+DNN classifier) and stage 2 (RFECV random forest regressor).

Notebook In [27], [33]-[37] (stage 1) and In [42], [44] plus the unexecuted fit cells (stage 2).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from .config import ClassifierConfig, RegressorConfig
from .preprocessing import make_oversampler

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------------

def _regularizer(kind: str, value: float):
    from keras.regularizers import l1, l2
    return {"l1": l1, "l2": l2}[kind](value)


def build_cnn_classifier(input_dim: int, num_classes: int, cfg: ClassifierConfig):
    """The notebook's create_model, parameterised.

    Input (d,) is reshaped to (d, 1, 1) so Conv2D with a (3, 1) kernel is a 1-D
    convolution along the feature index. For VAE latents that index has no
    ordering, so "neighbouring" dimensions are arbitrary.
    """
    from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input,
                              LeakyReLU, MaxPooling2D, Reshape)
    from keras.models import Model

    regs = cfg.regularizers
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1, 1))(inputs)
    for i, filters in enumerate(cfg.conv_filters):
        x = Conv2D(filters, kernel_size=(3, 1), padding="same",
                   kernel_regularizer=_regularizer(*regs[i % len(regs)]))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(cfg.dropout)(x)
    x = Flatten()(x)
    for i, units in enumerate(cfg.dense_units):
        x = Dense(units, kernel_regularizer=_regularizer(*regs[i % len(regs)]))(x)
        x = LeakyReLU()(x)
        x = Dropout(cfg.dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


class StageOneClassifier:
    """Oversample, train the CNN+DNN with early stopping, predict class probabilities."""

    def __init__(self, cfg: ClassifierConfig, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = None
        self.history: dict = {}

    def _oversample(self, X, y):
        sampler = make_oversampler(self.cfg.oversampler, self.cfg.oversampling_strategy,
                                   self.cfg.oversampling_seed)
        return (X, y) if sampler is None else sampler.fit_resample(X, y)

    def fit(self, X: np.ndarray, y_class: np.ndarray,
            groups: Optional[np.ndarray] = None) -> "StageOneClassifier":
        """Train on (X, y_class). ``groups`` (trajectory ids) is required for grouped_holdout."""
        self._configure_device()
        import keras
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam
        from keras.utils import to_categorical

        cfg = self.cfg
        if cfg.seed is not None:
            keras.utils.set_random_seed(cfg.seed)

        self.model = build_cnn_classifier(X.shape[1], self.num_classes, cfg)
        self.model.compile(optimizer=Adam(learning_rate=cfg.learning_rate),
                           loss="categorical_crossentropy", metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor="val_loss", restore_best_weights=True,
                                       patience=cfg.patience, min_delta=cfg.min_delta)
        onehot = lambda labels: to_categorical(labels, num_classes=self.num_classes)  # noqa: E731

        if cfg.validation == "keras_tail":
            # Notebook behaviour. Keras takes the LAST validation_fraction of the
            # arrays, and the oversampler appends its duplicates at the end,
            # grouped by class, so validation = copies of minority-class
            # training rows. val_loss then drives early stopping and the
            # restored weights.
            log.warning("validation='keras_tail': validation rows are oversampled duplicates "
                        "of training rows (notebook behaviour)")
            X_res, y_res = self._oversample(X, y_class)
            hist = self.model.fit(X_res, onehot(y_res), epochs=cfg.epochs, batch_size=cfg.batch_size,
                                  validation_split=cfg.validation_fraction,
                                  callbacks=[early_stopping], verbose=cfg.verbose)
        elif cfg.validation == "grouped_holdout":
            # Hold out whole trajectories, then oversample only the training part.
            # Splitting rows at random would put adjacent, near-identical frames
            # of one cell on both sides and make val_loss optimistic.
            if groups is None:
                raise ValueError("validation='grouped_holdout' needs groups (trajectory ids per row)")
            from sklearn.model_selection import GroupShuffleSplit

            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=cfg.validation_fraction,
                                            random_state=cfg.oversampling_seed).split(X, y_class, groups))
            X_tr, X_val, y_tr, y_val = X[tr], X[va], y_class[tr], y_class[va]
            X_res, y_res = self._oversample(X_tr, y_tr)
            hist = self.model.fit(X_res, onehot(y_res), epochs=cfg.epochs, batch_size=cfg.batch_size,
                                  validation_data=(X_val, onehot(y_val)),
                                  callbacks=[early_stopping], verbose=cfg.verbose)
        else:
            raise ValueError(f"unknown validation {cfg.validation!r}")

        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def _configure_device(self) -> None:
        from .compute import configure_tensorflow

        configure_tensorflow(self.cfg.device, self.cfg.tf_intra_op_threads, self.cfg.tf_inter_op_threads)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "stage1_classifier.keras"))
        with open(os.path.join(directory, "stage1_history.json"), "w") as fh:
            json.dump({"num_classes": self.num_classes, "history": self.history}, fh, indent=2)

    @classmethod
    def load(cls, directory: str, cfg: ClassifierConfig) -> "StageOneClassifier":
        with open(os.path.join(directory, "stage1_history.json")) as fh:
            meta = json.load(fh)
        obj = cls(cfg, meta["num_classes"])
        obj._configure_device()
        import keras

        obj.model = keras.models.load_model(os.path.join(directory, "stage1_classifier.keras"))
        obj.history = meta["history"]
        return obj


def out_of_fold_proba(X: np.ndarray, y_class: np.ndarray, folds: np.ndarray,
                      cfg: ClassifierConfig, num_classes: int,
                      groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Class probabilities for every row from a classifier that never saw that row's fold.

    With folds = FOV of each training row this is leave-one-FOV-out
    cross-fitting: the stage-2 model is then trained on probabilities that are
    as (un)confident as the ones it will see on new FOVs.
    """
    proba = np.zeros((X.shape[0], num_classes))
    for k in np.unique(folds):
        held = folds == k
        log.info("out-of-fold classifier: holding out fold %s (%d rows)", k, held.sum())
        clf = StageOneClassifier(cfg, num_classes).fit(
            X[~held], y_class[~held], None if groups is None else groups[~held])
        proba[held] = clf.predict_proba(X[held])
    return proba


# ----------------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------------

def stack_features(X: np.ndarray, class_proba: np.ndarray) -> np.ndarray:
    """[scaled features | stage-1 class probabilities] (In [42])."""
    return np.concatenate((X, class_proba), axis=1)


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
