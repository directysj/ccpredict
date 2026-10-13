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

    def fit(self, X: np.ndarray, y_class: np.ndarray) -> "StageOneClassifier":
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
        elif cfg.validation == "stratified_holdout":
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_class, test_size=cfg.validation_fraction, stratify=y_class,
                random_state=cfg.oversampling_seed)
            X_res, y_res = self._oversample(X_tr, y_tr)
            hist = self.model.fit(X_res, onehot(y_res), epochs=cfg.epochs, batch_size=cfg.batch_size,
                                  validation_data=(X_val, onehot(y_val)),
                                  callbacks=[early_stopping], verbose=cfg.verbose)
        else:
            raise ValueError(f"unknown validation {cfg.validation!r}")

        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

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
        import keras

        with open(os.path.join(directory, "stage1_history.json")) as fh:
            meta = json.load(fh)
        obj = cls(cfg, meta["num_classes"])
        obj.model = keras.models.load_model(os.path.join(directory, "stage1_classifier.keras"))
        obj.history = meta["history"]
        return obj


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
    210 times; that is where the runtime goes.
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
        cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.cv_seed)
        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("rfecv", RFECV(estimator=forest, step=cfg.rfecv_step, cv=cv, scoring=cfg.scoring,
                            min_features_to_select=cfg.min_features_to_select, n_jobs=cfg.n_jobs)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StageTwoRegressor":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    @property
    def rfecv(self):
        return self.pipeline.named_steps["rfecv"]

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
