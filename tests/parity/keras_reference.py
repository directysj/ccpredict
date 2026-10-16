"""The notebook's stage-1 classifier in Keras/TensorFlow, kept only as a reference.

The package's stage 1 is PyTorch (ccpredict/cnn.py). This file preserves the
Keras version that reproduced the notebook bit for bit, so the port can be
checked against it (tests/parity/test_torch_port.py, torch_vs_keras.py).
Needs TensorFlow + Keras 3; nothing in the package imports it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _regularizer(kind: str, value: float):
    from keras.regularizers import l1, l2
    return {"l1": l1, "l2": l2}[kind](value)


def build_keras_cnn(input_dim: int, num_classes: int, cfg, dropout: Optional[float] = None):
    """The notebook's create_model (In [27]), parameterised as in ClassifierConfig."""
    from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input,
                              LeakyReLU, MaxPooling2D, Reshape)
    from keras.models import Model

    rate = cfg.dropout if dropout is None else dropout
    regs = cfg.regularizers
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1, 1))(inputs)
    for i, filters in enumerate(cfg.conv_filters):
        x = Conv2D(filters, kernel_size=(3, 1), padding="same",
                   kernel_regularizer=_regularizer(*regs[i % len(regs)]))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(rate)(x)
    x = Flatten()(x)
    for i, units in enumerate(cfg.dense_units):
        x = Dense(units, kernel_regularizer=_regularizer(*regs[i % len(regs)]))(x)
        x = LeakyReLU()(x)
        x = Dropout(rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


class KerasStageOne:
    """Same fit protocol as ccpredict.models.StageOneClassifier, in Keras (notebook In [33]-[37])."""

    def __init__(self, cfg, num_classes: int):
        self.cfg, self.num_classes = cfg, num_classes
        self.model, self.history = None, {}

    def fit(self, X, y_class, groups=None):
        import keras
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam
        from keras.utils import to_categorical

        from ccpredict.preprocessing import make_oversampler

        cfg = self.cfg
        if cfg.seed is not None:
            keras.utils.set_random_seed(cfg.seed)
        self.model = build_keras_cnn(X.shape[1], self.num_classes, cfg)
        self.model.compile(optimizer=Adam(learning_rate=cfg.learning_rate),
                           loss="categorical_crossentropy", metrics=["accuracy"])
        stop = EarlyStopping(monitor="val_loss", restore_best_weights=True,
                             patience=cfg.patience, min_delta=cfg.min_delta)
        onehot = lambda v: to_categorical(v, num_classes=self.num_classes)  # noqa: E731
        sampler = make_oversampler(cfg.oversampler, cfg.oversampling_strategy, cfg.oversampling_seed)
        resample = (lambda a, b: (a, b)) if sampler is None else sampler.fit_resample
        if cfg.validation == "keras_tail":
            X_res, y_res = resample(X, y_class)
            hist = self.model.fit(X_res, onehot(y_res), epochs=cfg.epochs, batch_size=cfg.batch_size,
                                  validation_split=cfg.validation_fraction, callbacks=[stop],
                                  verbose=cfg.verbose)
        else:
            from sklearn.model_selection import GroupShuffleSplit

            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=cfg.validation_fraction,
                                            random_state=cfg.oversampling_seed).split(X, y_class, groups))
            X_res, y_res = resample(X[tr], y_class[tr])
            hist = self.model.fit(X_res, onehot(y_res), epochs=cfg.epochs, batch_size=cfg.batch_size,
                                  validation_data=(X[va], onehot(y_class[va])), callbacks=[stop],
                                  verbose=cfg.verbose)
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)


def copy_keras_weights_to_torch(keras_model, torch_model) -> None:
    """Load a Keras CNN's weights (and BatchNorm moving statistics) into ccpredict.cnn.FeedForwardNet."""
    import torch

    convs = [l for l in keras_model.layers if l.__class__.__name__ == "Conv2D"]
    bns = [l for l in keras_model.layers if l.__class__.__name__ == "BatchNormalization"]
    denses = [l for l in keras_model.layers if l.__class__.__name__ == "Dense"]
    with torch.no_grad():
        for kl, tl in zip(convs, torch_model.convs):
            kernel, bias = kl.get_weights()                      # (3, 1, in, out), (out,)
            tl.weight.copy_(torch.from_numpy(np.ascontiguousarray(kernel[:, 0].transpose(2, 1, 0))))
            tl.bias.copy_(torch.from_numpy(bias))
        for kl, tl in zip(bns, torch_model.norms):
            gamma, beta, mean, var = kl.get_weights()
            tl.weight.copy_(torch.from_numpy(gamma)); tl.bias.copy_(torch.from_numpy(beta))
            tl.running_mean.copy_(torch.from_numpy(mean)); tl.running_var.copy_(torch.from_numpy(var))
        for kl, tl in zip(denses, list(torch_model.denses) + [torch_model.out]):
            kernel, bias = kl.get_weights()                      # (in, out), (out,)
            tl.weight.copy_(torch.from_numpy(np.ascontiguousarray(kernel.T)))
            tl.bias.copy_(torch.from_numpy(bias))
