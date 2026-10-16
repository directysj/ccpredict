"""Exact checks of the PyTorch stage 1 against the notebook's Keras model.

Needs both TensorFlow/Keras 3 and PyTorch, so it lives with the other parity
tools and is skipped when either is missing:

    pytest tests/parity/test_torch_port.py

What can be exact is checked exactly: the forward pass with identical weights,
the loss (cross-entropy + L1/L2 penalties), the gradients and the Adam update
over several steps with dropout off, the BatchNorm moving statistics, and the
early-stopping decisions. What cannot (random dropout masks, shuffling order,
float kernels) is compared statistically in torch_vs_keras.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from ccpredict.cnn import EarlyStopping, FeedForwardNet, KerasAdam  # noqa: E402
from ccpredict.config import ClassifierConfig  # noqa: E402
from keras_reference import build_keras_cnn, copy_keras_weights_to_torch  # noqa: E402

D, K = 256, 3


def _pair(dropout=None, seed=0):
    cfg = ClassifierConfig()
    keras.utils.set_random_seed(seed)
    km = build_keras_cnn(D, K, cfg, dropout=dropout)
    tm = FeedForwardNet(D, K, cfg.conv_filters, cfg.dense_units,
                       cfg.dropout if dropout is None else dropout, cfg.regularizers)
    copy_keras_weights_to_torch(km, tm)
    return cfg, km, tm


def test_forward_pass_identical_with_copied_weights():
    _, km, tm = _pair()
    rng = np.random.default_rng(1)
    # Non-trivial BatchNorm moving statistics.
    for layer in km.layers:
        if layer.__class__.__name__ == "BatchNormalization":
            g, b, m, v = layer.get_weights()
            layer.set_weights([g * rng.uniform(0.5, 1.5, g.shape).astype("float32"),
                               rng.normal(0, 0.3, b.shape).astype("float32"),
                               rng.normal(0, 0.5, m.shape).astype("float32"),
                               rng.uniform(0.5, 2.0, v.shape).astype("float32")])
    copy_keras_weights_to_torch(km, tm)
    X = rng.random((500, D)).astype("float32")
    pk = km.predict(X, verbose=0)
    tm.eval()
    with torch.no_grad():
        pt = torch.softmax(tm(torch.from_numpy(X)), 1).numpy()
    assert np.max(np.abs(pk - pt)) < 2e-6


def test_penalty_matches_keras_regularisation_losses():
    _, km, tm = _pair()
    keras_pen = float(sum(np.asarray(l) for l in km.losses))
    assert abs(keras_pen - tm.penalty().item()) < 1e-4 * max(1.0, keras_pen)


def test_gradients_identical():
    """One forward/backward pass in training mode on MinMax-like inputs in [0, 1]."""
    _, km, tm = _pair(dropout=0.0)
    rng = np.random.default_rng(2)
    X = rng.random((64, D)).astype("float32")
    y = rng.integers(0, K, 64)
    with tf.GradientTape() as tape:
        p = km(X, training=True)
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(keras.utils.to_categorical(y, K), p))
        loss = loss + tf.add_n(km.losses)
    k_grads = tape.gradient(loss, km.trainable_weights)
    probe = FeedForwardNet(D, K, (32, 64), (128, 64), 0.0)
    tm.train()
    t_loss = torch.nn.functional.cross_entropy(tm(torch.from_numpy(X)), torch.from_numpy(y)) + tm.penalty()
    t_loss.backward()
    # Map Keras gradients onto torch parameter names via the weight-copy routine.
    for var, g in zip(km.trainable_weights, k_grads):
        var.assign(g)
    copy_keras_weights_to_torch(km, probe)
    for (name, a), (_, b) in zip(probe.named_parameters(), tm.named_parameters()):
        a = a.detach()
        assert float((a - b.grad).abs().max()) < 1e-5 * max(float(a.abs().max()), 1.0), name


def test_training_steps_identical_without_dropout():
    """Same weights, same batches, dropout off: loss, weights and BN statistics after
    each step must agree (float32 noise only).

    Inputs are centred here on purpose. With inputs in [0, 1], a first-layer
    channel whose three kernel weights share a sign never leaves LeakyReLU's
    linear branch, so the BatchNorm right after it cancels that channel's bias
    exactly: its true gradient is 0 and both frameworks compute float noise of
    ~1e-7 (test_gradients_identical sees this; the absolute error is tiny).
    Adam divides by sqrt(v), turning that noise into full lr-sized steps in a
    random direction, so those biases random-walk differently in Keras and in
    PyTorch (and between two Keras runs) without affecting any output. Centred
    inputs give every bias a real gradient, so the trajectories can be compared."""
    cfg, km, tm = _pair(dropout=0.0)
    km.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy")
    opt = KerasAdam(tm.parameters(), lr=1e-3)
    rng = np.random.default_rng(2)
    tm.train()
    for step in range(10):
        X = (rng.random((20, D)) - 0.5).astype("float32")
        y = rng.integers(0, K, 20)
        k_loss = float(km.train_on_batch(X, keras.utils.to_categorical(y, K)))
        logits = tm(torch.from_numpy(X))
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)) + tm.penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert abs(k_loss - loss.item()) < 1e-4 * max(1.0, abs(k_loss)), (step, k_loss, loss.item())

    # Weights after 10 Adam steps.
    probe = FeedForwardNet(D, K, cfg.conv_filters, cfg.dense_units, 0.0, cfg.regularizers)
    copy_keras_weights_to_torch(km, probe)
    for (name, a), (_, b) in zip(probe.state_dict().items(), tm.state_dict().items()):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), name


def test_whole_fit_loop_matches_keras_without_dropout_or_shuffling():
    """cnn.train against Keras' model.fit end to end: per-epoch loss and val_loss
    (penalties included), early stopping, restored weights and predictions.
    Dropout and shuffling are off so both see the same batches; everything else
    is the real loop. Float32 differences compound over ~100 steps, hence rtol."""
    from ccpredict.cnn import predict_proba, train

    d, n_tr, n_va = 64, 400, 100
    rng = np.random.default_rng(4)
    X = rng.random((n_tr + n_va + 200, d)).astype("float32")
    score = X[:, :8].sum(1) - X[:, 8:16].sum(1) + rng.normal(0, 0.5, X.shape[0])
    y = np.digitize(score, np.quantile(score, [0.3, 0.7]))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    X_te = X[n_tr + n_va:]

    cfg = ClassifierConfig()
    keras.utils.set_random_seed(5)
    km = build_keras_cnn(d, K, cfg, dropout=0.0)
    tm = FeedForwardNet(d, K, cfg.conv_filters, cfg.dense_units, 0.0, cfg.regularizers)
    copy_keras_weights_to_torch(km, tm)

    epochs, batch, patience, min_delta = 12, 32, 3, 1e-3
    km.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy",
               metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=min_delta,
                                         restore_best_weights=True)
    kh = km.fit(X_tr, keras.utils.to_categorical(y_tr, K), epochs=epochs, batch_size=batch,
                validation_data=(X_va, keras.utils.to_categorical(y_va, K)), shuffle=False,
                callbacks=[stop], verbose=0).history
    th = train(tm, X_tr, y_tr, X_va, y_va, epochs=epochs, batch_size=batch, learning_rate=1e-3,
               patience=patience, min_delta=min_delta, device=torch.device("cpu"),
               rng=np.random.default_rng(0), verbose=0, shuffle=False)

    assert len(kh["val_loss"]) == len(th["val_loss"])
    np.testing.assert_allclose(th["loss"], kh["loss"], rtol=2e-3)
    np.testing.assert_allclose(th["val_loss"], kh["val_loss"], rtol=2e-3)
    # Accuracy flips for rows sitting on a decision boundary; a few of 100 is float noise.
    np.testing.assert_allclose(th["val_accuracy"], kh["val_accuracy"], atol=0.05)
    print("max rel diff loss %.1e, val_loss %.1e" % (
        np.max(np.abs(np.array(th["loss"]) / kh["loss"] - 1)),
        np.max(np.abs(np.array(th["val_loss"]) / kh["val_loss"] - 1))))
    pk = km.predict(X_te, verbose=0)
    pt = predict_proba(tm, X_te, torch.device("cpu"))
    print("restored epoch", th["best_epoch"], "max |p diff|", np.max(np.abs(pk - pt)))
    assert np.max(np.abs(pk - pt)) < 5e-3
    assert np.mean(pk.argmax(1) == pt.argmax(1)) > 0.99


def test_dropout_noise_matches_keras():
    """Same weights, training mode (dropout on): the distribution of outputs over
    dropout masks must agree. Catches dropout placed differently or a different rate,
    which the dropout-free tests above cannot see."""
    _, km, tm = _pair(seed=6)
    rng = np.random.default_rng(6)
    X = rng.random((32, D)).astype("float32")
    n_mc = 400
    keras.utils.set_random_seed(7)
    pk = np.stack([np.asarray(km(X, training=True)) for _ in range(n_mc)])
    torch.manual_seed(7)
    tm.train()
    with torch.no_grad():
        pt = np.stack([torch.softmax(tm(torch.from_numpy(X)), 1).numpy() for _ in range(n_mc)])
    se = np.sqrt((pk.var(0) + pt.var(0)) / n_mc) + 1e-6
    z = (pk.mean(0) - pt.mean(0)) / se
    assert np.abs(z).max() < 5, np.abs(z).max()           # 96 means, Bonferroni-ish bound
    ratio = pt.std(0).mean() / pk.std(0).mean()
    assert 0.9 < ratio < 1.1, ratio
    print(f"max |z| {np.abs(z).max():.2f}, sd ratio torch/keras {ratio:.3f}")


def test_early_stopping_decisions_match_keras():
    """Drive Keras' own EarlyStopping callback and ours with the same val_loss sequences."""
    class Dummy:
        stop_training = False

        def get_weights(self):
            return [self.epoch]

        def set_weights(self, w):
            self.restored = w[0]

    rng = np.random.default_rng(3)
    for trial in range(50):
        losses = list(np.cumsum(rng.normal(-0.002, 0.01, 40)) + 1.0)
        cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-3,
                                           restore_best_weights=True)
        dummy = Dummy()
        cb.set_model(dummy)
        cb.on_train_begin()
        keras_stop = None
        for e, v in enumerate(losses):
            dummy.epoch = e
            cb.on_epoch_end(e, {"val_loss": v})
            if dummy.stop_training:
                keras_stop = e
                break
        cb.on_train_end()

        ours = EarlyStopping(5, 1e-3)
        our_stop = None
        for e, v in enumerate(losses):
            if ours.update(e, v, lambda e=e: e):
                our_stop = e
                break
        assert keras_stop == our_stop, trial
        assert dummy.restored == ours.best_state, trial
