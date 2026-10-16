"""Networks and the training loop, in PyTorch.

Two things live here. First, the notebook's Keras stage-1 classifier reproduced
exactly (the table below). Second, the switches that turn the same code into
the fixed models the benchmark compares against it: an MLP encoder instead of a
convolution along the unordered latent index, BatchNorm before the activation,
and regression or histogram heads instead of three class bins (see NetConfig in
config.py for why each exists). The training loop is shared: what changes is the
``Objective``, which owns the target encoding, the loss and the prediction.

Each Keras default the notebook relied on is reproduced explicitly, because
PyTorch's defaults differ and each difference changes the trained network:

  Keras (notebook)                              PyTorch default -> what is done here
  Glorot-uniform kernels, zero biases           Kaiming-uniform, random bias -> Glorot, zeros
  BatchNorm momentum 0.99, eps 1e-3,            momentum 0.1 (=0.9), eps 1e-5, unbiased
    biased batch variance for moving stats        running var -> KerasBatchNorm1d
  LeakyReLU slope 0.3                           0.01 -> 0.3
  kernel_regularizer l1/l2: lambda*sum|w|,      no equivalent (weight_decay is coupled L2)
    lambda*sum w^2 added to the loss              -> explicit penalty in the loss
  val_loss includes the penalties               -> same
  Adam: w -= lr*sqrt(1-b2^t)/(1-b1^t) * m /     eps inside the bias correction
    (sqrt(v) + 1e-7)                              -> KerasAdam
  validation_split = last fraction of arrays    no such thing -> taken explicitly
  shuffle every epoch, last partial batch kept  -> same
  EarlyStopping(val_loss, min_delta, patience,  -> same rule; best weights always restored
    restore_best_weights) in Keras 3              at the end, as Keras 3 does

The input (d,) is treated as a 1-channel sequence of length d; Keras' Conv2D
with a (3, 1) kernel on (d, 1, 1) is the same operation. Flattening is done
channels-last, as Keras does, so weights can be copied between the two.

Floating-point kernels still differ between frameworks, so results are
statistically, not bitwise, equivalent; tests/parity checks both levels.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KerasBatchNorm1d(nn.Module):
    """BatchNorm over (N, C, L) with Keras semantics (momentum on the old value, biased variance)."""

    def __init__(self, num_features: int, momentum: float = 0.99, eps: float = 1e-3):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2))
            var = x.var(dim=(0, 2), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean, alpha=1 - self.momentum)
                self.running_var.mul_(self.momentum).add_(var, alpha=1 - self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        inv = torch.rsqrt(var + self.eps) * self.weight
        return (x - mean[None, :, None]) * inv[None, :, None] + self.bias[None, :, None]


class FeedForwardNet(nn.Module):
    """The notebook's create_model, with the departures from it as explicit switches.

    Default (encoder="conv", norm_position="after_activation", conv_bias=True) is the
    notebook: 2 x [conv, LeakyReLU, BN, maxpool, dropout], 2 x [dense, LeakyReLU,
    dropout], linear output. forward returns logits (or, for a regression head, the
    raw output); the Objective turns them into predictions.

    encoder="mlp" drops the convolutions and feeds the features straight to the dense
    layers. norm_position="before_activation" normalises before LeakyReLU instead of
    after; the conv bias is then provably redundant, so conv_bias=False goes with it.
    """

    def __init__(self, input_dim: int, out_features: int, conv_filters=(32, 64),
                 dense_units=(128, 64), dropout: float = 0.2,
                 regularizers=(("l2", 0.01), ("l1", 0.02)), negative_slope: float = 0.3,
                 encoder: str = "conv", norm_position: str = "after_activation",
                 conv_bias: bool = True):
        super().__init__()
        if encoder not in ("conv", "mlp"):
            raise ValueError(f"encoder must be 'conv' or 'mlp', not {encoder!r}")
        if norm_position not in ("before_activation", "after_activation"):
            raise ValueError(f"unknown norm_position {norm_position!r}")
        self.negative_slope = negative_slope
        self.encoder, self.norm_position = encoder, norm_position
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        width = input_dim
        if encoder == "conv":
            channels, length = 1, input_dim
            for f in conv_filters:
                self.convs.append(nn.Conv1d(channels, f, kernel_size=3, padding=1, bias=conv_bias))
                self.norms.append(KerasBatchNorm1d(f))
                channels, length = f, length // 2
            width = channels * length
        self.denses = nn.ModuleList()
        for u in dense_units:
            self.denses.append(nn.Linear(width, u))
            width = u
        self.out = nn.Linear(width, out_features)
        self.drop = nn.Dropout(dropout)
        # As in the notebook: regulariser k % len(regs) for the k-th conv, and again
        # from the start for the dense layers (conv1 l2, conv2 l1, dense1 l2, dense2 l1).
        regs = list(regularizers)
        self._penalties = ([(c, regs[i % len(regs)]) for i, c in enumerate(self.convs)]
                           + [(d, regs[i % len(regs)]) for i, d in enumerate(self.denses)])
        for m in list(self.convs) + list(self.denses) + [self.out]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder == "conv":
            x = x.unsqueeze(1)                                     # (N, 1, d)
            for conv, norm in zip(self.convs, self.norms):
                if self.norm_position == "after_activation":
                    x = norm(F.leaky_relu(conv(x), self.negative_slope))   # notebook order
                else:
                    x = F.leaky_relu(norm(conv(x)), self.negative_slope)
                x = self.drop(F.max_pool1d(x, 2))
            x = x.transpose(1, 2).reshape(x.shape[0], -1)          # channels-last flatten, like Keras
        for dense in self.denses:
            x = self.drop(F.leaky_relu(dense(x), self.negative_slope))
        return self.out(x)

    def penalty(self) -> torch.Tensor:
        """Sum of the kernel regularisers, exactly as Keras adds them to the loss."""
        total = torch.zeros((), device=self.out.weight.device)
        for layer, (kind, lam) in self._penalties:
            w = layer.weight
            total = total + (lam * w.abs().sum() if kind == "l1" else lam * (w * w).sum())
        return total


# ----------------------------------------------------------------------------
# Objectives: what the network is asked to predict
# ----------------------------------------------------------------------------

class Objective:
    """Target encoding, loss and prediction for one kind of task.

    ``encode`` maps labels to what the network is trained against, ``loss`` is
    added to the weight penalty, ``predict`` maps outputs back to the quantity
    being reported, and ``metric`` is logged per epoch (early stopping always
    watches the loss, never the metric).
    """

    out_features: int = 1
    target_dtype = torch.float32
    metric_name: str = "metric"

    def encode(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss(self, out: torch.Tensor, t: torch.Tensor, weights=None,
             reduction: str = "mean") -> torch.Tensor:
        raise NotImplementedError

    def predict(self, out: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def metric(self, out: torch.Tensor, t: torch.Tensor) -> float:
        raise NotImplementedError

    @staticmethod
    def _reduce(per_row: torch.Tensor, weights, reduction: str) -> torch.Tensor:
        if weights is not None:
            # Mean of w*loss / mean of w, so the scale does not depend on the weights.
            per_row = per_row * weights
            return per_row.sum() if reduction == "sum" else per_row.sum() / weights.sum()
        return per_row.sum() if reduction == "sum" else per_row.mean()


class ClassificationObjective(Objective):
    """The notebook's stage 1: softmax cross-entropy over N/C bins."""

    target_dtype = torch.long
    metric_name = "accuracy"

    def __init__(self, num_classes: int):
        self.out_features = num_classes

    def encode(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.int64)

    def loss(self, out, t, weights=None, reduction="mean"):
        return self._reduce(F.cross_entropy(out, t, reduction="none"), weights, reduction)

    def predict(self, out):
        return torch.softmax(out, dim=1).cpu().numpy()

    def metric(self, out, t):
        return float((out.argmax(1) == t).double().mean())


class RegressionObjective(Objective):
    """Predict the N/C ratio itself: one output, squared or Huber loss.

    With transform="log" the network fits log(N/C) and predictions are
    exponentiated back, so metrics stay on the raw ratio either way.
    """

    out_features = 1
    metric_name = "pearson_r"

    def __init__(self, transform: str = "identity", loss: str = "mse", huber_delta: float = 1.0):
        if transform not in ("identity", "log"):
            raise ValueError(f"unknown target_transform {transform!r}")
        if loss not in ("mse", "huber"):
            raise ValueError(f"unknown loss {loss!r}")
        self.transform, self.loss_name, self.huber_delta = transform, loss, huber_delta

    def encode(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.transform == "log":
            if np.any(y <= 0):
                raise ValueError("target_transform='log' needs strictly positive labels")
            y = np.log(y)
        return y.astype(np.float32)

    def decode(self, t: np.ndarray) -> np.ndarray:
        return np.exp(t) if self.transform == "log" else t

    def loss(self, out, t, weights=None, reduction="mean"):
        pred = out[:, 0]
        if self.loss_name == "mse":
            per_row = (pred - t) ** 2
        else:
            per_row = F.huber_loss(pred, t, reduction="none", delta=self.huber_delta)
        return self._reduce(per_row, weights, reduction)

    def predict(self, out):
        return self.decode(out[:, 0].cpu().numpy().astype(np.float64))

    def metric(self, out, t):
        return _pearson(out[:, 0], t)


class HistogramObjective(Objective):
    """Cross-entropy over equal-count bins of the label, predicting the expectation.

    ``edges`` are interior bin edges and ``centers`` the mean label within each
    bin, both from the training rows only. Equal-count bins remove the class
    imbalance by construction, and the expectation sum_k p_k * center_k keeps
    the prediction continuous, so this is a conditional distribution over N/C
    rather than three coarse classes.
    """

    metric_name = "pearson_r"

    def __init__(self, edges: np.ndarray, centers: np.ndarray):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.out_features = self.centers.size
        if self.edges.size != self.out_features - 1:
            raise ValueError("need one interior edge fewer than centers")
        self._centers_t: Optional[torch.Tensor] = None

    @classmethod
    def from_labels(cls, y: np.ndarray, n_bins: int) -> "HistogramObjective":
        y = np.asarray(y, dtype=np.float64)
        edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
        index = np.digitize(y, edges)
        centers = np.array([y[index == k].mean() if np.any(index == k) else 0.0
                            for k in range(edges.size + 1)])
        return cls(edges, centers)

    target_dtype = torch.long

    def encode(self, y: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(y, dtype=np.float64), self.edges).astype(np.int64)

    def loss(self, out, t, weights=None, reduction="mean"):
        return self._reduce(F.cross_entropy(out, t, reduction="none"), weights, reduction)

    def _centers(self, device) -> torch.Tensor:
        if self._centers_t is None or self._centers_t.device != device:
            self._centers_t = torch.as_tensor(self.centers, dtype=torch.float32, device=device)
        return self._centers_t

    def expectation(self, out: torch.Tensor) -> torch.Tensor:
        return torch.softmax(out, dim=1) @ self._centers(out.device)

    def predict(self, out):
        return self.expectation(out).cpu().numpy().astype(np.float64)

    def metric(self, out, t):
        return _pearson(self.expectation(out), self._centers(out.device)[t])


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float(a @ b / denom) if denom > 0 else 0.0


def inverse_density_weights(y: np.ndarray, bins: int = 50, clip: float = 10.0) -> np.ndarray:
    """1 / (label density), scaled to mean 1 and then clipped at ``clip``.

    The regression analogue of oversampling minority classes: rare N/C values
    count for more. After the clip no row counts for more than ``clip`` average
    rows, which keeps a nearly empty bin from dominating the loss. Only the
    relative weights matter: the loss divides by their sum.
    """
    y = np.asarray(y, dtype=np.float64)
    counts, edges = np.histogram(y, bins=bins)
    index = np.clip(np.digitize(y, edges[1:-1]), 0, counts.size - 1)
    w = 1.0 / np.maximum(counts[index], 1)
    return np.minimum(w / w.mean(), clip).astype(np.float32)


class KerasAdam(torch.optim.Optimizer):
    """Adam with Keras' update rule: w -= lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps)."""

    def __init__(self, params, lr: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-7):
        super().__init__(params, dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2, eps, lr = group["beta_1"], group["beta_2"], group["epsilon"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v, g = state["m"], state["v"], p.grad
                alpha = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                m.add_(g - m, alpha=1 - b1)
                v.add_(g * g - v, alpha=1 - b2)
                p.sub_(alpha * m / (v.sqrt() + eps))
        return None


class EarlyStopping:
    """Keras 3 EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True), on_epoch_end."""

    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, abs(min_delta)
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.best_state = None
        self.wait = 0

    def update(self, epoch: int, val_loss: float, snapshot) -> bool:
        """Record this epoch; return True when training should stop.

        snapshot() returns the state to restore and must not return None:
        like Keras, "no state stored yet" is detected by best_state is None."""
        if self.best_state is None:
            self.best_state, self.best_epoch = snapshot(), epoch
        self.wait += 1
        if self.best is None or val_loss + self.min_delta < self.best:
            self.best, self.best_epoch, self.wait = val_loss, epoch, 0
            self.best_state = snapshot()
            return False
        return self.wait >= self.patience and epoch > 0


def _evaluate(model, X: torch.Tensor, y: torch.Tensor, batch_size: int,
              obj: Objective) -> Tuple[float, float]:
    """Keras-style validation: mean loss over all rows + penalty, and the metric."""
    model.eval()
    loss_sum, outs = 0.0, []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            out = model(X[s:s + batch_size])
            loss_sum += obj.loss(out, y[s:s + batch_size], reduction="sum").item()
            outs.append(out)
        pen = model.penalty().item()
        metric = obj.metric(torch.cat(outs), y)
    return loss_sum / X.shape[0] + pen, metric


def train(model: FeedForwardNet, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
          y_val: np.ndarray, *, epochs: int, batch_size: int, learning_rate: float, patience: int,
          min_delta: float, device, rng: np.random.Generator, verbose: int = 1,
          shuffle: bool = True, objective: Optional[Objective] = None,
          weights: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """Keras' fit loop with EarlyStopping(monitor='val_loss', restore_best_weights=True).

    ``y_tr``/``y_val`` are already encoded for ``objective`` (class indices, bin
    indices or transformed values); the default objective is the notebook's
    classification over ``model.out.out_features`` classes. ``weights`` are
    per-training-row loss weights; validation is always unweighted.

    shuffle=False (rows in order every epoch) exists only so the whole loop can
    be compared step for step with Keras' fit(shuffle=False) in tests/parity."""
    obj = objective or ClassificationObjective(model.out.out_features)
    model.to(device)
    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=obj.target_dtype, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=obj.target_dtype, device=device)
    wt = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    opt = KerasAdam(model.parameters(), lr=learning_rate)
    m = obj.metric_name
    history: Dict[str, List[float]] = {"loss": [], m: [], "val_loss": [], f"val_{m}": []}
    stopper = EarlyStopping(patience, min_delta)
    n = Xt.shape[0]
    n_batches = math.ceil(n / batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = torch.as_tensor(rng.permutation(n) if shuffle else np.arange(n), device=device)
        loss_sum, outs = 0.0, []
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            out = model(Xt[idx])
            loss = obj.loss(out, yt[idx], None if wt is None else wt[idx]) + model.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * idx.numel()
            outs.append(out.detach())
        with torch.no_grad():
            train_metric = obj.metric(torch.cat(outs), yt[order])
        val_loss, val_metric = _evaluate(model, Xv, yv, max(batch_size, 1024), obj)
        for k, v in (("loss", loss_sum / n), (m, train_metric), ("val_loss", val_loss),
                     (f"val_{m}", val_metric)):
            history[k].append(float(v))
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - {time.time() - t0:.0f}s - loss: {loss_sum / n:.4f} - "
                  f"{m}: {train_metric:.4f} - val_loss: {val_loss:.4f} - val_{m}: {val_metric:.4f}",
                  flush=True)

        if stopper.update(epoch, val_loss, lambda: copy.deepcopy(model.state_dict())):
            if verbose:
                print(f"Epoch {epoch + 1}: early stopping", flush=True)
            break
    model.load_state_dict(stopper.best_state)                      # Keras 3 restores in any case
    history["best_epoch"] = [stopper.best_epoch + 1]
    return history


def predict(model: FeedForwardNet, X: np.ndarray, device, objective: Optional[Objective] = None,
            batch_size: int = 4096) -> np.ndarray:
    """Run the model over X in batches and map its outputs through ``objective``."""
    obj = objective or ClassificationObjective(model.out.out_features)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[s:s + batch_size], dtype=torch.float32, device=device)
            out.append(obj.predict(model(xb)))
    if out:
        return np.concatenate(out)
    return np.zeros((0, obj.out_features) if obj.out_features > 1 else (0,))


def predict_proba(model: FeedForwardNet, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    """Class probabilities (stage 1 of the notebook pipeline)."""
    return predict(model, X, device, ClassificationObjective(model.out.out_features),
                   batch_size).astype(np.float32)
