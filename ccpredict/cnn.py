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


def _disable_attention(block: nn.Module) -> None:
    """Make one TransformerEncoderLayer's attention sublayer the exact identity.

    The output projection is zeroed and frozen, so ``self_attn`` returns exactly
    zero for every input and, with pre-norm residuals ``h = h + attn(norm(h))``,
    contributes nothing. No gradient reaches the query/key/value projections
    either (they are multiplied by a zero matrix), so they stay at their
    initialisation. The module itself is left in place rather than replaced,
    because nn.TransformerEncoder introspects ``self_attn`` and a stand-in has
    to mimic attributes that differ between torch versions.

    Everything else in the block -- token projection, both LayerNorms, the
    feed-forward sublayer -- is untouched. That is what prices attention
    itself: hold the rest of the encoder fixed and see what the query-key
    machinery adds on top of it (see transformer_attention in config.py).
    """
    with torch.no_grad():
        block.self_attn.out_proj.weight.zero_()
        if block.self_attn.out_proj.bias is not None:
            block.self_attn.out_proj.bias.zero_()
    block.self_attn.requires_grad_(False)


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


ENCODERS = ("conv", "mlp", "snapshot_shared", "temporal_conv", "transformer")
# Encoders that read the input as a sequence of snapshots and so need ``steps``.
SEQUENCE_ENCODERS = ("snapshot_shared", "temporal_conv", "transformer")
POOLS = ("concat", "mean", "attention", "last")


class FeedForwardNet(nn.Module):
    """The notebook's create_model, with the departures from it as explicit switches.

    A row of features is ``trajl`` snapshots of the same cell concatenated:
    ``[snapshot 0 features | snapshot 1 features | ...]``, nf features each
    (data.py builds it that way, and benchmark.column_selector keeps that
    stride). The nf axis is a set of channels with no order; the ``steps`` axis
    is time, 15 min apart. Which of the two an encoder works along is the whole
    difference between these options:

    encoder="conv" (the notebook, the default)
        Conv1d with a width-3 kernel along the *flattened* vector, so it mixes
        three neighbouring entries of ``[s0 | s1 | ...]``. At trajl=8, nf=256
        only 14 of 2048 kernel positions straddle a snapshot boundary: 99.3% of
        the kernel mixes arbitrary feature dimensions *within* one snapshot.
        Kept bit-exact for comparison, not because it is sensible.
    encoder="mlp"
        No convolution; the flat vector goes straight to the dense layers.
    encoder="snapshot_shared"
        One MLP (``snapshot_hidden``) applied to every snapshot with the weights
        shared across time, then ``pool``. Weight sharing along the only axis
        that has an order, and ``steps`` times fewer parameters than the flat
        net for the same width.
    encoder="temporal_conv"
        Conv1d along *time* with nf input channels (``temporal_filters``,
        ``temporal_kernel``), i.e. the notebook's idea applied to the axis where
        neighbours are neighbours. No pooling over time: the axis is short.
    encoder="transformer"
        Each snapshot is one token: ``nf`` features projected to
        ``transformer_dim`` and added to a *learned* positional embedding, then
        ``transformer_layers`` pre-norm blocks of multi-head self-attention plus
        a feed-forward sublayer. Attention runs over the ``steps`` snapshots of
        the snippet, so every snapshot can read every other one; the two
        encoders above cannot do that (``snapshot_shared`` never mixes
        snapshots at all, ``temporal_conv`` only mixes neighbours within its
        kernel). Positions are learned rather than sinusoidal because ``steps``
        is small and fixed, and absolute embeddings over 8 slots already span
        every relative offset the head could want.

        Two caveats worth knowing before reading a comparison. The sequence is
        8 tokens long, which is short enough that pairwise attention has little
        to find that an attention *pool* over a shared encoder cannot;
        ``pool="attention"`` on ``snapshot_shared`` is the cheap version of the
        same idea and the honest baseline to beat. And the self-attention
        internals keep PyTorch's own initialisation and carry no l1/l2 penalty
        (only the token projection does), because Keras-style per-kernel
        penalties have no standard counterpart inside a transformer block. The
        feed-forward sublayer uses GELU, as transformers do, not the LeakyReLU
        the rest of this file keeps for Keras parity; the dense head after the
        pool is unchanged.

    ``pool`` reduces the per-step representation for the dense head: "concat"
    keeps the order (a weight per step), "mean" discards it, "attention" learns
    one weight per step from its own content, and "last" takes the final step
    only. "last" is the right default when the label belongs to the last
    snapshot of the snippet, which is how this pipeline defines it: that step's
    representation has already seen the earlier ones through the receptive
    field, and the head needs a weight per feature instead of per feature and
    step. "mean" is the wrong choice for such a label, since it dilutes the
    snapshot the label is about. Both sequence encoders end in the same pool and
    head, so a comparison between them is a comparison of the encoder alone.

    norm_position="before_activation" normalises before LeakyReLU instead of
    after; the conv bias is then provably redundant, so conv_bias=False goes with it.
    forward returns logits (or, for a regression head, the raw output).
    """

    def __init__(self, input_dim: int, out_features: int, conv_filters=(32, 64),
                 dense_units=(128, 64), dropout: float = 0.2,
                 regularizers=(("l2", 0.01), ("l1", 0.02)), negative_slope: float = 0.3,
                 encoder: str = "conv", norm_position: str = "after_activation",
                 conv_bias: bool = True, steps: int = 1, snapshot_hidden=(64,),
                 temporal_filters=(64, 64), temporal_kernel: int = 3, pool: str = "concat",
                 transformer_dim: int = 64, transformer_heads: int = 4,
                 transformer_layers: int = 2, transformer_ff: int = 128,
                 transformer_attention: str = "self"):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, not {encoder!r}")
        if norm_position not in ("before_activation", "after_activation"):
            raise ValueError(f"unknown norm_position {norm_position!r}")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, not {pool!r}")
        self.negative_slope = negative_slope
        self.encoder, self.norm_position, self.pool = encoder, norm_position, pool
        self.steps, self.features_per_step = 1, input_dim
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        self.snapshot, self.tconvs, self.tnorms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.attention = None
        self.token = self.blocks = self.final_norm = None
        self.positions = None
        width = input_dim

        if encoder in SEQUENCE_ENCODERS:
            if steps < 1:
                raise ValueError(f"steps must be at least 1, got {steps}")
            if input_dim % steps:
                raise ValueError(
                    f"encoder={encoder!r} reads the input as {steps} snapshots, but {input_dim} "
                    f"features do not divide by {steps}. Pass steps=trajl, and note that "
                    "features.append_com appends its columns after all the snapshot blocks, "
                    "which breaks the layout these encoders rely on.")
            self.steps = steps
            self.features_per_step = nf = input_dim // steps
            if encoder == "snapshot_shared":
                w = nf
                for u in snapshot_hidden:
                    self.snapshot.append(nn.Linear(w, u))
                    w = u
            elif encoder == "transformer":
                if transformer_dim % transformer_heads:
                    raise ValueError(
                        f"transformer_dim ({transformer_dim}) must divide by transformer_heads "
                        f"({transformer_heads}): each head takes dim/heads of the embedding.")
                if transformer_layers < 1:
                    raise ValueError(f"transformer_layers must be at least 1, "
                                     f"got {transformer_layers}")
                w = transformer_dim
                self.token = nn.Linear(nf, w)                  # one token per snapshot
                # Learned absolute positions: ``steps`` is small, fixed and known here.
                self.positions = nn.Parameter(torch.zeros(steps, w))
                layer = nn.TransformerEncoderLayer(
                    d_model=w, nhead=transformer_heads, dim_feedforward=transformer_ff,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, num_layers=transformer_layers,
                                                    enable_nested_tensor=False)
                if transformer_attention not in ("self", "none"):
                    raise ValueError("transformer_attention must be 'self' or 'none', "
                                     f"not {transformer_attention!r}")
                if transformer_attention == "none":     # ablation: keep the block, drop attention
                    for block in self.blocks.layers:
                        _disable_attention(block)
                self.final_norm = nn.LayerNorm(w)
            else:
                channels = nf
                for f in temporal_filters:
                    self.tconvs.append(nn.Conv1d(channels, f, kernel_size=temporal_kernel,
                                                 padding=temporal_kernel // 2, bias=conv_bias))
                    self.tnorms.append(KerasBatchNorm1d(f))
                    channels = f
                w = channels
            width = w * steps if pool == "concat" else w     # mean/attention/last keep one step's width
            if pool == "attention":
                self.attention = nn.Linear(w, 1)
        elif encoder == "conv":
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
        # As in the notebook: regulariser k % len(regs) for the k-th encoder layer, and
        # again from the start for the dense layers (conv1 l2, conv2 l1, dense1 l2,
        # dense2 l1). An empty tuple means no penalty at all.
        regs = list(regularizers)
        encoder_layers = (list(self.convs) + list(self.snapshot) + list(self.tconvs)
                          + ([self.token] if self.token is not None else []))
        self._penalties = ([] if not regs else
                           [(c, regs[i % len(regs)]) for i, c in enumerate(encoder_layers)]
                           + [(d, regs[i % len(regs)]) for i, d in enumerate(self.denses)])
        for m in encoder_layers + list(self.denses) + [self.out] + (
                [self.attention] if self.attention is not None else []):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.positions is not None:
            # Small, not zero: identical positions would leave the stack
            # permutation-equivariant at init and the order would have to be
            # learned out of the attention weights alone.
            nn.init.normal_(self.positions, std=0.02)

    def _per_step(self, x: torch.Tensor) -> torch.Tensor:
        """(N, steps * nf) -> (N, steps, width): one representation per snapshot."""
        x = x.reshape(x.shape[0], self.steps, self.features_per_step)
        if self.encoder == "snapshot_shared":
            # nn.Linear acts on the last axis, so every snapshot goes through the
            # same weights: sharing along time, not a separate net per snapshot.
            for lin in self.snapshot:
                x = self.drop(F.leaky_relu(lin(x), self.negative_slope))
            return x
        if self.encoder == "transformer":
            # (N, steps, nf) -> (N, steps, d_model). Attention is taken over dim 1,
            # the snapshots of the snippet, so token t can read every other token.
            x = self.drop(self.token(x) + self.positions)
            return self.final_norm(self.blocks(x))
        x = x.transpose(1, 2)                                      # (N, nf, steps): nf = channels
        for conv, norm in zip(self.tconvs, self.tnorms):
            if self.norm_position == "after_activation":
                x = norm(F.leaky_relu(conv(x), self.negative_slope))
            else:
                x = F.leaky_relu(norm(conv(x)), self.negative_slope)
            x = self.drop(x)
        return x.transpose(1, 2)                                   # (N, steps, filters)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "concat":
            return x.reshape(x.shape[0], -1)
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "last":
            return x[:, -1]                                         # the snapshot the label is for
        weights = torch.softmax(self.attention(x), dim=1)           # (N, steps, 1)
        return (x * weights).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder in SEQUENCE_ENCODERS:
            x = self._pool(self._per_step(x))
        elif self.encoder == "conv":
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

    Where each one runs matters, because Apple's MPS backend has no float64:

      loss     on the device, in the model's dtype (float32)
      predict  a tensor already moved to the CPU
      metric   numpy arrays, so correlations can use float64 exactly

    Nothing here may ask a non-CPU tensor for float64.
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

    def metric(self, out: np.ndarray, t: np.ndarray) -> float:
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
        return torch.softmax(out, dim=1).numpy()

    def metric(self, out, t):
        return float(np.mean(out.argmax(1) == t))


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
        return self.decode(out[:, 0].numpy().astype(np.float64))

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
        # Equal-count bins, so a bin is empty only for a degenerate label
        # distribution; fall back to the middle of the bin, never 0.
        padded = np.concatenate([[2 * edges[0] - edges[-1]] if edges.size else [y.min() - 1],
                                 edges, [2 * edges[-1] - edges[0]] if edges.size else [y.max() + 1]])
        centers = np.array([y[index == k].mean() if np.any(index == k)
                            else 0.5 * (padded[k] + padded[k + 1])
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

    def expectation_np(self, out: np.ndarray) -> np.ndarray:
        """The same expectation in numpy (float64), for the metric."""
        z = np.asarray(out, dtype=np.float64)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return (z / z.sum(axis=1, keepdims=True)) @ self.centers

    def predict(self, out):
        return self.expectation(out).numpy().astype(np.float64)

    def metric(self, out, t):
        # The bin centre stands in for the label, which is all the encoded target keeps.
        return _pearson(self.expectation_np(out), self.centers[t])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson R in float64 on the host (MPS has no float64; see Objective)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt(a @ a) * np.sqrt(b @ b)
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


def _evaluate(model, X: torch.Tensor, y: torch.Tensor, y_np: np.ndarray, batch_size: int,
              obj: Objective) -> Tuple[float, float]:
    """Keras-style validation: mean loss over all rows + penalty, and the metric.

    The loss runs on the device; the metric gets the outputs as numpy on the
    host, because it needs float64 and MPS has none."""
    model.eval()
    loss_sum, outs = 0.0, []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            out = model(X[s:s + batch_size])
            loss_sum += obj.loss(out, y[s:s + batch_size], reduction="sum").item()
            outs.append(out.cpu().numpy())
        pen = model.penalty().item()
    return loss_sum / X.shape[0] + pen, obj.metric(np.concatenate(outs), y_np)


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
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    obj = objective or ClassificationObjective(model.out.out_features)
    model.to(device)
    # Cast on the host first: MPS has no float64, so a float64 array must never
    # be handed to the device, not even for it to convert.
    y_tr_np, y_val_np = np.asarray(y_tr), np.asarray(y_val)
    Xt = torch.as_tensor(np.asarray(X_tr, dtype=np.float32), device=device)
    yt = torch.as_tensor(y_tr_np, dtype=obj.target_dtype, device=device)
    Xv = torch.as_tensor(np.asarray(X_val, dtype=np.float32), device=device)
    yv = torch.as_tensor(y_val_np, dtype=obj.target_dtype, device=device)
    wt = (None if weights is None
          else torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device))
    opt = KerasAdam(model.parameters(), lr=learning_rate)
    m = obj.metric_name
    history: Dict[str, List[float]] = {"loss": [], m: [], "val_loss": [], f"val_{m}": []}
    stopper = EarlyStopping(patience, min_delta)
    n = Xt.shape[0]
    n_batches = math.ceil(n / batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order_np = rng.permutation(n) if shuffle else np.arange(n)
        order = torch.as_tensor(order_np, device=device)
        loss_sum, outs = 0.0, []
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            out = model(Xt[idx])
            loss = obj.loss(out, yt[idx], None if wt is None else wt[idx]) + model.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * idx.numel()
            outs.append(out.detach().cpu().numpy())
        train_metric = obj.metric(np.concatenate(outs), y_tr_np[order_np])
        val_loss, val_metric = _evaluate(model, Xv, yv, y_val_np, max(batch_size, 1024), obj)
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
            xb = torch.as_tensor(np.asarray(X[s:s + batch_size], dtype=np.float32), device=device)
            out.append(obj.predict(model(xb).cpu()))      # Objective.predict expects a CPU tensor
    if out:
        return np.concatenate(out)
    return np.zeros((0, obj.out_features) if obj.out_features > 1 else (0,))


def predict_proba(model: FeedForwardNet, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    """Class probabilities (stage 1 of the notebook pipeline)."""
    return predict(model, X, device, ClassificationObjective(model.out.out_features),
                   batch_size).astype(np.float32)
