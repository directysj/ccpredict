"""The one-stage regressor: one network, N/C ratio out.

This branch has no classification stage. The published model binned the ratio
with a CNN and then regressed from the class probabilities with a random
forest; ``DirectRegressor`` replaces both with a single network trained on the
ratio itself (see NetConfig in config.py for the heads and encoders, and cnn.py
for the training loop). The two-stage code lives on branch ``ccp_torch``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from typing import Optional

import numpy as np

from .config import NetConfig

log = logging.getLogger(__name__)


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
                          temporal_kernel=cfg.temporal_kernel, pool=cfg.pool,
                          transformer_dim=cfg.transformer_dim,
                          transformer_heads=cfg.transformer_heads,
                          transformer_layers=cfg.transformer_layers,
                          transformer_ff=cfg.transformer_ff,
                          transformer_attention=cfg.transformer_attention)


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
