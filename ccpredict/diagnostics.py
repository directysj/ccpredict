"""Data and protocol checks the benchmark writes to diagnostics.json.

Each answers a question about how trustworthy an evaluation is, using only
the data, so they run before any model is trained.
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

log = logging.getLogger(__name__)


def trajectory_overlap(trajectories: Sequence[np.ndarray]) -> dict:
    """Do trajectories share cells? If they do, grouping rows by trajectory is not a clean split."""
    if not trajectories:
        return {"n_trajectories": 0}
    cells = np.concatenate([np.asarray(t) for t in trajectories])
    _, counts = np.unique(cells, return_counts=True)
    lengths = np.array([len(t) for t in trajectories])
    return {
        "n_trajectories": int(len(trajectories)),
        "n_cells": int(counts.size),
        "frac_cells_in_more_than_one_trajectory": float(np.mean(counts > 1)),
        "length_quantiles_5_50_95": np.percentile(lengths, [5, 50, 95]).tolist(),
    }


def split_optimism(X: np.ndarray, y: np.ndarray, fold: np.ndarray, groups: np.ndarray,
                   test_sets: Dict[str, tuple], regressor_cfg, n_estimators: int = 50,
                   seed: int = 0) -> dict:
    """How optimistic is each way of validating inside the training FOVs?

    Same forest (paper hyperparameters, fewer trees), scored four ways:
      random_rows   random 80/20 split of rows: adjacent frames of one cell on both sides
      trajectories  80/20 split of whole trajectories
      fovs          leave one training FOV out (mean over FOVs)
      test FOVs     trained on all training rows, scored on each test FOV
    If random_rows >> test FOVs, any number obtained with a random split
    (including the CV inside RFECV) should not be trusted.
    """
    import copy

    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

    from .arms import _forest

    cfg = copy.deepcopy(regressor_cfg)
    cfg.n_estimators = n_estimators

    def r_of(tr, te, Xte=None, yte=None):
        model = _forest(cfg).fit(X[tr], y[tr])
        Xe, ye = (X[te], y[te]) if Xte is None else (Xte, yte)
        return float(np.corrcoef(ye, model.predict(Xe))[0, 1])

    out = {}
    tr, te = next(ShuffleSplit(1, test_size=0.2, random_state=seed).split(X))
    out["random_rows"] = r_of(tr, te)
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))
    out["trajectories"] = r_of(tr, te)
    if np.unique(fold).size > 1:
        out["fovs"] = float(np.mean([r_of(np.flatnonzero(fold != k), np.flatnonzero(fold == k))
                                     for k in np.unique(fold)]))
    allrows = np.arange(y.size)
    for name, (Xte, yte) in test_sets.items():
        out[f"test:{name}"] = r_of(allrows, None, Xte, yte)
    return out
