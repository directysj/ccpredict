"""Train/test assembly, histogram trimming, scaling, class binning (In [22]-[23], [25], [32])."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import SplitConfig
from .data import LabeledFOV


def trim_by_histogram(values: np.ndarray, num_bins: int, target_density: int,
                      seed: int = 42) -> np.ndarray:
    """Indices of a subsample with at most target_density values per histogram bin.

    Same as the notebook's trim_cc_vals (same pandas sampling, same seed), but
    returns the kept indices in ascending order instead of a DataFrame.
    Bins are linear in the value although the N/C ratio is plotted on a log
    axis, so the low-ratio end is split into fewer bins.
    """
    df = pd.DataFrame(np.asarray(values), columns=["CC"])
    df["original_index"] = df.index
    _, bin_edges = np.histogram(df["CC"], bins=num_bins)
    df["bin"] = pd.cut(df["CC"], bins=bin_edges, labels=False, include_lowest=True)

    kept = []
    for i in range(num_bins):
        bin_data = df[df["bin"] == i]
        if bin_data.shape[0] > target_density:
            bin_data = bin_data.sample(n=target_density, random_state=seed)
        kept.append(bin_data)
    trimmed = pd.concat(kept).sort_values("original_index")
    return trimmed["original_index"].to_numpy(dtype=int)


def discretize(values: np.ndarray, boundaries: Sequence[float]) -> np.ndarray:
    """Class label per value: 0 below boundaries[0], ..., len(boundaries) at or above the last.

    Equivalent to the notebook's class_conditions + np.select. The notebook
    silently put NaN into class 0; here NaN raises.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("NaN in values to discretize")
    return np.digitize(values, np.asarray(boundaries, dtype=float), right=False).astype(int)


def make_oversampler(name: str, strategy: str, seed: int):
    """imblearn oversampler by name; 'none' returns None."""
    if name == "none":
        return None
    from imblearn.over_sampling import ADASYN, SMOTE, KMeansSMOTE, RandomOverSampler

    cls = {"random": RandomOverSampler, "smote": SMOTE,
           "kmeans_smote": KMeansSMOTE, "adasyn": ADASYN}[name]
    return cls(random_state=seed, sampling_strategy=strategy)


@dataclass
class TrainTestData:
    """Scaled, trimmed training set and scaled test set (end state of In [23])."""

    X_train: np.ndarray        # trimmed, MinMax-scaled
    y_train: np.ndarray        # trimmed N/C ratio
    X_test: np.ndarray         # scaled with the training scaler
    y_test: np.ndarray
    scaler: MinMaxScaler
    train_rows: np.ndarray     # rows of the concatenated training FOVs that survived trimming
    train_fovs: tuple
    test_fov: str


def build_train_test(fovs: Dict[str, LabeledFOV], split: SplitConfig) -> TrainTestData:
    missing = [f for f in (*split.train_fovs, split.test_fov) if f not in fovs]
    if missing:
        raise KeyError(f"FOVs not available (failed QC or not loaded): {missing}")

    X_train = np.concatenate([fovs[f].X for f in split.train_fovs], axis=0)
    y_train = np.concatenate([fovs[f].y for f in split.train_fovs], axis=0)

    rows = trim_by_histogram(y_train, split.trim_bins, split.trim_target_density, split.trim_seed)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train[rows, :])
    X_test_scaled = scaler.transform(fovs[split.test_fov].X)

    return TrainTestData(X_train_scaled, y_train[rows], X_test_scaled, fovs[split.test_fov].y,
                         scaler, rows, tuple(split.train_fovs), split.test_fov)
