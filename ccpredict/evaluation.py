"""Metrics and figures (In [24], [38]-[41] and the unexecuted plotting cells).

Numbers are computed exactly as in the notebook. Three plot labels the
notebook got wrong are corrected: the RFECV x-axis now starts at
min_features_to_select (not 1), its y-axis says negative MSE (not accuracy),
and the regression plots name the random forest (not CNN+DNN).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Accuracy, weighted precision/recall/F1 and row-normalised confusion matrix.

    Precision uses zero_division=np.nan as in the notebook. sklearn then drops
    classes that were never predicted from the weighted average, so precision
    is averaged only over predicted classes and is biased upward. Per-class
    values and support are returned so this is visible.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, precision_score, recall_score)

    labels = list(range(num_classes))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=np.nan)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted",
                                                    zero_division=np.nan)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {
            "precision": [float(v) for v in p], "recall": [float(v) for v in r],
            "f1": [float(v) for v in f], "support": [int(v) for v in support],
            "n_predicted": [int(np.sum(y_pred == c)) for c in labels],
        },
        "confusion_matrix_true_normalized": confusion_matrix(
            y_true, y_pred, labels=labels, normalize="true").tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _log_ratio_axis(ax):
    from matplotlib import ticker

    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))


def plot_label_distributions(y_train: np.ndarray, y_test: np.ndarray,
                             train_name: str = "Training", test_name: str = "Test"):
    """Histograms of the N/C ratio, training (trimmed) vs test (In [24])."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for ax, values, title in ((axs[0], y_train, train_name), (axs[1], y_test, test_name)):
        ax.hist(values, bins=100, density=True)
        _log_ratio_axis(ax)
        ax.set_xlabel("N/C Ratio")
        ax.set_ylabel("Density")
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(conf_matrix, title: str = "Confusion Matrix: Pred. vs Act. N/C ratio of Test Set"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(9, 6))
    ConfusionMatrixDisplay(np.asarray(conf_matrix)).plot(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_training_history(history: dict):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    for ax, key, title in ((axs[0], "accuracy", "Model accuracy"), (axs[1], "loss", "Model loss")):
        ax.plot(history.get(key, []))
        ax.plot(history.get(f"val_{key}", []))
        ax.set_title(title)
        ax.set_xlabel("No of Epochs")
        ax.set_ylabel(key.capitalize())
        ax.legend(["Train", "Validation"])
    fig.tight_layout()
    return fig


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, title: str,
                    kde_max_points: Optional[int] = None, seed: int = 0):
    """Predicted vs actual, plain and KDE-coloured (unexecuted cell after the RFECV fit).

    The KDE is O(N^2) in the number of points. kde_max_points subsamples the
    points used to fit the KDE (density is still evaluated at every point);
    None uses all points, as in the notebook.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    m = regression_metrics(y_true, y_pred)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].scatter(y_true, y_pred, s=1)
    lo, hi = y_true.min(), y_true.max()
    axs[0].plot([lo, hi], [lo, hi], "r--")
    axs[0].set_xlabel("Actual N/C Ratio")
    axs[0].set_ylabel("Predicted N/C Ratio")
    axs[0].set_title(title)
    axs[0].text(0.05, 0.95, f"MSE: {m['mse']:.3f}, R: {m['pearson_r']:.3f}, $R^2$: {m['r2']:.3f}",
                transform=axs[0].transAxes, verticalalignment="top", horizontalalignment="left")

    xy = np.vstack([y_true, y_pred])
    fit_xy = xy
    if kde_max_points is not None and xy.shape[1] > kde_max_points:
        pick = np.random.default_rng(seed).choice(xy.shape[1], kde_max_points, replace=False)
        fit_xy = xy[:, pick]
    z = gaussian_kde(fit_xy)(xy)
    axs[1].scatter(y_true, y_pred, c=z, s=1)
    axs[1].tricontour(y_true, y_pred, z, levels=14, linewidths=0.5, colors="k")
    filled = axs[1].tricontourf(y_true, y_pred, z, levels=14, cmap="RdBu_r")
    fig.colorbar(filled, ax=axs[1], label="Density")
    axs[1].set_xlabel("Actual N/C Ratio")
    axs[1].set_ylabel("Predicted N/C Ratio")
    axs[1].set_title(title)
    fig.tight_layout()
    return fig


def plot_rfecv_scores(rfecv):
    """Cross-validated score vs number of selected features (last unexecuted plotting cell, axis labels corrected)."""
    import matplotlib.pyplot as plt

    results = rfecv.cv_results_
    n_features = np.asarray(results["n_features"]) if "n_features" in results else (
        np.arange(len(results["mean_test_score"])) + rfecv.min_features_to_select)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(n_features, results["mean_test_score"], yerr=results["std_test_score"])
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel(f"Mean CV score ({rfecv.scoring})")
    ax.set_title("RFECV feature elimination")
    fig.tight_layout()
    return fig


def save_figure(fig, path: str, dpi: int = 400) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def as_jsonable(obj):
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


__all__: Sequence[str] = (
    "classification_metrics", "regression_metrics", "plot_label_distributions",
    "plot_confusion_matrix", "plot_training_history", "plot_regression", "plot_rfecv_scores",
    "save_figure",
)
