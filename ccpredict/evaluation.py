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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def per_trajectory_r(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                     min_len: int = 20) -> np.ndarray:
    """Pearson R between actual and predicted N/C within each trajectory (paper Fig. 6).

    Only trajectories with at least min_len labelled rows and non-constant
    actual and predicted values are scored.
    """
    order = np.argsort(groups, kind="stable")
    g, yt, yp = groups[order], y_true[order], y_pred[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], g.size]
    out = []
    for s, e in zip(starts, ends):
        if e - s >= min_len and yt[s:e].std() > 1e-12 and yp[s:e].std() > 1e-12:
            out.append(np.corrcoef(yt[s:e], yp[s:e])[0, 1])
    return np.asarray(out)


def _group_sums(y_true, y_pred, groups):
    """Per-trajectory sufficient statistics for R, MSE and R^2."""
    _, inv = np.unique(groups, return_inverse=True)
    cols = [np.ones_like(y_true), y_true, y_pred, y_true ** 2, y_pred ** 2, y_true * y_pred,
            (y_true - y_pred) ** 2]
    return np.stack([np.bincount(inv, weights=c) for c in cols], axis=1)


def _stats_from_sums(t):
    n, sy, sp, syy, spp, syp, se = t.T if t.ndim == 2 else t
    r = (n * syp - sy * sp) / np.sqrt((n * syy - sy ** 2) * (n * spp - sp ** 2))
    mse = se / n
    r2 = 1 - se / (syy - sy ** 2 / n)
    return r, mse, r2


def trajectory_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                         n_boot: int = 200, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile CIs for R, MSE and R^2, resampling whole trajectories.

    Rows of one trajectory are strongly correlated (adjacent frames of one
    cell), so a row-level bootstrap would give intervals that are far too narrow.
    """
    sums = _group_sums(y_true, y_pred, groups)
    rng = np.random.default_rng(seed)
    T = sums.shape[0]
    counts = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(n_boot)])
    r, mse, r2 = _stats_from_sums(counts @ sums)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]
            for k, v in (("pearson_r", r), ("mse", mse), ("r2", r2))} | {"n_trajectories": int(T)}


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, groups: Optional[np.ndarray] = None,
                      *, min_traj_len: int = 20, n_boot: int = 200, seed: int = 0) -> dict:
    """Everything used to compare models on one test FOV.

    Raw-scale MSE and Pearson R are the paper's metrics. Log-scale metrics
    treat a 2x error the same at N/C 1 and N/C 2, which matches a ratio
    readout. Calibration slope: regression of actual on predicted (1 = no
    shrinkage; >1 = predictions too compressed).
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    # A constant prediction (e.g. the training mean) has no correlation; report NaN, not noise.
    constant = np.ptp(y_pred) <= 1e-12 * max(1.0, float(np.abs(y_pred).max()))
    with np.errstate(all="ignore"):
        out = regression_metrics(y_true, y_pred)
        lt, lp = np.log(y_true), np.log(np.clip(y_pred, 1e-6, None))
        # ddof=1 to match np.cov below, which uses it; mixing the two biased the
        # slope by n/(n-1).
        var_p = np.var(y_pred, ddof=1)
        out.update({
            "n": int(y_true.size),
            "spearman": float("nan") if constant else float(spearmanr(y_true, y_pred)[0]),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "mae_log": float(np.mean(np.abs(lt - lp))),
            "r2_log": float(1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)),
            "mean_bias": float(np.mean(y_pred - y_true)),
            "calibration_slope": float("nan") if constant else float(np.cov(y_true, y_pred)[0, 1] / var_p),
        })
    if constant:
        out["pearson_r"] = float("nan")
    if groups is not None and not constant:
        traj_r = per_trajectory_r(y_true, y_pred, groups, min_traj_len)
        out["per_trajectory"] = {
            "n_scored": int(traj_r.size), "min_len": min_traj_len,
            "median_r": float(np.median(traj_r)) if traj_r.size else float("nan"),
            "frac_r_gt_0.5": float(np.mean(traj_r > 0.5)) if traj_r.size else float("nan"),
        }
        with np.errstate(all="ignore"):
            out["ci95"] = trajectory_bootstrap(y_true, y_pred, groups, n_boot=n_boot, seed=seed)
    return out


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
    "regression_metrics", "regression_report", "per_trajectory_r", "trajectory_bootstrap",
    "plot_training_history", "plot_regression", "save_figure", "as_jsonable",
)
