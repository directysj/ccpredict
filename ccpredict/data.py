"""Tracked FOVs -> QC -> trajectory snippets -> reporter labels (notebook In [4]-[21]).

The two O(n^2) loops in the notebook (snippet building with np.append, and the
per-snippet np.where label lookup) are vectorised here. The tests check that
both give the same arrays, in the same order, as the notebook functions.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import FeatureConfig, PathsConfig, QCConfig

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Loading (In [1], [4]-[7], [15])
# ----------------------------------------------------------------------------

def ensure_celltraj_importable(celltraj_path: str | None) -> None:
    """Put jcTrajectory_CP on sys.path; the pickles reference that module."""
    if celltraj_path and celltraj_path not in sys.path:
        sys.path.append(celltraj_path)


def load_tracked_fov(path: str):
    """Unpickle one jcTrajectory_CP.Trajectory object.

    The notebook calls len(model.linSet) to check lineage tracking exists, and
    calls sys.exit(0) on any failure. Here a failure raises with the cause.
    """
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    try:
        len(model.linSet)
    except AttributeError as exc:
        raise ValueError(f"{path}: tracked object has no linSet (lineage tracking missing)") from exc
    log.info("loaded %s with %d cells", path, model.cells_indSet.size)
    return model


def load_vae_embeddings(path: str, n_cells: int | None = None) -> np.ndarray:
    """Load O2VAE embeddings; row i must be cell index i of the tracked FOV."""
    embeddings = np.load(path)["embeddings"]
    if n_cells is not None and embeddings.shape[0] != n_cells:
        raise ValueError(
            f"{path}: {embeddings.shape[0]} embedding rows but {n_cells} tracked cells; "
            "rows must be indexed by cell index"
        )
    return embeddings


# ----------------------------------------------------------------------------
# QC on cell counts (In [8]-[11])
# ----------------------------------------------------------------------------

def cells_per_frame(cells_frameSet: np.ndarray, n_frames: int) -> np.ndarray:
    """ncells[f] = number of cells in frame f, for f in range(n_frames)."""
    frames = np.asarray(cells_frameSet)
    return np.array([np.sum(frames == f) for f in range(n_frames)], dtype=float)


def cell_count_noise(cells_frameSet: np.ndarray, qc: QCConfig) -> float:
    """Relative std of cell counts around a smoothing spline, over late frames.

    Notebook defect reproduced when qc.evaluate_on_frames is False: the spline
    is evaluated on np.linspace(0, n_frames - 1, 10000) and then indexed with
    frame numbers, so late counts are compared with the smoothed count at
    frames ~2.6-3.7. The score then behaves like a coefficient of variation
    of the late counts, not a tracking-noise measure.
    """
    from csaps import csaps

    frames = np.arange(qc.n_frames)
    ncells = cells_per_frame(cells_frameSet, qc.n_frames)
    if qc.evaluate_on_frames:
        x_eval = frames
    else:
        x_eval = np.linspace(frames[0], frames[-1], qc.n_smooth_points)
    smooth = csaps(frames, ncells / ncells[0], x_eval, smooth=qc.smooth) * ncells[0]
    window = qc.window_frames()
    return float(np.std(ncells[window] - smooth[window]) / np.mean(ncells[window]))


def qc_filter(models: Dict[str, object], qc: QCConfig) -> tuple[list[str], dict[str, float]]:
    """Return (FOVs passing QC in input order, per-FOV noise score).

    Same comparison as the notebook: a FOV is dropped only if score > std_cut,
    so a NaN score is kept.
    """
    scores = {fov: cell_count_noise(m.cells_frameSet, qc) for fov, m in models.items()}
    good = [fov for fov in models if not scores[fov] > qc.std_cut]
    for fov in models:
        if fov not in good:
            log.warning("QC: dropping %s (cell-count noise %.3f > %.3f)", fov, scores[fov], qc.std_cut)
    return good, scores


def clean_handcrafted_features(models: Sequence[object]) -> None:
    """In place: NaN -> 0 in Xf (In [10]). Only matters if Xf columns are used."""
    for m in models:
        m.Xf[np.isnan(m.Xf)] = 0.0


def standardize_com_displacement(models: Sequence[object]) -> None:
    """In place: z-score Xf_com[:, 0] with mean/std pooled over the given FOVs (In [11])."""
    pooled = np.concatenate([m.Xf_com for m in models], axis=0)
    mean, std = np.nanmean(pooled[:, 0]), np.nanstd(pooled[:, 0])
    for m in models:
        m.Xf_com[:, 0] = (m.Xf_com[:, 0] - mean) / std


def extract_trajectories(model) -> list:
    """Run the tracker's get_unique_trajectories() and return a copy of the list (In [14])."""
    model.get_unique_trajectories()
    return list(model.trajectories)


# ----------------------------------------------------------------------------
# Snippets (In [16]-[18])
# ----------------------------------------------------------------------------

def sliding_snippets(
    trajectories: Sequence[np.ndarray], trajl: int, include_last_window: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of length trajl over every trajectory.

    Returns (traj_index, cells): traj_index[j] is the trajectory snippet j came
    from, cells[j] its trajl cell indices. Order matches the notebook's
    get_snippets_with_traj_inds: trajectory by trajectory, window by window.

    With include_last_window=False (notebook) a trajectory of length L yields
    L - trajl windows; with True it yields L - trajl + 1.
    """
    extra = 1 if include_last_window else 0
    owners, windows = [], []
    for k, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        n_win = traj.size - trajl + extra
        if traj.size < trajl or n_win <= 0:
            continue
        windows.append(np.lib.stride_tricks.sliding_window_view(traj, trajl)[:n_win])
        owners.append(np.full(n_win, k, dtype=int))
    if not windows:
        return np.zeros(0, dtype=int), np.zeros((0, trajl), dtype=int)
    return np.concatenate(owners), np.concatenate(windows).astype(int)


@dataclass
class FOVSnippets:
    """All snippets of one FOV with their features and bookkeeping."""

    fov: str
    X: np.ndarray            # (n, n_features): per-cell features concatenated over the snippet
    cells: np.ndarray        # (n, trajl) cell indices
    traj_index: np.ndarray   # (n,) index into the FOV's trajectory list
    start_frame: np.ndarray  # (n,) frame of the first cell
    label_frame: np.ndarray  # (n,) frame at which the reporter label is read
    image_stack: np.ndarray  # (n,) cells_imgfileSet of the first cell

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, rows: np.ndarray) -> "FOVSnippets":
        return FOVSnippets(
            self.fov, self.X[rows], self.cells[rows], self.traj_index[rows],
            self.start_frame[rows], self.label_frame[rows], self.image_stack[rows],
        )


def per_cell_features(model, vae_embeddings: np.ndarray, feat: FeatureConfig) -> np.ndarray:
    """VAE embeddings, optionally with handcrafted columns Xf[:, k:] appended."""
    if feat.handcrafted_from_col is None:
        return vae_embeddings
    return np.concatenate((vae_embeddings, model.Xf[:, feat.handcrafted_from_col:]), axis=1)


def build_fov_snippets(fov: str, model, trajectories: Sequence[np.ndarray],
                       cell_features: np.ndarray, feat: FeatureConfig) -> FOVSnippets:
    """In [18] for one FOV."""
    traj_index, cells = sliding_snippets(trajectories, feat.trajl, feat.include_last_window)
    n = cells.shape[0]
    if n and cells.max() >= cell_features.shape[0]:
        raise ValueError(f"{fov}: snippet references cell {cells.max()} but only "
                         f"{cell_features.shape[0]} feature rows")

    X = cell_features[cells, :].reshape(n, cell_features.shape[1] * feat.trajl)
    if feat.append_com:
        com = model.Xf_com[cells, :].reshape(n, model.Xf_com.shape[1] * feat.trajl)
        X = np.append(X, com, axis=1)
    # The notebook accumulates into a float64 array with np.append, which
    # upcasts float32 embeddings. Kept, so scaling matches bit for bit.
    X = X.astype(np.float64)

    good = np.where(np.sum(np.isnan(X), axis=1) == 0)[0]
    X, cells, traj_index = X[good], cells[good], traj_index[good]

    frame_of = model.cells_frameSet
    start = frame_of[cells[:, 0]].astype(int)
    if feat.label_frame == "start_plus_trajl":
        label_frame = start + feat.trajl
    elif feat.label_frame == "snippet_end":
        label_frame = frame_of[cells[:, -1]].astype(int)
    else:
        raise ValueError(f"unknown label_frame {feat.label_frame!r}")

    return FOVSnippets(fov, X, cells, traj_index, start, label_frame,
                       model.cells_imgfileSet[cells[:, 0]])


# ----------------------------------------------------------------------------
# Reporter labels (In [19]-[21])
# ----------------------------------------------------------------------------

@dataclass
class ReporterTracks:
    """Per-trajectory reporter readouts, NaN N/C entries already removed."""

    n2c: List[np.ndarray]     # nucleus/cytoplasm ratio
    cc: List[np.ndarray]      # cross-correlation values (parsed, unused downstream)
    frames: List[np.ndarray]  # frame index of each readout

    def __len__(self) -> int:
        return len(self.n2c)


def _parse_list_line(line: str, dtype) -> np.ndarray:
    inner = line.strip()[1:-1]
    if not inner.strip():
        return np.array([], dtype=dtype)
    return np.array(inner.split(", "), dtype=dtype)


def read_reporter_file(path: str) -> ReporterTracks:
    """Parse the 3-lines-per-trajectory .dat file (get_cellCycInfoTrajs).

    Line 1: N/C ratios, line 2: CC values, line 3: frames, each as a Python
    list literal. Entries where the N/C ratio is NaN are removed from all three.
    """
    with open(path) as fh:
        lines = fh.readlines()
    if len(lines) % 3:
        log.warning("%s: %d lines is not a multiple of 3; trailing lines ignored", path, len(lines))

    n2c, cc, frames = [], [], []
    for l1, l2, l3 in zip(lines[::3], lines[1::3], lines[2::3]):
        r, c, f = _parse_list_line(l1, float), _parse_list_line(l2, float), _parse_list_line(l3, int)
        mask = ~np.isnan(r)
        n2c.append(r[mask])
        cc.append(c[mask])
        frames.append(f[mask])
    return ReporterTracks(n2c, cc, frames)


def match_labels(traj_index: np.ndarray, label_frame: np.ndarray,
                 reporter: ReporterTracks) -> tuple[np.ndarray, np.ndarray]:
    """For each snippet, the N/C ratio of its trajectory at label_frame.

    Returns (rows, y): the snippet rows that have a reporter value at that
    frame, and the values. First match wins, as in the notebook.

    This assumes trajectory k of the tracked object is trajectory k of the
    reporter file. Nothing in the data enforces that ordering.
    """
    if traj_index.size and traj_index.max() >= len(reporter):
        raise ValueError(
            f"snippet refers to trajectory {traj_index.max()} but the reporter file has "
            f"{len(reporter)} trajectories; the two files are not aligned"
        )
    lookup: Dict[int, Dict[int, int]] = {}
    rows, y = [], []
    for row, (k, frame) in enumerate(zip(traj_index.tolist(), label_frame.tolist())):
        table = lookup.get(k)
        if table is None:
            table = {}
            for pos, fr in enumerate(reporter.frames[k].tolist()):
                table.setdefault(fr, pos)
            lookup[k] = table
        pos = table.get(frame)
        if pos is not None:
            rows.append(row)
            y.append(reporter.n2c[k][pos])
    return np.asarray(rows, dtype=int), np.asarray(y, dtype=float)


@dataclass
class LabeledFOV:
    """Snippets of one FOV that received a label, with that label."""

    snippets: FOVSnippets
    y: np.ndarray

    @property
    def X(self) -> np.ndarray:
        return self.snippets.X

    @property
    def fov(self) -> str:
        return self.snippets.fov


@dataclass
class PreparedData:
    fovs: Dict[str, LabeledFOV]       # only FOVs that passed QC
    good_fovs: List[str]
    cell_count_noise: Dict[str, float]
    trajectories: Dict[str, list]


def prepare_data(fovs: Sequence[str], paths: PathsConfig, qc: QCConfig,
                 feat: FeatureConfig) -> PreparedData:
    """Notebook In [4]-[21] end to end."""
    ensure_celltraj_importable(paths.celltraj_path)
    models = {fov: load_tracked_fov(paths.model_file(fov)) for fov in fovs}

    good, scores = qc_filter(models, qc)
    good_models = [models[f] for f in good]
    clean_handcrafted_features(good_models)
    standardize_com_displacement(good_models)

    trajectories, labeled = {}, {}
    for fov in good:
        model = models[fov]
        log.info("extracting single-cell trajectories of %s", fov)
        trajectories[fov] = extract_trajectories(model)

        vae = load_vae_embeddings(paths.vae_file(fov, feat.latent_dim))
        if vae.shape[0] != model.cells_indSet.size:
            log.warning("%s: %d VAE rows vs %d tracked cells", fov, vae.shape[0], model.cells_indSet.size)
        snippets = build_fov_snippets(fov, model, trajectories[fov],
                                      per_cell_features(model, vae, feat), feat)

        reporter = read_reporter_file(paths.reporter_file(fov))
        if len(reporter) != len(trajectories[fov]):
            log.warning("%s: %d tracked trajectories vs %d reporter trajectories",
                        fov, len(trajectories[fov]), len(reporter))
        rows, y = match_labels(snippets.traj_index, snippets.label_frame, reporter)
        labeled[fov] = LabeledFOV(snippets.subset(rows), y)
        log.info("%s: %d snippets, %d labelled", fov, len(snippets), rows.size)

    return PreparedData(labeled, good, scores, trajectories)
