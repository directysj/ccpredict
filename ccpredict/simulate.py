"""Synthetic stand-in for the LI204601 movies, calibrated to what is known about the real data.

The point is to develop and test the pipeline without the data. It is NOT
evidence that a modelling change helps on real cells: a method looks good
here when its assumptions match the simulator's. Use it for correctness,
for sizing evaluation artefacts, and for checking that a method recovers a
signal that is known to be there.

What is calibrated, and to what:
  * 6 FOVs, A2_* = EGF1 and B2_* = EGF2 (paper: biological replicates);
    193 frames at 15 min; cell-frame totals per FOV match the real .obj files
    (times ``scale``).
  * N/C marginal: matched by quantile mapping to the B2_4 histogram digitised
    from the notebook's own figure (median 1.43, 5-95% range 1.02-2.28,
    ~1% at >= 2.7). EGF1 gets the same reporter physics but different cycle
    parameters, so its marginal differs somewhat, as replicates do.
  * N/C dynamics: an HDHB-like (CDK2 sensor) trace per cycle. High N/C at
    birth, falling as CDK2 activity rises, lowest in G2/M, reset at division;
    a minority of cells pause in a CDK2-low state after mitosis.
  * Morphology: cell size grows over the cycle (area ~ volume^(2/3)), cells
    round up and brighten in mitosis, a weak direct CDK2-texture coupling,
    plus nuisance factors (elongation, local density, persistent cell-specific
    factors, slow per-cell fluctuations). These pass through one random
    nonlinear "encoder" to 256-d embeddings, plus noise.
  * Embedding noise is tuned so that LASSO trained on trimmed EGF1 and tested
    on EGF2 FOVs gives Pearson R near 0.53, the paper's snapshot VAE LASSO.
  * Tracking: celltraj-style disjoint trajectories built by tracing parent
    links backwards from the latest cells, so trajectories span divisions;
    1% missed detections, 0.3% per-frame track breaks, 10% of divisions not
    linked; 3% NaN N/C readouts and 1% gross label outliers.

Everything above that is not a number from the notebook or the paper is an
assumption, listed in SimConfig. Change them and the conclusions you draw from
synthetic runs can change too.

    python -m ccpredict.simulate OUT_DIR [--scale 0.25] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from .config import DEFAULT_FOV_REPLICATE as FOV_REPLICATE
# Cell-frame totals of the real tracked objects (notebook In [7] output).
REAL_CELL_FRAMES: Dict[str, int] = {
    "A2_1": 51054, "A2_2": 97764, "A2_3": 58579,
    "B2_1": 67836, "B2_3": 57884, "B2_4": 46096,
}

# B2_4 N/C histogram digitised from the notebook figure (In [24]); relative
# density on a 0.02 grid from 0.81 to 3.85. Linear-x density.
_NC_GRID_START, _NC_GRID_STEP = 0.81, 0.02
_NC_DENSITY = (
    0.002, 0.002, 0.002, 0.002, 0.013, 0.080, 0.080, 0.157, 0.285, 0.285, 0.495, 0.732, 0.732,
    0.939, 1.000, 1.000, 0.927, 0.843, 0.843, 0.749, 0.657, 0.657, 0.568, 0.539, 0.539, 0.490,
    0.467, 0.444, 0.452, 0.429, 0.406, 0.441, 0.441, 0.425, 0.437, 0.437, 0.438, 0.456, 0.456,
    0.395, 0.441, 0.441, 0.410, 0.432, 0.432, 0.413, 0.389, 0.389, 0.386, 0.357, 0.357, 0.346,
    0.340, 0.340, 0.342, 0.313, 0.284, 0.276, 0.257, 0.257, 0.242, 0.244, 0.247, 0.224, 0.224,
    0.229, 0.172, 0.172, 0.171, 0.150, 0.150, 0.147, 0.126, 0.126, 0.106, 0.089, 0.089, 0.097,
    0.082, 0.082, 0.064, 0.056, 0.056, 0.046, 0.046, 0.046, 0.039, 0.042, 0.042, 0.039, 0.033,
    0.033, 0.034, 0.031, 0.028, 0.021, 0.021, 0.021, 0.021, 0.021, 0.013, 0.016, 0.015, 0.013,
    0.015, 0.015, 0.007, 0.007, 0.007, 0.010, 0.006, 0.006, 0.006, 0.007, 0.007, 0.006, 0.004,
    0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.003, 0.003, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.002,
)

TRACKER_MODULE = '''\
import numpy as np


class Trajectory:
    """Minimal stand-in for celltraj's jcTrajectory_CP.Trajectory (synthetic data only)."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def get_unique_trajectories(self):
        self.trajectories = [np.asarray(t).copy() for t in self._tracks]
'''


@dataclass
class ReplicateParams:
    cycle_median_h: float      # G1+S+G2+M of cycling cells
    quiescent_frac: float      # fraction of births that pause CDK2-low first
    image_offset: float        # imaging/brightness shift on the halo factor (batch effect)
    density_offset: float      # plating-density difference


@dataclass
class SimConfig:
    scale: float = 0.25
    seed: int = 0
    n_frames: int = 193
    dt_h: float = 0.25
    cycle_cv: float = 0.15
    mitosis_h: float = 1.0
    quiescent_median_h: float = 8.0
    quiescent_cv: float = 0.5
    a_birth: float = 0.15            # CDK2 activity at birth of cycling cells
    a_gamma: float = 0.6             # activity rises fast early, then plateaus
    nc_cell_log_sd: float = 0.04     # per-cell reporter offset (not predictable from shape)
    label_log_sd: float = 0.05       # per-frame N/C measurement noise
    label_outlier_frac: float = 0.01
    label_outlier_log_sd: float = 0.4
    label_nan_frac: float = 0.03
    size_cell_sd: float = 0.20       # cell-to-cell spread of log size at birth
    size_inherit_corr: float = 0.7
    p_missed_detection: float = 0.01
    p_track_break: float = 0.003
    p_division_linked: float = 0.9
    n_hidden: int = 48
    latent_dim: int = 256
    n_classical: int = 92
    vae_noise_sd: float = 1.2        # tuned: snapshot LASSO EGF1->EGF2 R 0.532 (paper 0.531)
    classical_noise_sd: float = 0.7  # tuned: classical LASSO R ~0.51 (paper 0.510)
    # Share of the feature-noise variance that persists along a cell's track
    # (AR(1), rho per frame), so adjacent frames of one cell look alike, as in
    # real time-lapse embeddings. The rest is independent per frame.
    noise_persistent_frac: float = 0.5
    noise_rho: float = 0.95
    fov_output_offset_sd: float = 0.15
    replicates: Dict[str, ReplicateParams] = field(default_factory=lambda: {
        "EGF1": ReplicateParams(17.5, 0.06, 0.0, 0.0),
        "EGF2": ReplicateParams(18.5, 0.08, 0.4, 0.3),
    })
    fov_replicate: Dict[str, str] = field(default_factory=lambda: dict(FOV_REPLICATE))
    cell_frames: Dict[str, int] = field(default_factory=lambda: dict(REAL_CELL_FRAMES))


# ----------------------------------------------------------------------------
# Reporter: CDK2 activity -> N/C, calibrated by quantile mapping
# ----------------------------------------------------------------------------

def target_nc_quantile(p: np.ndarray) -> np.ndarray:
    """Quantile function of the digitised B2_4 N/C histogram."""
    dens = np.asarray(_NC_DENSITY)
    edges = _NC_GRID_START - _NC_GRID_STEP / 2 + _NC_GRID_STEP * np.arange(dens.size + 1)
    cdf = np.concatenate([[0.0], np.cumsum(dens)])
    cdf /= cdf[-1]
    keep = np.concatenate([[True], np.diff(cdf) > 0])  # strictly increasing for interp
    return np.interp(p, cdf[keep], edges[keep])


def _lognormal(rng, median, cv, size):
    sigma = math.sqrt(math.log(1 + cv ** 2))
    return median * np.exp(sigma * rng.standard_normal(size))


# ----------------------------------------------------------------------------
# Lineages
# ----------------------------------------------------------------------------

@dataclass
class Lineage:
    birth: np.ndarray        # h
    total: np.ndarray        # cycle duration incl. any CDK2-low pause, h
    pause: np.ndarray        # CDK2-low pause, h (0 for most cells)
    mother: np.ndarray       # -1 for founders
    size_offset: np.ndarray
    traits: np.ndarray       # persistent cell-specific nuisance factors (n_cells, 10)
    a_pause: np.ndarray      # residual activity during the pause


def simulate_lineage(n0: int, rep: ReplicateParams, cfg: SimConfig, rng) -> Lineage:
    t_end = cfg.n_frames * cfg.dt_h

    def new_cycle(n):
        active = _lognormal(rng, rep.cycle_median_h, cfg.cycle_cv, n)
        paused = rng.random(n) < rep.quiescent_frac
        pause = np.where(paused, _lognormal(rng, cfg.quiescent_median_h, cfg.quiescent_cv, n), 0.0)
        return pause, active + pause

    birth, total, pause, mother, size, traits = [], [], [], [], [], []
    p0, tot0 = new_cycle(n0)
    # Asynchronous start: ages from the steady-state age density of a growing population.
    x = -np.log2(1 - rng.random(n0) / 2)
    birth.extend((-x * tot0).tolist()); total.extend(tot0.tolist()); pause.extend(p0.tolist())
    mother.extend([-1] * n0)
    size.extend((cfg.size_cell_sd * rng.standard_normal(n0)).tolist())
    traits.extend(list(rng.standard_normal((n0, 10))))

    i = 0
    while i < len(birth):
        div = birth[i] + total[i]
        if div < t_end:
            p, tot = new_cycle(2)
            for k in range(2):
                birth.append(div); total.append(tot[k]); pause.append(p[k]); mother.append(i)
                rho = cfg.size_inherit_corr
                size.append(rho * size[i] + math.sqrt(1 - rho ** 2) * cfg.size_cell_sd * rng.standard_normal())
                traits.append(0.6 * traits[i] + 0.8 * rng.standard_normal(10))
        i += 1
    n = len(birth)
    return Lineage(np.array(birth), np.array(total), np.array(pause), np.array(mother, dtype=int),
                   np.array(size), np.vstack(traits), 0.02 + 0.04 * rng.random(n))


def _ar1(n: int, rho: float, rng) -> np.ndarray:
    from scipy.signal import lfilter

    e = rng.standard_normal(n) * math.sqrt(1 - rho ** 2)
    e[0] = rng.standard_normal()
    return lfilter([1.0], [1.0, -rho], e)


@dataclass
class Detections:
    cell: np.ndarray
    frame: np.ndarray
    age: np.ndarray


def detect(lin: Lineage, cfg: SimConfig, rng) -> Detections:
    cells, frames = [], []
    for c in range(lin.birth.size):
        f0 = max(0, math.ceil(lin.birth[c] / cfg.dt_h - 1e-9))
        f1 = min(cfg.n_frames, math.ceil((lin.birth[c] + lin.total[c]) / cfg.dt_h - 1e-9))
        if f1 > f0:
            fr = np.arange(f0, f1)
            cells.append(np.full(fr.size, c)); frames.append(fr)
    cell, frame = np.concatenate(cells), np.concatenate(frames)
    keep = rng.random(cell.size) >= cfg.p_missed_detection
    cell, frame = cell[keep], frame[keep]
    # Detection index = celltraj cell index: ordered by frame, random within a frame.
    order = np.lexsort((rng.random(cell.size), frame))
    cell, frame = cell[order], frame[order]
    return Detections(cell, frame, frame * cfg.dt_h - lin.birth[cell])


def cell_state(lin: Lineage, det: Detections, cfg: SimConfig):
    """Per detection: CDK2 activity a, cycle fraction phi, mitotic and post-mitotic flags."""
    c = det.cell
    total, pause, age = lin.total[c], lin.pause[c], det.age
    phi = np.clip(age / total, 0, 1)
    mitotic = age >= total - cfg.mitosis_h
    post_mitotic = age < 0.5
    tau = np.clip((age - pause) / np.maximum(total - pause - cfg.mitosis_h, 1e-6), 0, 1)
    a = cfg.a_birth + (1 - cfg.a_birth) * tau ** cfg.a_gamma
    a = np.where(age < pause, lin.a_pause[c], a)
    a = np.where(mitotic, 1.0, a)
    return a, phi, mitotic.astype(float), post_mitotic.astype(float)


def build_reporter_map(cfg: SimConfig, rep: ReplicateParams, rng, n_pilot: int = 3000):
    """Monotone decreasing map a -> N/C whose population marginal matches the B2_4 histogram."""
    lin = simulate_lineage(n_pilot, rep, cfg, rng)
    det = detect(lin, cfg, rng)
    a, *_ = cell_state(lin, det, cfg)
    a_sorted = np.sort(a + 1e-9 * rng.standard_normal(a.size))
    grid = np.linspace(0, 1, 2001)
    frac_below = np.searchsorted(a_sorted, grid) / a_sorted.size
    nc = target_nc_quantile(np.clip(1 - frac_below, 1e-4, 1 - 1e-4))
    return grid, nc


# ----------------------------------------------------------------------------
# One FOV
# ----------------------------------------------------------------------------

@dataclass
class Encoder:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    Wc1: np.ndarray
    bc1: np.ndarray
    Wc2: np.ndarray


N_FACTORS = 16


def make_encoder(cfg: SimConfig, rng) -> Encoder:
    h = cfg.n_hidden
    return Encoder(
        rng.standard_normal((N_FACTORS, h)) / math.sqrt(N_FACTORS), 0.3 * rng.standard_normal(h),
        rng.standard_normal((h, cfg.latent_dim)) / math.sqrt(h) * 2.0,
        rng.standard_normal((N_FACTORS, h)) / math.sqrt(N_FACTORS), 0.3 * rng.standard_normal(h),
        rng.standard_normal((h, cfg.n_classical)) / math.sqrt(h) * 2.0,
    )


@dataclass
class SimulatedFOV:
    fov: str
    replicate: str
    cell_frame: np.ndarray
    embeddings_clean: np.ndarray
    classical_clean: np.ndarray
    com: np.ndarray
    nc_obs: np.ndarray
    nc_true: np.ndarray
    trajectories: List[np.ndarray]
    parent: np.ndarray
    truth: dict


def simulate_fov(fov: str, cfg: SimConfig, enc: Encoder, reporter_map, rng) -> SimulatedFOV:
    rep_name = cfg.fov_replicate[fov]
    rep = cfg.replicates[rep_name]
    doubling = rep.cycle_median_h + rep.quiescent_frac * cfg.quiescent_median_h
    growth = sum(2 ** (f * cfg.dt_h / doubling) for f in range(cfg.n_frames))
    n0 = max(5, round(cfg.cell_frames[fov] * cfg.scale / growth))

    lin = simulate_lineage(n0, rep, cfg, rng)
    det = detect(lin, cfg, rng)
    n = det.cell.size
    a, phi, mitotic, post = cell_state(lin, det, cfg)

    # Reporter.
    grid, nc_of_a = reporter_map
    cell_offset = cfg.nc_cell_log_sd * rng.standard_normal(lin.birth.size)
    nc_true = np.interp(a, grid, nc_of_a) * np.exp(cell_offset[det.cell])
    log_noise = cfg.label_log_sd * rng.standard_normal(n)
    outlier = rng.random(n) < cfg.label_outlier_frac
    log_noise[outlier] += cfg.label_outlier_log_sd * rng.standard_normal(outlier.sum())
    nc_obs = nc_true * np.exp(log_noise)
    nc_obs[rng.random(n) < cfg.label_nan_frac] = np.nan

    # Slow per-cell fluctuations (AR(1) along each cell's frames).
    slow = np.zeros((n, 5))
    for c in np.unique(det.cell):
        rows = np.where(det.cell == c)[0]
        for k in range(5):
            slow[rows, k] = _ar1(rows.size, 0.9, rng)

    # Local density grows with the population; per-FOV and per-replicate offsets.
    counts = np.bincount(det.frame, minlength=cfg.n_frames).astype(float)
    density = np.log(counts / counts[0])[det.frame] + rep.density_offset + 0.3 * slow[:, 4]

    tr = lin.traits[det.cell]
    F = np.column_stack([
        # log area, in units of 0.23: birth-size spread (sd 0.2) vs growth over a
        # cycle ((2/3) ln 2 = 0.46 in log area); rounded mitotic cells look smaller
        (lin.size_offset[det.cell] + (2 / 3) * math.log(2) * phi) / 0.23 - 1.2 * mitotic - 0.8 * post,
        tr[:, 0] + 0.5 * slow[:, 0] + 2.0 * mitotic + 1.0 * post,      # roundness
        0.5 * slow[:, 1] + 1.5 * mitotic + rep.image_offset,            # phase halo / brightness
        0.4 * (a - 0.6) / 0.25 + 0.8 * slow[:, 2],                      # weak CDK2-coupled texture
        tr[:, 1] + 0.6 * slow[:, 3] - 1.0 * mitotic,                    # elongation
        density,                                                         # local crowding
        tr[:, 2:10],                                                     # persistent cell traits
        slow[:, :2] @ np.array([[0.7, 0.3], [-0.3, 0.7]]),               # more slow nuisance
    ])
    assert F.shape[1] == N_FACTORS, F.shape

    fov_offset = cfg.fov_output_offset_sd * rng.standard_normal(cfg.latent_dim)
    emb = np.tanh(F @ enc.W1 + enc.b1) @ enc.W2 + fov_offset
    cls = np.tanh(F @ enc.Wc1 + enc.bc1) @ enc.Wc2
    com = np.column_stack([np.abs(rng.standard_normal(n)) * (1 + mitotic), rng.standard_normal((n, 2))])

    # Tracking: parent links along each cell, bridged over missed frames; divisions
    # linked with probability p_division_linked; random breaks.
    parent = np.full(n, -1)
    last_det = {}
    for d in range(n):  # detections are in frame order
        c = det.cell[d]
        if c in last_det:
            if det.frame[d] - det.frame[last_det[c]] <= 3:
                parent[d] = last_det[c]
        else:
            m = lin.mother[c]
            if m >= 0 and m in last_det and rng.random() < cfg.p_division_linked:
                parent[d] = last_det[m]
        last_det[c] = d
    parent[rng.random(n) < cfg.p_track_break] = -1
    trajectories = _trace_back(parent, det.frame, rng)

    truth = dict(cell=det.cell, frame=det.frame, a=a, phi=phi, mitotic=mitotic,
                 nc_true=nc_true, paused=(det.age < lin.pause[det.cell]).astype(float))
    return SimulatedFOV(fov, rep_name, det.frame.astype(float), emb, cls, com, nc_obs, nc_true,
                        trajectories, parent, truth)


def _trace_back(parent: np.ndarray, frame: np.ndarray, rng) -> List[np.ndarray]:
    """Disjoint trajectories: from the latest unassigned detection, follow parents back."""
    order = np.lexsort((rng.random(frame.size), -frame))
    used = np.zeros(frame.size, dtype=bool)
    trajectories = []
    for d in order:
        if used[d]:
            continue
        path = [d]
        used[d] = True
        p = parent[d]
        while p >= 0 and not used[p]:
            path.append(p)
            used[p] = True
            p = parent[p]
        trajectories.append(np.array(path[::-1], dtype=int))
    return trajectories


# ----------------------------------------------------------------------------
# Whole experiment
# ----------------------------------------------------------------------------

def simulate(cfg: SimConfig) -> Dict[str, SimulatedFOV]:
    rng = np.random.default_rng(cfg.seed)
    enc = make_encoder(cfg, rng)
    reporter_map = build_reporter_map(cfg, cfg.replicates["EGF2"], rng)
    return {fov: simulate_fov(fov, cfg, enc, reporter_map, rng) for fov in cfg.fov_replicate}


def _track_noise(cell: np.ndarray, frame: np.ndarray, dim: int, cfg: SimConfig, rng) -> np.ndarray:
    """Unit-variance noise: part independent per frame, part AR(1) along each cell's frames."""
    from scipy.signal import lfilter

    p, rho = cfg.noise_persistent_frac, cfg.noise_rho
    noise = math.sqrt(1 - p) * rng.standard_normal((cell.size, dim))
    if p > 0:
        order = np.lexsort((frame, cell))
        c_sorted = cell[order]
        starts = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]])
        ends = np.r_[starts[1:], c_sorted.size]
        persistent = np.empty((cell.size, dim))
        for s, e in zip(starts, ends):
            w = rng.standard_normal((e - s, dim)) * math.sqrt(1 - rho ** 2)
            w[0] = rng.standard_normal(dim)
            persistent[order[s:e]] = lfilter([1.0], [1.0, -rho], w, axis=0)
        noise += math.sqrt(p) * persistent
    return noise


def add_feature_noise(sim: SimulatedFOV, cfg: SimConfig, rng):
    cell, frame = sim.truth["cell"], sim.truth["frame"]
    emb = sim.embeddings_clean + cfg.vae_noise_sd * _track_noise(cell, frame, cfg.latent_dim, cfg, rng)
    cls = sim.classical_clean + cfg.classical_noise_sd * _track_noise(cell, frame, cfg.n_classical, cfg, rng)
    cls[rng.random(cls.shape) < 0.01] = np.nan
    return emb.astype(np.float32), cls


def write_dataset(sims: Dict[str, SimulatedFOV], cfg: SimConfig, out_dir: str,
                  sys_name: str = "LI204601_P", reporter_name: str = "LI204601_G") -> None:
    """Write the same file layout the pipeline reads from the cluster."""
    os.makedirs(os.path.join(out_dir, "o2vae"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reporter"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "truth"), exist_ok=True)
    with open(os.path.join(out_dir, "jcTrajectory_CP.py"), "w") as fh:
        fh.write(TRACKER_MODULE)
    if out_dir not in sys.path:
        sys.path.insert(0, out_dir)
    sys.modules.pop("jcTrajectory_CP", None)
    import jcTrajectory_CP

    rng = np.random.default_rng(cfg.seed + 1)
    for fov, s in sims.items():
        emb, cls = add_feature_noise(s, cfg, rng)
        n = s.cell_frame.size
        np.savez(os.path.join(out_dir, "o2vae", f"features_vae{cfg.latent_dim}_{sys_name[:-1]}{fov}.npz"),
                 embeddings=emb, labels=np.zeros(n))
        linset = [s.parent[s.cell_frame == f] for f in range(cfg.n_frames)]
        obj = jcTrajectory_CP.Trajectory(
            cells_indSet=np.arange(n), cells_frameSet=s.cell_frame,
            cells_imgfileSet=s.cell_frame.astype(int), Xf=cls, Xf_com=s.com,
            linSet=linset, _tracks=s.trajectories)
        with open(os.path.join(out_dir, f"{sys_name}_{fov}.obj"), "wb") as fh:
            pickle.dump(obj, fh)
        with open(os.path.join(out_dir, "reporter", f"{reporter_name}_{fov}.dat"), "w") as fh:
            cc = rng.random(n)
            for t in s.trajectories:
                fh.write(str([float(v) for v in s.nc_obs[t]]) + "\n")
                fh.write(str([float(v) for v in cc[t]]) + "\n")
                fh.write(str([int(s.cell_frame[c]) for c in t]) + "\n")
        np.savez(os.path.join(out_dir, "truth", f"{fov}.npz"), **s.truth)

    meta = asdict(cfg)
    meta["summary"] = {fov: {"replicate": s.replicate, "cell_frames": int(s.cell_frame.size),
                             "trajectories": len(s.trajectories),
                             "nc_quantiles_5_50_95": np.nanpercentile(s.nc_obs, [5, 50, 95]).round(3).tolist(),
                             "frac_nc_ge_2.7": float(np.nanmean(s.nc_obs >= 2.7)),
                             "frac_nc_lt_0.9": float(np.nanmean(s.nc_obs < 0.9))}
                       for fov, s in sims.items()}
    with open(os.path.join(out_dir, "simulation.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ccpredict.simulate", description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--scale", type=float, default=0.25, help="fraction of the real cell counts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = SimConfig(scale=args.scale, seed=args.seed)
    sims = simulate(cfg)
    write_dataset(sims, cfg, args.out_dir)
    with open(os.path.join(args.out_dir, "simulation.json")) as fh:
        for fov, s in json.load(fh)["summary"].items():
            print(fov, s)


if __name__ == "__main__":
    main()
