"""Synthetic stand-in for the real inputs, with the same file layout and object shape.

Writes, under one directory:
  jcTrajectory_CP.py                  fake tracker module (pickles reference it)
  LI204601_P_<fov>.obj                pickled Trajectory objects
  o2vae/features_vae256_LI204601_<fov>.npz
  reporter/LI204601_G_<fov>.dat       3-lines-per-trajectory N/C, CC, frames

Cells divide every 18-32 frames; the N/C ratio rises with cell-cycle phase and
the embeddings carry a noisy phase signal, so the classifier has something to
learn. A few tracks have a missed frame, 3% of N/C readouts are NaN, and FOV
B2_3 loses most cells in many late frames so it fails the QC.

    python tests/synthetic.py OUT_DIR
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

FOVS = ("A2_1", "A2_2", "A2_3", "B2_1", "B2_3", "B2_4")
N_FRAMES = 193
LATENT_DIM = 256

TRACKER_MODULE = '''\
import numpy as np


class Trajectory:
    """Minimal stand-in for celltraj's jcTrajectory_CP.Trajectory."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def get_unique_trajectories(self):
        self.trajectories = [np.asarray(t).copy() for t in self._tracks]
'''


def simulate_tracks(rng, n0=8, cap=40, dropout=False):
    """Return a list of tracks; each track is (start_frame, life, list_of_frames)."""
    alive = [dict(start=0, life=int(rng.integers(18, 33)), frames=[]) for _ in range(n0)]
    done = []
    for f in range(N_FRAMES):
        nxt = []
        for tr in alive:
            tr["frames"].append(f)
            if f - tr["start"] + 1 >= tr["life"] and f < N_FRAMES - 1:
                done.append(tr)
                n_children = 2 if len(alive) < cap else int(rng.choice([0, 1, 1, 2]))
                for _ in range(n_children):
                    nxt.append(dict(start=f + 1, life=int(rng.integers(18, 33)), frames=[]))
            else:
                nxt.append(tr)
        alive = nxt
    tracks = done + alive

    # Missed detections: drop one interior frame from ~5% of tracks.
    for tr in tracks:
        if len(tr["frames"]) > 4 and rng.random() < 0.05:
            tr["frames"].pop(int(rng.integers(1, len(tr["frames"]) - 1)))
    # Segmentation failure in late frames (only for the QC-failing FOV).
    if dropout:
        bad = set(rng.choice(np.arange(136, 191), size=25, replace=False).tolist())
        for tr in tracks:
            tr["frames"] = [f for f in tr["frames"] if f not in bad or rng.random() < 0.15]
    return [tr for tr in tracks if tr["frames"]]


def build_fov(rng, fov, out_dir, projection):
    import jcTrajectory_CP  # noqa: F401  (written to out_dir, imported via sys.path)

    tracks = simulate_tracks(rng, dropout=(fov == "B2_3"))
    order = rng.permutation(len(tracks))
    tracks = [tracks[k] for k in order]

    # Cell indices ordered by frame, as a tracker would number detections.
    entries = [(f, k) for k, tr in enumerate(tracks) for f in tr["frames"]]
    entries.sort(key=lambda e: (e[0], e[1]))
    n_cells = len(entries)
    frame_of = np.array([e[0] for e in entries], dtype=float)
    cell_of = {}
    for idx, (f, k) in enumerate(entries):
        cell_of[(k, f)] = idx
    traj_cells = [np.array([cell_of[(k, f)] for f in tr["frames"]], dtype=int)
                  for k, tr in enumerate(tracks)]

    phase = np.empty(n_cells)
    for k, tr in enumerate(tracks):
        for f in tr["frames"]:
            phase[cell_of[(k, f)]] = (f - tr["start"]) / tr["life"]

    n2c = np.exp(1.6 * phase - 0.4 + 0.15 * rng.standard_normal(n_cells))
    n2c[rng.random(n_cells) < 0.03] = np.nan
    cc = rng.random(n_cells)

    basis = np.stack([np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase), phase], axis=1)
    shift = 0.3 if fov.startswith("B") else 0.0
    emb = basis @ projection + rng.standard_normal((n_cells, LATENT_DIM)) + shift
    np.savez(os.path.join(out_dir, "o2vae", f"features_vae{LATENT_DIM}_LI204601_{fov}.npz"),
             embeddings=emb.astype(np.float32), labels=np.zeros(n_cells))

    Xf = rng.standard_normal((n_cells, 92))
    Xf[rng.random(Xf.shape) < 0.01] = np.nan
    obj = sys.modules["jcTrajectory_CP"].Trajectory(
        cells_indSet=np.arange(n_cells), cells_frameSet=frame_of,
        cells_imgfileSet=frame_of.astype(int), Xf=Xf, Xf_com=rng.standard_normal((n_cells, 3)),
        linSet=[np.zeros(1)] * N_FRAMES, _tracks=traj_cells)
    with open(os.path.join(out_dir, f"LI204601_P_{fov}.obj"), "wb") as fh:
        pickle.dump(obj, fh)

    with open(os.path.join(out_dir, "reporter", f"LI204601_G_{fov}.dat"), "w") as fh:
        for cells in traj_cells:
            fh.write(str([float(v) for v in n2c[cells]]) + "\n")
            fh.write(str([float(v) for v in cc[cells]]) + "\n")
            fh.write(str([int(frame_of[c]) for c in cells]) + "\n")
    return n_cells, len(traj_cells)


def main(out_dir: str, seed: int = 7) -> None:
    os.makedirs(os.path.join(out_dir, "o2vae"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reporter"), exist_ok=True)
    with open(os.path.join(out_dir, "jcTrajectory_CP.py"), "w") as fh:
        fh.write(TRACKER_MODULE)
    sys.path.insert(0, out_dir)

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((3, LATENT_DIM)) * 1.5
    for fov in FOVS:
        n_cells, n_tracks = build_fov(rng, fov, out_dir, projection)
        print(f"{fov}: {n_cells} cells, {n_tracks} trajectories")


if __name__ == "__main__":
    main(sys.argv[1])
