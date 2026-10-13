"""The vectorised rewrites vs verbatim copies of the notebook's functions.

The notebook functions below are copied unchanged (apart from wrapping) so the
tests do not need the .ipynb. Run with: pytest tests/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccpredict.data import match_labels, read_reporter_file, sliding_snippets  # noqa: E402
from ccpredict.preprocessing import discretize, trim_by_histogram  # noqa: E402


# ---------------------------------------------------------------- notebook code, verbatim
class _Holder:
    def __init__(self, trajectories):
        self.trajectories = trajectories


def nb_get_snippets_with_traj_inds(self, seg_length):
    n_sctraj = len(self.trajectories)
    traj_segSet = np.zeros((0, seg_length)).astype(int)
    ind_map_snippet_fulltraj = np.array([])
    for ind_traj in range(n_sctraj):
        cell_traj = self.trajectories[ind_traj]
        traj_len = cell_traj.size
        if traj_len >= seg_length:
            for ic in range(traj_len - seg_length):
                traj_seg = cell_traj[ic:ic+seg_length]
                traj_segSet = np.append(traj_segSet, traj_seg[np.newaxis, :], axis=0)
                ind_map_snippet_fulltraj = np.append(ind_map_snippet_fulltraj, ind_traj)
    return ind_map_snippet_fulltraj, traj_segSet


def nb_get_cellCycInfoTrajs(file_info):
    with open(file_info, 'r') as fp:
        lines = fp.readlines()
    nuc2cytoRatio, CC_vals, frames = [], [], []
    for line1, line2, line3 in zip(lines[::3], lines[1::3], lines[2::3]):
        line1 = np.array(line1.strip()[1:-1].split(', '), dtype=float)
        line2 = np.array(line2.strip()[1:-1].split(', '), dtype=float)
        line3 = np.array(line3.strip()[1:-1].split(', '), dtype=int)
        mask = ~np.isnan(line1)
        nuc2cytoRatio.append(line1[mask]); CC_vals.append(line2[mask]); frames.append(line3[mask])
    return nuc2cytoRatio, CC_vals, frames


def nb_match(indc_map_fulltraj_snippets, fid_snippets, indstm, frame_numbers, n2c_ratio):
    valid_indstm, nuc2Cyto_ratio_last_frames = [], []
    for j, ind_fulltraj in enumerate(indc_map_fulltraj_snippets):
        possible_indices = np.where(frame_numbers[ind_fulltraj] == fid_snippets[j])[0]
        if len(possible_indices) > 0:
            nuc2Cyto_ratio_last_frames.append(n2c_ratio[ind_fulltraj][possible_indices[0]])
            valid_indstm.append(indstm[j])
    return np.array(valid_indstm), np.array(nuc2Cyto_ratio_last_frames)


def nb_trim_cc_vals(ratio_values, num_bins, target_density):
    df = pd.DataFrame(ratio_values, columns=['CC'])
    df['original_index'] = df.index
    counts, bin_edges = np.histogram(df['CC'], bins=num_bins)
    df['bin'] = pd.cut(df['CC'], bins=bin_edges, labels=False, include_lowest=True)
    trimmed_data = []
    for i in range(num_bins):
        bin_data = df[(df['bin'] == i)]
        if bin_data.shape[0] > target_density:
            sampled_data = bin_data.sample(n=target_density, random_state=42)
        else:
            sampled_data = bin_data
        trimmed_data.append(sampled_data)
    return pd.concat(trimmed_data).sort_values('original_index')


def nb_class_conditions(data, boundaries):
    conditions = []
    if len(boundaries) == 1:
        conditions.append(data < boundaries[0]); conditions.append(data >= boundaries[0])
    else:
        for i in range(len(boundaries) + 1):
            if i == 0:
                condition = data < boundaries[i]
            elif i == len(boundaries):
                condition = data >= boundaries[i-1]
            else:
                condition = (data >= boundaries[i-1]) & (data < boundaries[i])
            conditions.append(condition)
    return conditions


# ---------------------------------------------------------------- tests
def _random_trajectories(rng, n=60):
    lengths = rng.integers(1, 60, size=n)
    start = 0
    out = []
    for L in lengths:
        out.append(np.arange(start, start + L))
        start += L
    out.insert(3, np.array([], dtype=int))  # empty trajectory
    return out


@pytest.mark.parametrize("trajl", [1, 2, 4, 8, 20, 40])
def test_snippets_match_notebook(trajl):
    trajs = _random_trajectories(np.random.default_rng(trajl))
    ref_idx, ref_seg = nb_get_snippets_with_traj_inds(_Holder(trajs), trajl)
    idx, seg = sliding_snippets(trajs, trajl)
    assert np.array_equal(ref_idx.astype(int), idx)
    assert np.array_equal(ref_seg, seg) and seg.dtype == ref_seg.dtype


@pytest.mark.parametrize("trajl", [1, 4])
def test_include_last_window_adds_one_window_per_trajectory(trajl):
    trajs = [np.arange(10), np.arange(10, 10 + trajl), np.arange(50, 50 + trajl - 1)]
    idx_nb, _ = sliding_snippets(trajs, trajl)
    idx_fix, seg = sliding_snippets(trajs, trajl, include_last_window=True)
    assert np.bincount(idx_nb, minlength=3).tolist() == [10 - trajl, 0, 0]
    assert np.bincount(idx_fix, minlength=3).tolist() == [11 - trajl, 1, 0]
    assert seg[10 - trajl, -1] == 9  # last window ends on the last cell


def _write_reporter(path, rng, n_traj=40):
    frames_all, n2c_all = [], []
    with open(path, "w") as fh:
        for _ in range(n_traj):
            L = int(rng.integers(1, 30))
            start = int(rng.integers(0, 150))
            frames = np.sort(rng.choice(np.arange(start, start + 2 * L), L, replace=False))
            if L > 3:
                frames[2] = frames[1]  # duplicate frame: first match must win
            n2c = rng.random(L) * 3
            n2c[rng.random(L) < 0.2] = np.nan
            fh.write(str([float(v) for v in n2c]) + "\n")
            fh.write(str([float(v) for v in rng.random(L)]) + "\n")
            fh.write(str([int(v) for v in frames]) + "\n")
            frames_all.append(frames); n2c_all.append(n2c)
    return frames_all


def test_reporter_and_label_matching_match_notebook(tmp_path):
    rng = np.random.default_rng(0)
    path = str(tmp_path / "rep.dat")
    frames_all = _write_reporter(path, rng)

    ref_n2c, ref_cc, ref_frames = nb_get_cellCycInfoTrajs(path)
    rep = read_reporter_file(path)
    for a, b in zip(ref_n2c + ref_cc + ref_frames, rep.n2c + rep.cc + rep.frames):
        assert np.array_equal(a, b) and a.dtype == b.dtype

    n = 3000
    traj_index = rng.integers(0, len(frames_all), n)
    label_frame = np.array([int(rng.choice(np.concatenate([frames_all[k], [999]])))
                            for k in traj_index])
    indstm = np.arange(n) + 10_000  # global row ids, as in the notebook
    ref_rows, ref_y = nb_match(traj_index, label_frame, indstm, ref_frames, ref_n2c)
    rows, y = match_labels(traj_index, label_frame, rep)
    assert np.array_equal(ref_rows, indstm[rows])
    assert np.array_equal(ref_y, y)


def test_label_matching_rejects_misaligned_files(tmp_path):
    rng = np.random.default_rng(1)
    path = str(tmp_path / "rep.dat")
    _write_reporter(path, rng, n_traj=5)
    with pytest.raises(ValueError, match="not aligned"):
        match_labels(np.array([0, 7]), np.array([1, 1]), read_reporter_file(path))


@pytest.mark.parametrize("density", [5, 50, 2000])
def test_trim_matches_notebook(density):
    rng = np.random.default_rng(density)
    y = np.exp(rng.normal(0.3, 0.5, size=5000))
    ref = nb_trim_cc_vals(y, 50, density)
    idx = trim_by_histogram(y, 50, density, seed=42)
    assert np.array_equal(ref["original_index"].to_numpy(), idx)
    assert np.array_equal(ref["CC"].to_numpy(), y[idx])


@pytest.mark.parametrize("boundaries", [[0.9], [0.9, 2.7], [0.5, 1.0, 2.0, 3.0]])
def test_discretize_matches_notebook(boundaries):
    y = np.concatenate([np.exp(np.random.default_rng(0).normal(0.3, 0.6, 4000)), boundaries])
    labels = list(range(len(boundaries) + 1))
    ref = np.select(nb_class_conditions(y, boundaries), labels)
    assert np.array_equal(ref, discretize(y, boundaries))


def test_discretize_rejects_nan():
    with pytest.raises(ValueError):
        discretize(np.array([1.0, np.nan]), [0.9, 2.7])
