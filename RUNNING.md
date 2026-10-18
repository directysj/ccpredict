# Running the benchmark on the real data

This is for whoever has access to the LI204601 files. It compares one-stage
neural regressors -- a single network predicting the N/C ratio directly from a
trajectory snippet -- with the paper's LASSO and random-forest baselines, all on
the same rows. Your run produces a results folder; send that back.

**This branch has no classification stage.** The published two-stage model
(CNN class probabilities + RFECV forest) is on branch `ccp_torch`; run that
branch if you want the published arms.

No code changes are needed, only a config file with the data paths.

## 1. Environment

The environment that ran `vaeFeatCCinfoABwells-NNclfRFreg.ipynb` works:
Python ≥ 3.9, numpy, pandas, scipy, scikit-learn, matplotlib, joblib, csaps
and PyTorch ≥ 2.5 (the networks are PyTorch; TensorFlow is not needed). Any
recent PyTorch build works; a CUDA build is only needed to use a GPU.

```bash
cd ccpredict
pip install -e .            # or: export PYTHONPATH=$PWD
pytest tests/               # optional, about a minute, no data needed
```

## 2. Config

```bash
python -m ccpredict.benchmark --write-config bench.json
```

Open `bench.json` and check `pipeline.paths`. The defaults are the notebook's paths:

| key | default |
|---|---|
| `traj_dir` | `/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/segsCellPose/bayesianTrackTest/` |
| `vae_dir` | `../o2vae` (relative to where you run; use an absolute path) |
| `reporter_dir` | `/home/groups/ZuckermanLab/jalim/LI204601_INCUCYTE/cellCycInfoRawReporterImgs/` |
| `celltraj_path` | `/home/groups/ZuckermanLab/jalim/instalLocal/celltraj/celltraj` (folder with `jcTrajectory_CP.py`) |

Also check `pipeline.net.device`: `"cpu"` (the default), `"gpu"` (CUDA, else
Apple MPS; fails if PyTorch sees neither) or `"auto"` (CUDA, then MPS, then
CPU). On a GPU node set it to `"auto"`.

And check `pipeline.features.trajl` -- the snippet length. The defaults read
the label at the snippet's **last** snapshot (`label_frame: "snippet_end"`,
`include_last_window: true`), which is where the ground truth belongs; the
notebook read it one frame later.

Files expected: `LI204601_P_<FOV>.obj`, `features_vae256_LI204601_<FOV>.npz`,
`LI204601_G_<FOV>.dat` for FOVs A2_1, A2_2, A2_3, B2_1, B2_3, B2_4.

## 3. Smoke test (minutes)

```bash
python -m ccpredict.benchmark --config bench.json --output-dir smoke --quick --arms constant,lasso_vae,rf_vae
```

This checks that the data loads and the paths are right. Its numbers are not
meaningful.

## 4. Full run

```bash
python -m ccpredict.benchmark --config bench.json --output-dir results
```

Train on the A wells (EGF1), test on each B well (EGF2). Run it as a batch job.
Results are written after every arm, so a run that is cut off still leaves what
finished; prepared features are cached in `results/cache/`, so rerunning skips
the slow loading.

**Runtime.** Much cheaper than the two-stage model: no RFECV, no 40-minute CNN
fits. At `trajl=1` every arm is minutes. At `trajl=8` the rows carry 2,048
features and the transformer arms took ~200-900 s per fit on two cores, so:

- request a node with many cores and **at least 16 GB** -- `trajl=8` with
  three training FOVs exceeded 6 GB and was OOM-killed on a small machine;
- the LASSO on 2,048 features is itself ~250 s;
- run arm subsets as separate jobs with separate `--output-dir` if you want
  early results: `--arms constant,lasso_classical,lasso_vae,rf_vae` first,
  then the `nn_*` arms.

If you change any setting, keep it the same across all arms and say so when you
send results back.

## 5. What to send back

Send the `results/` folder **without `cache/`** (it holds the features).
`predictions/` holds per-cell predicted and measured N/C; include it if that is
fine for you, otherwise leave it out. `report.md` alone already answers the
main questions.

## What the arms are

| arm | what it is |
|---|---|
| `constant` | predicts the training mean; anything useful must beat it |
| `lasso_classical`, `lasso_vae` | the paper's LASSO baselines (paper: R 0.510, 0.531) |
| `rf_vae` | forest on VAE features, fitted on A, scored on B (paper: R 0.539) |
| `rf_vae_insample` | the same forest fitted and scored on each B FOV |
| `nn_mlp` | one network, no class bins and no forest: MLP on the same features, squared loss on N/C |
| `nn_mlp_log`, `nn_mlp_huber` | the same with log target / Huber loss |
| `nn_mlp_hist` | the same with cross-entropy over 32 equal-count bins of N/C, predicting the expectation |
| `nn_mlp_weighted` | `nn_mlp` with rows weighted by 1 / label density |
| `nn_conv` | `nn_mlp` with the notebook's conv encoder (BatchNorm before the activation, no conv bias) |
| `nn_conv_notebook_bn` | as `nn_conv` but with the notebook's BatchNorm order and conv bias |
| `nn_mlp_heavy_reg` | `nn_mlp` with the notebook's l2 0.01 / l1 0.02 penalties |
| `nn_snapshot`, `nn_snapshot_hist` | `trajl>1`: shared per-snapshot MLP, read out at the last snapshot |
| `nn_tconv`, `nn_tconv_hist` | `trajl>1`: Conv1d along time |
| `nn_transformer`, `nn_transformer_hist` | `trajl>1`: self-attention among the snippet's snapshots |
| `nn_transformer_noattn` | the same block with the attention made an exact identity (prices the attention) |
| `nn_snapshot_attn_hist` | shared encoder + attention pooling: the cheap attention control |

README.md explains what each arm is for.
[results/simulator_tweaks.md](results/simulator_tweaks.md) has what they
scored on simulated data at `trajl=1` (`nn_mlp_hist` best of them, `nn_conv`
cost 0.05 R, `nn_mlp_heavy_reg` collapsed to a constant) and
[results/regression_trajl8.md](results/regression_trajl8.md) at `trajl=8`
with the corrected label.

**Treat every one of those numbers as one fit per arm.** The seed spread at
`trajl=8` is unmeasured and looks large for the attention arms. If you have
the compute, run `tests/seed_spread.py` with 5 seeds before believing any
ranking; that is the measurement we could not make here.

**Please also run this one extra comparison.** On the simulator the largest
single gain was not a model change but a one-line fix in data preparation: the
label is read one frame past the snippet it belongs to, and reading it at the
snippet's own frame added 0.03-0.05 R to *every* arm. To check it on the real
data, copy `bench.json`, set

```json
"features": {"label_frame": "snippet_end", "include_last_window": true}
```

and run the same fast arms into a second output directory:

```bash
python -m ccpredict.benchmark --config bench_labelfix.json --output-dir results_labelfix \
    --arms constant,lasso_vae,rf_vae,nn_mlp,nn_mlp_hist
```

Send both folders. Features are prepared again for this one (different labels),
so it costs one data load on top of the arms.

## One thing to watch in the tracker

`ccpredict` re-runs the tracker's `get_unique_trajectories()` on each loaded
`.obj` (`data.extract_trajectories`). In `jcTrajectory_CP` that method's
`extra_depth` argument defaults to `self.trajl - 1` whenever the object happens
to carry a `trajl` attribute, and where it is non-zero, a trajectory that meets
an already-tracked one is cut `extra_depth` frames *before* the junction rather
than at it -- so consecutive trajectories share up to `trajl-1` cells. That is
deliberate in the tracker (it keeps snippets spanning a junction
constructible), but it would break every grouped split here, because
`preprocessing.trajectory_groups` keys on the trajectory index: two snippets
sharing 7 of 8 snapshots could land on opposite sides of a train/validation
split and the validation curve would be quietly optimistic.

The notebook got `extra_depth = 0`, because it calls
`get_unique_trajectories()` (In [14]) *before* assigning `modelSet[i].trajl`
(In [20]). We now pass `extra_depth=0` explicitly, so a pickle that carries the
attribute cannot change the grouping, and `extract_trajectories` logs a warning
naming the two trajectories if any cell ever turns up in both. **If you see
that warning, stop and tell us** -- the cross-replicate numbers would still be
sound (train and test are different FOVs, with no cells in common), but
anything relying on a within-replicate grouped split would not be, which here
means the networks' early stopping.

You can check a loaded object directly: `hasattr(model, 'trajl')` after
unpickling tells you whether the attribute was saved, which is what would have
triggered it.

The key comparisons on this branch: the best `nn_*` arm vs `lasso_vae` and
`rf_vae` (does a network beat the linear and forest baselines on held-out
cells at all?), `rf_vae` vs `rf_vae_insample` (how much does in-sample fitting
add by itself -- on the simulator it was most of the published gap), and
`nn_transformer_hist` vs `nn_transformer_noattn` vs `nn_snapshot_attn_hist`
(does attention among snapshots earn its parameters, or is a single attention
weight per snapshot enough?). `diagnostics.json` also reports whether
trajectories share cells and how optimistic a random row split is compared
with a held-out FOV.

For the published two-stage comparisons -- `published_insample` vs
`published_crossfit`, and whether the paper's R 0.755 is an in-sample number --
check out branch `ccp_torch`, which still has those arms.
