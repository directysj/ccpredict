# Running the benchmark on the real data

This is for whoever has access to the LI204601 files. It compares the model
published in the preprint (VAE features + CNN class probabilities + RFECV
random forest) with baselines and with honest versions of the same model, all
on the same rows. Your run produces a results folder; send that back.

No code changes are needed, only a config file with the data paths.

## 1. Environment

The environment that ran `vaeFeatCCinfoABwells-NNclfRFreg.ipynb` works:
Python ≥ 3.9, numpy, pandas, scipy, scikit-learn, matplotlib, joblib, csaps,
imbalanced-learn, and PyTorch ≥ 2.5 (new: stage 1 was ported from Keras to
PyTorch; TensorFlow is no longer needed). Any recent PyTorch build works; a CUDA
build is only needed to use a GPU.

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

Also check `pipeline.classifier.device`: `"cpu"` (the default), `"gpu"` (CUDA,
else Apple MPS; fails if PyTorch sees neither) or `"auto"` (CUDA, then MPS, then
CPU). Only the networks use it, and `pipeline.net.device` sets the same thing
for the `nn_*` arms. On a GPU node set both to `"auto"`.

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

**Runtime.** The notebook's settings are slow on CPU: the CNN (batch 20) took
about 40 min per fit in the original run, and the benchmark fits it 8 times. The
RFECV forest with `step=1` refits a 5-fold forest ~210 times and runs 5 times.
Expect most of a day on one CPU node. To cut it:

- `"pipeline": {"regressor": {"rfecv_step": 16}}` makes RFECV about 13× faster;
- forests use every core of the node (`n_jobs: -1`), so request a node with
  many cores; the CNN gains less from extra cores;
- run arm subsets as separate jobs with separate `--output-dir`, e.g.
  `--arms constant,lasso_classical,lasso_vae,rf_vae,rf_vae_insample` (fast, minutes),
  `--arms published_insample,published_crossfit` and
  `--arms published_crossfit_groupedval`; the `nn_*` arms are minutes each and
  can go in the fast job.

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
| `published_insample` | the notebook exactly: CNN trained on A; RFECV forest fitted on the B FOV and scored on the same rows (paper: R 0.755) |
| `published_crossfit` | same CNN; forest fitted on A with out-of-fold CNN probabilities, scored on B |
| `published_crossfit_groupedval` | as above, but the CNN's early stopping uses held-out trajectories instead of the oversampled tail of the training set |
| `nn_mlp` | one network, no class bins and no forest: MLP on the same features, squared loss on N/C |
| `nn_mlp_log`, `nn_mlp_huber` | the same with log target / Huber loss |
| `nn_mlp_hist` | the same with cross-entropy over 32 equal-count bins of N/C, predicting the expectation |
| `nn_mlp_weighted` | `nn_mlp` with rows weighted by 1 / label density |
| `nn_conv` | `nn_mlp` with the notebook's conv encoder (BatchNorm before the activation, no conv bias) |
| `nn_conv_notebook_bn` | as `nn_conv` but with the notebook's BatchNorm order and conv bias |
| `nn_mlp_heavy_reg` | `nn_mlp` with the notebook's l2 0.01 / l1 0.02 penalties |
| `clf_conv` | stage 1 only, the notebook's encoder: 4.2M parameters at trajl=8 |
| `clf_mlp` | stage 1 only, no convolution (270k parameters) |
| `clf_snapshot_shared` | stage 1 only, one MLP per snapshot with weights shared across time (91k) |
| `clf_temporal_conv` | stage 1 only, Conv1d along time with nf input channels (136k) |
| `published_holdout` | stage 2 trained on held-out training trajectories, one classifier fit |

The four `clf_*` arms are one classifier fit each and no forest, so they are the
cheap way to compare architectures on the classification task: the report prints
accuracy, majority-class accuracy, balanced accuracy and macro F1 for each. Their
R is not comparable with the other arms' - a 3-class classifier can only emit
three values, so `y_pred` is the probability-weighted class mean.

The `nn_*` arms are minutes each, not hours, so include them all. README.md
explains what each one is for and [results/simulator_tweaks.md](results/simulator_tweaks.md)
has what they scored on simulated data: `nn_mlp_hist` was the best of them,
`nn_conv` cost 0.05 R, and `nn_mlp_heavy_reg` (the notebook's penalties)
collapsed to a constant.

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
anything relying on a within-replicate grouped split would not be: early
stopping, `fit_on="train_holdout"` and `fit_on="train_crossfit"`.

You can check a loaded object directly: `hasattr(model, 'trajl')` after
unpickling tells you whether the attribute was saved, which is what would have
triggered it.

The key comparisons: `published_insample` vs `published_crossfit` (is the
published R an in-sample number?), `rf_vae` vs `rf_vae_insample` (how much does
in-sample fitting add by itself?), and `published_crossfit` vs `rf_vae` (do the
CNN probabilities help on held-out data?). `diagnostics.json` also reports
whether trajectories share cells, how different the A and B label
distributions are, what the notebook's validation slice consisted of, and how
optimistic a random row split is compared with a held-out FOV.
