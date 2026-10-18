# ccpredict

Predicts the cell-cycle reporter readout (nucleus/cytoplasm ratio, N/C) of single
cells from O2VAE morphology embeddings along their tracked trajectories. It is a
package version of `vaeFeatCCinfoABwells-NNclfRFreg.ipynb`.

The published model has two stages:
1. **Stage 1:** a CNN+DNN classifies the N/C ratio into bins (default `<0.9`, `0.9–2.7`, `≥2.7`).
2. **Stage 2:** a random forest with RFECV regresses the N/C ratio from the
   scaled features plus the stage-1 class probabilities.

With default settings the package reproduces the notebook exactly, defects
included. Each defect that changes numbers has an off-by-default switch,
listed below.

## This branch: `ccp_torch_reg`

**One-stage regression.** The two-stage model forces the N/C ratio through three
hard class boundaries and then regresses from the class probabilities; this
branch asks what happens when a single network predicts the ratio directly, with
no bins and no forest. The two-stage path is still here, reproduced exactly, and
`published_crossfit` stays in the default run as the reference to beat — but the
`nn_*` arms are what the branch is about.

Two defaults are flipped here, because the ground truth N/C belongs to the **last
snapshot of the snippet** and the notebook's settings do not reflect that:

| setting | notebook / `ccp_torch` | here | why |
|---|---|---|---|
| `features.label_frame` | `"start_plus_trajl"` | `"snippet_end"` | the notebook reads the label one frame *past* the snippet |
| `features.include_last_window` | `false` | `true` | the tracker's own `get_traj_segments` includes it (`range(traj_len - seg_length + 1)`) |

Both are worth 0.03–0.05 R on the simulator, and `tests/parity/*` pin the old
values explicitly so notebook parity still means parity.

`DEFAULT_ARMS` (in `arms.py`) is a regression-focused subset, not every
registered arm — at `trajl=8` running them all is hours. The protocol
ablations, the `clf_*` classification arms and the loss ablations stay
registered and run by name with `--arms`.

## Install and run

```bash
pip install -e .                                   # needs torch>=2.5, csaps, imbalanced-learn
python -m ccpredict --write-config config.json     # default = notebook settings; edit paths
python -m ccpredict --config config.json --output-dir results/
```

```python
from ccpredict import PipelineConfig, CellCyclePipeline
from ccpredict.models import StageOneClassifier

pipe = CellCyclePipeline(PipelineConfig.from_json("config.json"))
data = pipe.prepare_data()              # load, QC, snippets, labels
tt = pipe.build_train_test(data)        # concat train FOVs, trim, scale
clf, clf_out = pipe.run_classifier(tt)  # stage 1
reg, reg_out = pipe.run_regressor(tt, clf, clf_out.proba_test)  # stage 2

# Reuse a trained stage 1 (about 40 min on the real data):
clf = StageOneClassifier.load("results/models/B2_4_trajl1_VAE256Cls3", pipe.config.classifier)
clf, clf_out = pipe.run_classifier(tt, classifier=clf)
```

**Stage 1 is PyTorch.** The notebook's Keras model and training loop are
reproduced in `ccpredict/cnn.py` with every Keras default the notebook relied
on made explicit (Glorot init, BatchNorm momentum 0.99 / eps 1e-3, LeakyReLU
0.3, L1/L2 penalties in the loss and in val_loss, Keras' Adam, `validation_split`,
Keras 3 early stopping). TensorFlow is no longer needed except for the parity
checks in `tests/parity` (`pip install -e .[parity]`). How the port was
checked is under Verification.

**Compute.** Only the stage-1 CNN uses PyTorch, and it runs on the CPU by
default (`classifier.device = "cpu"`). `"auto"` picks CUDA, then Apple Metal
(MPS), then CPU; `"gpu"` requires CUDA or MPS. `classifier.num_threads` caps
PyTorch's CPU threads (default: PyTorch's own choice, all physical cores).
Old configs with `tf_intra_op_threads` still load; it is read as `num_threads`.
Random forests build their trees in parallel on all cores
(`regressor.n_jobs = -1`; near-linear: 8.1 s on 1 core, 2.2 s on 4). RFECV runs
its CV folds one at a time (`regressor.rfecv_n_jobs = 1`) so each forest gets
every core; running folds in parallel instead caps RFECV at 5 busy cores.
Both give the same selected features. The CNN gains less from extra cores: its
steps are tiny, so batch size matters more than core count.

Outputs under `output_dir`: `figs/*.png`, `metrics/performance_<tag>.json`,
`models/<tag>/` (`stage1_classifier.pt` state dict, history, joblib regressor). The tag is
`<testFOV>_trajl<L>_VAE<d>Cls<k>`, the same naming as the notebook's savefig paths.

## Honest evaluation: the benchmark

`python -m ccpredict.benchmark` compares the published model with baselines and
with corrected protocols, on the same rows. **[RUNNING.md](RUNNING.md) is the
guide for running it on the real data.**

- **Question (paper):** train on EGF1 (A wells), test on each EGF2 FOV (B wells)
  separately: `scheme="cross_replicate"`. `"lofo"` (leave one FOV out) and
  `"fixed"` (the notebook's A→B2_4) are also available.
- **Grouping:** anything chosen on held-out data inside the training FOVs (CNN
  early stopping, LASSO alpha, out-of-fold stacking) holds out whole
  trajectories or whole FOVs, never random rows.
- **Metrics per test FOV:** the paper's MSE and Pearson R, plus R², Spearman,
  MAE and R² on log N/C, calibration slope, per-trajectory R (paper Fig. 6),
  and 95% intervals that resample whole trajectories. For the classifier:
  accuracy next to the majority-class accuracy, balanced accuracy and macro F1.
- **Arms:** `constant`, `lasso_classical`, `lasso_vae`, `rf_vae`,
  `rf_vae_insample`, `published_insample` (the notebook), `published_crossfit`,
  `published_crossfit_groupedval`, plus the one-stage networks below. See
  `ccpredict/arms.py` and RUNNING.md.
- **Diagnostics** (`diagnostics.json`): trajectory overlap, label distribution
  per FOV and after trimming, the exact composition of the notebook's Keras
  validation slice, and forest R under random-row, trajectory, FOV and test
  splits.

## Encoders: which axis of the input the network works along

A row of features is `trajl` snapshots of one cell concatenated,
`[snapshot 0 features | snapshot 1 features | ...]` with nf features each
(`data.py` builds it that way; `benchmark.column_selector` keeps the stride).
The nf axis is a set of channels with no order. The `trajl` axis is time, 15 min
apart. Which one an encoder works along is the whole difference:

| `classifier.encoder` / `net.encoder` | what it does |
|---|---|
| `"conv"` (the notebook, default) | width-3 Conv1d along the **flattened** vector |
| `"mlp"` | no convolution; the flat vector into the dense layers |
| `"snapshot_shared"` | one MLP per snapshot, **weights shared across time**, then `pool` |
| `"temporal_conv"` | Conv1d along **time** with nf input channels |
| `"transformer"` | **self-attention among the snapshots** of a snippet, then `pool` |

The notebook's convolution is the odd one out. At `trajl=8`, nf=256 only **14 of
2048** kernel positions straddle a snapshot boundary, so 99.3% of the kernel
mixes arbitrary feature dimensions *inside* one snapshot, and the few that cross
mix the tail dimensions of one snapshot with the head dimensions of the next.
It is also where the parameters go, because the flatten after pooling feeds one
dense layer:

| encoder | parameters (trajl=8, nf=256, 3 classes) |
|---|---|
| `conv` (notebook) | **4,198,595** |
| `mlp` | 270,787 |
| `temporal_conv` | 136,131 |
| `transformer` | 100,673 |
| `snapshot_shared` | 90,627 |

Against ~28k training rows with 97% of them in one class. `pool` reduces the
per-snapshot representation for the dense head: `"concat"` keeps the time order
(one weight block per snapshot), `"mean"` discards it, `"attention"` learns one
weight per snapshot from its content, `"last"` takes the final snapshot — the
one the label belongs to. Every sequence encoder ends in the same pool and head,
so comparing them compares the encoder alone.

### The transformer encoder

Each snapshot is one token: its nf features projected to `transformer_dim`, plus
a **learned** positional embedding over the `trajl` slots (the axis is short and
fixed, so absolute embeddings already span every relative offset), then
`transformer_layers` pre-norm blocks of multi-head self-attention and a GELU
feed-forward sublayer. Attention runs over the snapshots, so snapshot *t* can
read any other snapshot; `snapshot_shared` never mixes snapshots at all and
`temporal_conv` mixes only within its kernel. `pool="last"` then reads out the
token for the snapshot the label belongs to, *after* it has attended to the rest
of the snippet.

Knobs: `transformer_dim` (must divide by `transformer_heads`), `transformer_heads`,
`transformer_layers`, `transformer_ff`. Two honest caveats:

- **The sequence is 8 tokens long.** Pairwise attention has little room to find
  something that a single attention *pool* over a shared encoder cannot, so
  `nn_snapshot_attn_hist` (`snapshot_shared` + `pool="attention"`, one learned
  weight per snapshot, no query-key term, no extra depth) is in the arm list as
  the control `nn_transformer_hist` has to beat. Attention blocks that do not
  beat it have bought nothing but parameters.
- **The block internals keep PyTorch's own initialisation and carry no l1/l2
  penalty** — only the token projection does, and the feed-forward sublayer uses
  GELU rather than the LeakyReLU the rest of the file keeps for Keras parity.
  Keras-style per-kernel penalties have no standard counterpart inside a
  transformer block, so inventing one would make the comparison worse, not
  better. The dense head after the pool is unchanged.

`clf_conv`, `clf_mlp`, `clf_snapshot_shared`, `clf_temporal_conv` and
`clf_transformer` are
benchmark arms that train **stage 1 only** (one fit, no forest) and report
accuracy, balanced accuracy and macro F1 next to the regression metrics, with
`y_pred` the probability-weighted class mean. That is the cheap way to ask which
architecture classifies best. The `nn_*` arms take the same `encoder` switch.

## Stage 2: where it gets its training rows

`regressor.fit_on` decides both which rows stage 2 is fitted on and whose
stage-1 probabilities it sees. Only the last two are honest about both:

| `fit_on` | stage 2 trained on | stage-1 probabilities | cost |
|---|---|---|---|
| `"test"` (notebook default) | the test FOV, then scored on the same rows | in-sample | 1 fit |
| `"train"` | training FOVs | stage 1's own training rows, over-confident | 1 fit |
| `"train_crossfit"` | training FOVs | out-of-fold, one stage 1 per held-out FOV | 1 + K fits |
| `"train_holdout"` | a held-out share of training trajectories | out-of-sample from one stage 1 | 1 fit |

`train_holdout` splits the training block by whole trajectories
(`stage2_fraction`, `stage2_seed`), trains stage 1 on one part and stage 2 on the
other, and scores the test FOVs with that same stage 1. It costs one classifier
fit instead of K, at the price of less data for each stage. Arms:
`published_crossfit`, `published_holdout`, and `_groupedval` variants of both.

## One-stage neural regressors

The published model turns a continuous, strongly skewed target into three bins
of which the middle one holds ~97% of the rows, then feeds the class
probabilities to a forest. The `nn_*` arms replace both stages with a single
network that predicts the ratio itself. Everything they share lives in
`NetConfig` (`ccpredict/config.py`), and each arm differs from `nn_mlp` by
exactly the fields its name mentions, which `tests/test_evaluation.py` checks:

| arm | what it changes |
|---|---|
| `nn_mlp` | MLP on the same MinMax-scaled VAE features, squared loss on N/C |
| `nn_mlp_log` | fits log(N/C); metrics stay on the raw ratio |
| `nn_mlp_huber` | Huber loss, for the long upper tail |
| `nn_mlp_hist` | cross-entropy over 32 equal-count bins, predicting `sum_k p_k * mean(y in bin k)` |
| `nn_mlp_weighted` | rows weighted by 1 / label density (the regression analogue of oversampling) |
| `nn_conv` | the notebook's conv encoder, BatchNorm before the activation, no conv bias |
| `nn_conv_notebook_bn` | as `nn_conv` with the notebook's BatchNorm order and conv bias |
| `nn_mlp_heavy_reg` | `nn_mlp` with the notebook's l2 0.01 / l1 0.02 penalties |

Sequence arms, for `trajl > 1` (all one-stage: no class bins, no forest). Pass
`steps=trajl`, which the benchmark does from `features.trajl`:

| arm | what |
|---|---|
| `nn_snapshot` / `nn_snapshot_hist` | shared per-snapshot MLP, `pool="last"`, squared / 32-bin head |
| `nn_tconv` / `nn_tconv_hist` | Conv1d along time, `pool="last"`, squared / 32-bin head |
| `nn_tconv_concat` | as `nn_tconv` with `pool="concat"`, to price the pooling |
| `nn_transformer` / `nn_transformer_hist` | self-attention among the snapshots, `pool="last"` |
| `nn_snapshot_attn_hist` | shared encoder + `pool="attention"`: the cheap attention control |

Three departures from the notebook are defaults here, each with its own
ablation arm so the cost of the original choice is visible:

- **No convolution.** The notebook convolves a (3,) kernel along the *latent
  index* of the VAE embedding, an index with no ordering: dimension 7 is not a
  neighbour of dimension 8. Weight sharing across the 256 of them encodes an
  adjacency that does not exist. `nn_conv` keeps the convolution to price it.
- **BatchNorm before the activation**, and therefore no conv bias: BatchNorm
  subtracts the batch mean, so a per-channel constant in front of it cannot
  change any output. `nn_conv_notebook_bn` keeps the notebook's order.
- **Light regularisation.** The notebook's l2 0.01 / l1 0.02 on every layer is
  a penalty of ~13 at initialisation against a cross-entropy of ~1.1, so 92% of
  the loss is the penalty. `nn_mlp_heavy_reg` keeps it.

Early stopping holds out whole trajectories (`validation="grouped_holdout"`),
and a regression head's output bias starts at the mean training target, so
epoch 0 is the constant baseline rather than a drift towards it.

**What these scored on the simulator** is in
[results/simulator_tweaks.md](results/simulator_tweaks.md) (and, at `trajl=8`
with the corrected label, [results/regression_trajl8.md](results/regression_trajl8.md)), with the seed
spread of each arm (differences below ~0.01 R are noise) and a measurement of
how much signal is left to win. In short: `nn_mlp_hist` was the best of them,
the convolution costs 0.05 R and 18x the runtime, the notebook's penalties
collapse a regression net to a constant, and the single largest gain available
anywhere was fixing `features.label_frame`, not changing the model.

## Synthetic data

There is no data access from here, so development uses a simulator calibrated
to what is known about the real data (`ccpredict/simulate.py`, docstring lists
every assumption):

```bash
python -m ccpredict.simulate sim/ --scale 0.25    # writes the real file layout
```

It reproduces the B2_4 N/C histogram (digitised from the notebook figure), an
HDHB-like trace per cycle, size growth and mitotic rounding, an A/B replicate
shift, celltraj-style trajectories, and the paper's snapshot baselines
(LASSO R 0.53 VAE, 0.51 classical). It is for correctness and for sizing
evaluation artefacts. **It cannot say whether a modelling change helps on real
cells**; that needs the benchmark run on the real data.

## Notebook to package

Cells are cited by the `In [n]` label Jupyter shows next to them. The RFECV fit
and every cell after it have no label: they were never executed in the saved
notebook.

| Notebook | What | Package |
|---|---|---|
| In [1]–[3] | imports | each module imports what it uses; `umap`, `h5py`, `seaborn`, `PCA`, `GridSearchCV` etc. were never used and are dropped |
| In [4]–[7] | FOV list, load `.obj` pickles | `PathsConfig`, `data.load_tracked_fov` |
| In [8]–[10] | cell-count spline QC | `data.cell_count_noise`, `data.qc_filter` (`QCConfig`) |
| In [10]–[11] | `Xf` NaN→0, `Xf_com[:,0]` z-score | `data.clean_handcrafted_features`, `data.standardize_com_displacement` |
| In [12]–[13] | concat `Xf` then split back | removed: it returned the same arrays |
| In [14] | `get_unique_trajectories` | `data.extract_trajectories` |
| In [15] | load VAE embeddings | `data.load_vae_embeddings`, `data.per_cell_features` |
| In [16]–[18] | sliding-window snippets, features, label frames | `data.sliding_snippets`, `data.build_fov_snippets` (vectorised; the notebook's `np.append` loop was O(n²)) |
| In [19]–[21] | parse reporter `.dat`, match labels | `data.read_reporter_file`, `data.match_labels` (dict lookup instead of `np.where` per snippet) |
| In [22]–[23] | histogram trimming, MinMax scaling, train/test FOVs | `preprocessing.trim_by_histogram`, `preprocessing.build_train_test` (`SplitConfig`) |
| In [24] | label histograms | `evaluation.plot_label_distributions` |
| In [25], [32] | class binning | `preprocessing.discretize` |
| In [27], [33]–[37] | CNN, compile, oversample, fit, predict | `models.build_cnn_classifier`, `models.StageOneClassifier` (`ClassifierConfig`) |
| In [38]–[41] | metrics, confusion matrix, history plot | `evaluation.classification_metrics`, `plot_confusion_matrix`, `plot_training_history` |
| In [42], [44] + unexecuted | stack probabilities, RFECV forest, fit, predict | `models.stack_features`, `models.StageTwoRegressor` (`RegressorConfig`) |
| unexecuted | regression plots, RFECV curve | `evaluation.plot_regression`, `plot_rfecv_scores` |
| unexecuted, last cell | multi-GPU `MirroredStrategy` | not ported (see below) |

## Behaviour preserved from the notebook (switches)

**Two defaults on this branch no longer match the notebook.** The ground truth
N/C belongs to the **last snapshot of the snippet**, so `label_frame` defaults to
`"snippet_end"` and `include_last_window` to `True`. The notebook read the label
at `frame(start) + trajl`, one frame (15 min) *past* the window, which is a
defect rather than a design choice: it fits labels shifted into the future, and
fixing it is worth 0.03-0.05 Pearson R to every arm on simulated data. The
notebook's values remain available and are pinned explicitly by `tests/parity`,
which still reproduces the notebook bit for bit.

| Switch | Notebook (default) | Alternative | What the default does |
|---|---|---|---|
| `regressor.fit_on` | `"test"` | `"train_crossfit"` (or `"train"`) | The forest is fitted on the test FOV and scored on the same rows, so stage-2 MSE/R/R² are training-set numbers. `"train_crossfit"` fits it on the training FOVs with out-of-fold CNN probabilities; `"train"` uses the CNN's in-sample probabilities, which are over-confident. |
| `classifier.validation` | `"keras_tail"` | `"grouped_holdout"` | The oversampler appends duplicates at the end, grouped by class, and Keras's `validation_split` takes the last 20%. Validation is therefore copies of minority-class training rows, and early stopping (which restores the best weights) is driven by them. `"grouped_holdout"` holds out 20% of trajectories before oversampling. |
| `qc.evaluate_on_frames` | `False` | `True` | The spline is evaluated on a 10,000-point grid and indexed by frame number, so frames 136–190 are compared with the smoothed count at frames ~2.6–3.7. The score is then roughly a coefficient of variation of late counts. If you switch this on, recalibrate `qc.std_cut`: the scores drop about tenfold. |
| `features.include_last_window` | `True` (branch default; notebook used `False`) | `False` | `range(traj_len - seg_length)` drops the last window of every trajectory, and trajectories exactly `trajl` long yield nothing. On its own this switch changes nothing: the last window's label is read past the end of the trajectory, so that row is dropped as unlabelled either way. Fixing `label_frame` recovers those rows as well. |
| `features.label_frame` | `"snippet_end"` (branch default; notebook used `"start_plus_trajl"`) | `"start_plus_trajl"` | The label is read at `frame(first cell) + trajl`, one frame after the snippet's last cell. With `trajl=1` that is the next frame's reporter value, so the model is scored on predicting the reporter 15 min into the future from the current morphology. **On the simulator this costs 0.03-0.05 R for every arm** ([results/simulator_tweaks.md](results/simulator_tweaks.md)), which is more than any model change tried so far. |
| `classifier.seed` | `None` | an int | The notebook sets no seed, so its training is not reproducible. |

`regressor.use_rfecv = False` replaces RFECV by a single forest on all features
(not a notebook defect, just much faster).

## Other things to know (no switch, because changing them changes the model)

- **Convolution over latent dimensions.** The input `(d,)` is reshaped to
  `(d, 1, 1)` and convolved with a `(3, 1)` kernel, which is a 1-D
  convolution along the latent index. VAE latent dimensions have no ordering,
  so locality and max-pooling along that axis carry no meaning. For
  `trajl > 1` the features are laid out `[t0 dims, t1 dims, …]`, so the
  kernel still mixes latent dimensions, not time steps.
- **Weighted precision with `zero_division=np.nan`.** Classes that are never
  predicted are left out of the weighted average, which pushes precision up.
  This is likely why the notebook shows 98% precision at 49% accuracy.
  `classification_metrics` also returns per-class values and prediction counts.
- **Trajectory alignment is assumed, not checked.** Trajectory *k* of the
  tracked pickle must be trajectory *k* of the reporter `.dat`. The package
  raises if indices run out of range and warns if the counts differ, but it
  cannot detect a reordering.
- **RFECV cost.** `step=1` from 259 features down to 50 means 210 eliminations,
  each a 5-fold forest fit. Stage 2 was never executed in the saved notebook,
  so there are no reference numbers for it from the real data.
- **Not ported: multi-GPU cell.** It uses the test FOV as validation data and
  calls `to_categorical(y_test)` on the continuous N/C ratio instead of class
  labels.
- **Plot labels corrected.** These change no numbers: the RFECV x-axis now
  starts at `min_features_to_select` (the notebook started at 1), its y-axis is
  labelled with the scoring metric (not "accuracy"), and the regression plots
  are titled as the forest (not "CNN+DNN").

## Verification

```bash
pytest tests/                                   # rewrites vs notebook functions, metrics, splits,
                                                # diagnostics, simulator, network determinism, benchmark smoke test
python tests/synthetic.py /tmp/fx               # small fixture used by the parity checks
python tests/seed_spread.py BENCH.json out --arms nn_mlp,nn_mlp_hist --seeds 5   # is a gap bigger than the seed?
python tests/ceiling.py sim/simulation.json     # how much signal is there to win? (simulated data only)
# The rest needs TensorFlow + Keras 3 as well (pip install -e .[parity]):
python tests/parity/notebook_reference.py vaeFeatCCinfoABwells-NNclfRFreg.ipynb /tmp/fx /tmp/ref.npz
python tests/parity/package_run.py /tmp/fx /tmp/pkg.npz --stage1-from /tmp/ref.npz
python tests/parity/compare.py /tmp/ref.npz /tmp/pkg.npz
python tests/parity/benchmark_arm_parity.py /tmp/fx /tmp/ref.npz /tmp/armcheck   # benchmark arm = notebook
pytest tests/parity/test_torch_port.py          # PyTorch CNN = Keras CNN, step by step
python tests/parity/torch_vs_keras.py BENCH.json /tmp/tvk --seeds 10 --max-train-rows 8000
```

**Everything except the CNN is checked bit for bit against the notebook.**
`notebook_reference.py` executes the notebook's own code cells. It changes only
the paths, the epoch count, the seed and the forest size, and asserts that each
of those edits matches exactly once. With `--stage1-from`, `package_run.py`
feeds the notebook's own class probabilities into stage 2, and on the synthetic
data every other array matches bit for bit: QC scores, per-FOV features and
labels, trimmed and scaled sets, oversampled set, RFECV support and metrics. The
one exception is the forest's predictions, which agree to ~1e-15.
`RandomForestRegressor.predict` with `n_jobs>1` sums tree outputs in thread
order, so repeated calls on one fitted model differ by that much too.
`benchmark_arm_parity.py` does the same for the benchmark's `published_insample`
arm, so the benchmark's "published" row is the notebook's model, not a
re-implementation that might differ.

**The CNN is checked at two levels.** (Before the port, the Keras version
matched the notebook bit for bit; it is kept in `tests/parity/keras_reference.py`.)

1. *Exact, same weights* (`test_torch_port.py`): with Keras weights copied into
   the PyTorch model, the forward pass agrees to 2e-6, the L1/L2 penalty and
   every gradient to float32 precision, and ten Adam steps without dropout give
   the same losses, weights and BatchNorm statistics. The whole training loop
   (`cnn.train` against Keras' `model.fit`, 12 epochs with validation and early
   stopping, dropout and shuffling off so both see the same batches) gives the
   same per-epoch loss and val_loss to 0.15% and the same predictions to 2e-3.
   With dropout on, the spread of outputs over dropout masks matches Keras'
   (same weights, 400 masks each; sd ratio 1.006). The early-stopping rule is
   run against Keras' own callback on 50 random val_loss sequences and makes
   the same stop and restore decisions.
2. *Statistical, independent training* (`torch_vs_keras.py`): dropout masks,
   shuffling order and float kernels differ between frameworks, so single runs
   cannot match. Both were trained with seeds 0-9 on simulator data under the
   notebook protocol (keras_tail validation, patience 15, up to 50 epochs;
   8,010 EGF1 training rows, 40,720 EGF2 test rows). Full table:
   [tests/parity/results/torch_vs_keras_sim.md](tests/parity/results/torch_vs_keras_sim.md).
   - Training is indistinguishable: train loss, val_loss and accuracy at every
     epoch, best val_loss (0.296 vs 0.303, p = 0.19), epochs run.
   - So is what the network learns: the Spearman correlation of
     p(class 2) - p(class 0) with N/C on the test replicate is 0.235 (Keras) vs
     0.241 (PyTorch), p = 0.91, and seed-averaged predictions agree as well
     across frameworks (r = 0.874) as within one (0.882 Keras, 0.899 PyTorch).
   - Confidence in class 1 on the test replicate came out lower in PyTorch:
     mean p(class 1) 0.86 vs 0.92, accuracy 0.871 vs 0.924, balanced accuracy
     0.401 vs 0.380, log-loss 0.37 vs 0.28 (p 0.02-0.04, about one
     within-framework sd; one effect seen through several metrics). Most of it
     follows the epoch that early stopping restores, which is noisy (Spearman
     0.66 with mean p(class 1), pooled); with that epoch in the model the
     framework term is about half as large and not significant (p = 0.08).
     Every component matches exactly or in distribution (level 1), so this is
     read as chance; another 20 seeds would settle it.
   - Seed alone moves test accuracy by about 5 points in either framework
     (sd 0.044 Keras, 0.057 PyTorch), because early stopping watches
     duplicates of ~20 class-2 rows (the keras_tail defect). That is larger
     than any framework difference.
   - PyTorch was ~20% slower per epoch on that 2-core machine (360 vs 428 s
     per run).

One finding from level 1 applies to the notebook's architecture itself: inputs
are MinMax-scaled to [0, 1], so a first-layer filter whose three weights share
a sign never leaves LeakyReLU's linear branch, and the BatchNorm right after it
cancels that filter's bias. The bias has zero gradient; Adam normalises the
float noise in it into full learning-rate steps, so these biases random-walk
(in Keras too) without changing any output. Harmless, but those are dead
parameters; putting BatchNorm before the activation, or dropping the conv bias,
removes them.
