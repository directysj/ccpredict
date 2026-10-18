# ccpredict (branch `ccp_torch_reg`)

Predicts the cell-cycle reporter readout (nucleus/cytoplasm ratio, N/C) of single
cells from O2VAE morphology embeddings along their tracked trajectories.
Preprint: doi 10.1101/2025.01.23.634593. MCF10A + EGF; A wells = EGF1, B wells =
EGF2, biological replicates.

**One stage.** A single network regresses the N/C ratio directly from a
trajectory snippet. The published model did it in two: a CNN classified the
ratio into three bins (`<0.9`, `0.9–2.7`, `≥2.7`, the middle one holding ~97% of
rows) and an RFECV random forest then regressed from the class probabilities.
Forcing a continuous skewed target through hard class boundaries is the thing
this branch drops.

**The two-stage model is not here.** `StageOneClassifier`, `StageTwoRegressor`,
the `published_*` and `clf_*` arms, the notebook pipeline runner and the
notebook-parity suite all live on branch **`ccp_torch`**. Checking one out is
the way to run them; nothing on this branch reproduces the notebook end to end.

What remains as a reference to beat: the LASSO and random-forest baselines on
the same rows (`lasso_vae`, `lasso_classical`, `rf_vae`, `rf_vae_insample`,
`constant`).

## Two defaults differ from the notebook

The ground truth N/C belongs to the **last snapshot of the snippet**.

| setting | notebook | here | why |
|---|---|---|---|
| `features.label_frame` | `"start_plus_trajl"` | `"snippet_end"` | the notebook reads the label one frame (15 min) *past* the snippet, so it fits labels shifted into the future |
| `features.include_last_window` | `False` | `True` | `range(traj_len - seg_length)` drops the last window of every trajectory; the tracker's own `get_traj_segments` uses `range(traj_len - seg_length + 1)` |

Worth 0.03–0.05 Pearson R to every arm on the simulator — more than any model
change tried so far. The two interact: `include_last_window` alone changes
nothing, because the last window's label was read past the end of the
trajectory and the row was dropped as unlabelled either way.

## Install and run

```bash
pip install -e .                                        # needs torch>=2.5, csaps
python -m ccpredict.benchmark --write-config bench.json  # edit paths, fovs, trajl
python -m ccpredict.benchmark --config bench.json --output-dir results/
python -m ccpredict.benchmark --config bench.json --arms nn_transformer_hist,lasso_vae
```

```python
from ccpredict import DirectRegressor, NetConfig

cfg = NetConfig(encoder="transformer", pool="last", head="histogram")
model = DirectRegressor(cfg, steps=8).fit(X, y, groups)   # X is (n, trajl*nf)
y_pred = model.predict(X_test)
```

**[RUNNING.md](RUNNING.md) is the guide for running this on the real data.**

**Compute.** CPU by default (`net.device = "cpu"`). `"auto"` picks CUDA, then
Apple Metal (MPS), then CPU; `"gpu"` requires one of them. `net.num_threads`
caps PyTorch's CPU threads. Forest baselines build trees on all cores
(`forest.n_jobs = -1`). Metrics are computed on numpy float64 on the host, never
on the device, because MPS has no float64.

## The model

`DirectRegressor` (`models.py`) wraps the network in `cnn.py` and the shared
training loop. What varies lives in `NetConfig`:

**Heads** — `head="regression"` with `loss` `"mse"` or `"huber"`, optionally on
`target_transform="log"`; or `head="histogram"`, cross-entropy over `n_bins`
equal-count bins predicting `sum_k p_k * mean(y in bin k)`. A regression head's
output bias starts at the mean training target, so epoch 0 is the constant
baseline rather than a drift towards it.

**Encoders** — which axis of the `(trajl × nf)` input the network works along.
A row is `[snapshot 0 features | snapshot 1 features | ...]`, nf each; `data.py`
builds it that way and `benchmark.column_selector` keeps the stride.

| `net.encoder` | what it does |
|---|---|
| `"mlp"` (default) | no convolution; the flat vector into the dense layers |
| `"conv"` | the notebook's width-3 Conv1d along the **flattened** vector |
| `"snapshot_shared"` | one MLP per snapshot, **weights shared across time**, then `pool` |
| `"temporal_conv"` | Conv1d along **time** with nf input channels |
| `"transformer"` | **self-attention among the snapshots** of a snippet, then `pool` |

The notebook's convolution is the odd one out. At `trajl=8`, nf=256 only **14 of
2048** kernel positions straddle a snapshot boundary, so 99.3% of the kernel
mixes arbitrary feature dimensions *inside* one snapshot — and VAE latent
dimensions have no ordering, so dimension 7 is not a neighbour of dimension 8.
It is kept to price it, not because it is sensible.

| encoder | parameters (trajl=8, nf=256) | ms / batch of 256 |
|---|---|---|
| `conv` (notebook) | 4,209,281 | 2035 |
| `mlp` | 270,593 | 5.0 |
| `transformer` | 100,673 | 56.9 |
| `temporal_conv` | 78,465 | 25.0 |
| `snapshot_shared` | 33,089 | 5.7 |

**Pooling** reduces the per-snapshot representation for the dense head:
`"concat"` keeps the time order (one weight block per snapshot), `"mean"`
discards it, `"attention"` learns one weight per snapshot from its content, and
`"last"` takes the final snapshot — the one the label belongs to. Every sequence
encoder ends in the same pool and head, so comparing them compares the encoder
alone.

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
`transformer_layers`, `transformer_ff`, and `transformer_attention`. Two honest
caveats:

- **The sequence is 8 tokens long.** Pairwise attention has little room to find
  something a single attention *pool* over a shared encoder cannot, so
  `nn_snapshot_attn_hist` (`snapshot_shared` + `pool="attention"`, one learned
  weight per snapshot, no query-key term, no extra depth) is in the arm list as
  the control `nn_transformer_hist` has to beat.
  `transformer_attention="none"` goes further: it zeroes and freezes the
  attention output projection, making that sublayer an exact identity while the
  token projection, both LayerNorms and the feed-forward sublayer stay. That is
  the ablation that prices attention itself, as arm `nn_transformer_noattn`.
- **The block internals keep PyTorch's initialisation and carry no l1/l2
  penalty** — only the token projection does, and the feed-forward sublayer uses
  GELU rather than the LeakyReLU the rest of `cnn.py` keeps for Keras
  equivalence. Keras-style per-kernel penalties have no standard counterpart
  inside a transformer block.

## Arms

Each `nn_*` arm differs from `nn_mlp` by exactly the fields its name mentions,
which `tests/test_evaluation.py` checks.

| arm | what it changes |
|---|---|
| `nn_mlp` | MLP on MinMax-scaled VAE features, squared loss on N/C |
| `nn_mlp_log` | fits log(N/C); metrics stay on the raw ratio |
| `nn_mlp_huber` | Huber loss, for the long upper tail |
| `nn_mlp_hist` | cross-entropy over 32 equal-count bins |
| `nn_mlp_weighted` | rows weighted by 1 / label density |
| `nn_conv` | the notebook's conv encoder, BatchNorm before the activation, no conv bias |
| `nn_conv_notebook_bn` | as `nn_conv` with the notebook's BatchNorm order and conv bias |
| `nn_mlp_heavy_reg` | `nn_mlp` with the notebook's l2 0.01 / l1 0.02 penalties |

Sequence arms, for `trajl > 1`. `steps=trajl` is passed by the benchmark from
`features.trajl`:

| arm | what |
|---|---|
| `nn_snapshot` / `nn_snapshot_hist` | shared per-snapshot MLP, `pool="last"`, squared / 32-bin head |
| `nn_tconv` / `nn_tconv_hist` | Conv1d along time, `pool="last"`, squared / 32-bin head |
| `nn_tconv_concat` | as `nn_tconv` with `pool="concat"`, to price the pooling |
| `nn_transformer` / `nn_transformer_hist` | self-attention among the snapshots, `pool="last"` |
| `nn_transformer_noattn` | the same block with the attention made an exact identity |
| `nn_snapshot_attn_hist` | shared encoder + `pool="attention"`: the cheap attention control |

`DEFAULT_ARMS` in `arms.py` is a subset — at `trajl=8` running every arm is
hours. `ALL_ARMS` has the rest; `--arms` takes any of them by name.

Three departures from the notebook are defaults, each with an ablation arm so
the cost of the original choice is visible: **no convolution** along the latent
index (`nn_conv` prices it); **BatchNorm before the activation**, and therefore
no conv bias, since BatchNorm subtracts the batch mean and a per-channel
constant in front of it cannot change any output (`nn_conv_notebook_bn`); and
**light regularisation**, because the notebook's l2 0.01 / l1 0.02 is a penalty
of ~13 at initialisation against a loss of ~1 and collapses a regression net to
a constant (`nn_mlp_heavy_reg`).

## Honest evaluation: the benchmark

- **Question (paper):** train on EGF1 (A wells), test on each EGF2 FOV (B wells)
  separately: `scheme="cross_replicate"`. `"lofo"` (leave one FOV out) and
  `"fixed"` (A→B2_4) are also available.
- **Grouping:** anything chosen on held-out data inside the training FOVs (early
  stopping, LASSO alpha) holds out whole trajectories or whole FOVs, never
  random rows.
- **Nothing is scored on rows it was fitted on**, except arms whose names say
  `_insample` — they exist to measure that effect.
- **Metrics per test FOV:** MSE and Pearson R, plus R², Spearman, MAE and R² on
  log N/C, calibration slope, per-trajectory R (paper Fig. 6), and 95% intervals
  that resample whole trajectories.
- **R is not R².** The reported `r2` is the coefficient of determination, which
  also charges for being wrong in slope or intercept; R is invariant to that.
  On the simulator the nets lose 0.02–0.05 to calibration and the LASSO almost
  nothing. Use R to compare models across replicates, `r2` for the error you
  would make on a fresh one. See `results/regression_trajl8.md`.
- **Bootstrap intervals resample rows, not training runs.** Two arms with
  non-overlapping intervals can still be indistinguishable once refitted. Run
  `tests/seed_spread.py` before ranking two arms.

## Synthetic data

There is no data access from here, so development uses a simulator calibrated to
what is known about the real data (`ccpredict/simulate.py`; its docstring lists
every assumption):

```bash
python -m ccpredict.simulate sim/ --scale 0.25    # writes the real file layout
```

It reproduces the B2_4 N/C histogram (digitised from the notebook figure), an
HDHB-like trace per cycle, size growth and mitotic rounding, an A/B replicate
shift, celltraj-style trajectories, and the paper's snapshot baselines (LASSO R
0.53 VAE, 0.51 classical). It is for correctness and for sizing evaluation
artefacts. **It cannot say whether a modelling change helps on real cells.**

## Other things to know

- **Trajectory extraction pins `extra_depth=0`.** The real
  `jcTrajectory_CP.get_unique_trajectories` defaults it to `self.trajl - 1` when
  the object carries a `trajl` attribute, which makes consecutive trajectories
  share up to `trajl-1` cells and would break every grouped split, since
  `trajectory_groups` keys on the trajectory index. The notebook got 0 (it
  extracts before assigning `trajl`); we pass 0 explicitly and warn if any cell
  turns up in two trajectories. See RUNNING.md.
- **`SplitContext.trajl` has no usable default.** 1 silently collapses the time
  axis — a sequence encoder becomes a plain net over the flattened row and a
  transformer gets a single token. It still trains and still reports a number.
- **Trajectory alignment is assumed, not checked.** Trajectory *k* of the tracked
  pickle must be trajectory *k* of the reporter `.dat`. The package raises if
  indices run out of range and warns if the counts differ, but it cannot detect
  a reordering.
- **Dead conv biases.** Inputs are MinMax-scaled to [0, 1], so a first-layer
  filter whose three weights share a sign never leaves LeakyReLU's linear
  branch, and a BatchNorm right after it cancels that filter's bias. The bias
  has zero gradient; Adam normalises the float noise in it into full
  learning-rate steps, so these biases random-walk without changing any output
  (this happened in Keras too). Putting BatchNorm before the activation, or
  dropping the conv bias, removes them — which is why both are defaults here.
- **QC spline grid** (`qc.evaluate_on_frames`, default `False`): the spline is
  evaluated on a 10,000-point grid and indexed by frame number, so frames
  136–190 are compared with the smoothed count at frames ~2.6–3.7. Switching it
  on requires recalibrating `qc.std_cut` — the scores drop about tenfold.

## From the notebook

Cells are cited by the `In [n]` label Jupyter shows next to them. Only the data
path is on this branch; the modelling rows are on `ccp_torch`.

| Notebook | What | Package |
|---|---|---|
| In [4]–[7] | FOV list, load `.obj` pickles | `PathsConfig`, `data.load_tracked_fov` |
| In [8]–[10] | cell-count spline QC | `data.cell_count_noise`, `data.qc_filter` (`QCConfig`) |
| In [10]–[11] | `Xf` NaN→0, `Xf_com[:,0]` z-score | `data.clean_handcrafted_features`, `data.standardize_com_displacement` |
| In [12]–[13] | concat `Xf` then split back | removed: it returned the same arrays |
| In [14] | `get_unique_trajectories` | `data.extract_trajectories` |
| In [15] | load VAE embeddings | `data.load_vae_embeddings`, `data.per_cell_features` |
| In [16]–[18] | sliding-window snippets, features, label frames | `data.sliding_snippets`, `data.build_fov_snippets` (vectorised; the notebook's `np.append` loop was O(n²)) |
| In [19]–[21] | parse reporter `.dat`, match labels | `data.read_reporter_file`, `data.match_labels` |
| In [22]–[23] | trimming, MinMax scaling, train/test FOVs | `preprocessing.trim_by_histogram`, `preprocessing.build_train_test` |
| In [25]–[44] | class binning, CNN, RFECV forest | **branch `ccp_torch`** |

## Verification

```bash
pytest tests/                                   # data rewrites vs notebook functions, metrics,
                                                # splits, simulator, network determinism, benchmark smoke test
python tests/synthetic.py /tmp/fx               # small fixture
python tests/seed_spread.py BENCH.json out --arms nn_snapshot_hist,nn_transformer_hist --seeds 5
python tests/ceiling.py sim/simulation.json     # how much signal is there? (simulated data only)
```

The data-preparation path is checked bit for bit against verbatim copies of the
notebook's own functions (`tests/test_equivalence.py`): snippet construction at
several `trajl`, the last-window rule, reporter parsing and label matching, and
histogram trimming. The Keras-vs-PyTorch parity suite tested the CNN classifier
and went with it to `ccp_torch`.

`tests/ceiling.py` reports what a forest and an MLP extract from noise-free
embeddings, from the generative state and from the label itself. **Those are not
upper bounds** — they bound two model classes at `trajl=1`, and a transformer
over the snippet's snapshots has already passed the "noisy embeddings" row. Read
a gap as suggestive, never as proof nothing is left.
