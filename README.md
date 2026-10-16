# ccpredict

Predicts the cell-cycle reporter readout (nucleus/cytoplasm ratio, N/C) of single
cells from O2VAE morphology embeddings along their tracked trajectories. It is a
package version of `vaeFeatCCinfoABwells-NNclfRFreg.ipynb`.

Two stages:
1. **Stage 1:** a CNN+DNN classifies the N/C ratio into bins (default `<0.9`, `0.9–2.7`, `≥2.7`).
2. **Stage 2:** a random forest with RFECV regresses the N/C ratio from the
   scaled features plus the stage-1 class probabilities.

With default settings the package reproduces the notebook exactly, defects
included. Each defect that changes numbers has an off-by-default switch,
listed below.

## Install and run

```bash
pip install -e .                                   # needs csaps, imbalanced-learn, keras + tensorflow
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

**Compute.** Only the stage-1 CNN uses TensorFlow, and it runs on the CPU by
default (`classifier.device = "cpu"`, which also hides an Apple Metal GPU).
Set `"auto"` to let TensorFlow use a GPU when one is present, or `"gpu"` to
require one. Random forests build their trees in parallel on all cores
(`regressor.n_jobs = -1`; near-linear: 8.1 s on 1 core, 2.2 s on 4). RFECV runs
its CV folds one at a time (`regressor.rfecv_n_jobs = 1`) so each forest gets
every core; running folds in parallel instead caps RFECV at 5 busy cores.
Both give the same selected features. The CNN gains less from extra cores: its
steps are tiny, so batch size matters more than core count.

Outputs under `output_dir`: `figs/*.png`, `metrics/performance_<tag>.json`,
`models/<tag>/` (Keras model, history, joblib regressor). The tag is
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
  `published_crossfit_groupedval`. See `ccpredict/arms.py` and RUNNING.md.
- **Diagnostics** (`diagnostics.json`): trajectory overlap, label distribution
  per FOV and after trimming, the exact composition of the notebook's Keras
  validation slice, and forest R under random-row, trajectory, FOV and test
  splits.

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

| Switch | Notebook (default) | Alternative | What the default does |
|---|---|---|---|
| `regressor.fit_on` | `"test"` | `"train_crossfit"` (or `"train"`) | The forest is fitted on the test FOV and scored on the same rows, so stage-2 MSE/R/R² are training-set numbers. `"train_crossfit"` fits it on the training FOVs with out-of-fold CNN probabilities; `"train"` uses the CNN's in-sample probabilities, which are over-confident. |
| `classifier.validation` | `"keras_tail"` | `"grouped_holdout"` | The oversampler appends duplicates at the end, grouped by class, and Keras's `validation_split` takes the last 20%. Validation is therefore copies of minority-class training rows, and early stopping (which restores the best weights) is driven by them. `"grouped_holdout"` holds out 20% of trajectories before oversampling. |
| `qc.evaluate_on_frames` | `False` | `True` | The spline is evaluated on a 10,000-point grid and indexed by frame number, so frames 136–190 are compared with the smoothed count at frames ~2.6–3.7. The score is then roughly a coefficient of variation of late counts. If you switch this on, recalibrate `qc.std_cut`: the scores drop about tenfold. |
| `features.include_last_window` | `False` | `True` | `range(traj_len - seg_length)` drops the last window of every trajectory, and trajectories exactly `trajl` long yield nothing. |
| `features.label_frame` | `"start_plus_trajl"` | `"snippet_end"` | The label is read at `frame(first cell) + trajl`, one frame after the snippet's last cell. With `trajl=1` that is the next frame's reporter value. |
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
                                                # diagnostics, simulator, CPU benchmark smoke test
python tests/synthetic.py /tmp/fx               # small fixture used by the parity check
python tests/parity/notebook_reference.py vaeFeatCCinfoABwells-NNclfRFreg.ipynb /tmp/fx /tmp/ref.npz
python tests/parity/package_run.py /tmp/fx /tmp/pkg.npz
python tests/parity/compare.py /tmp/ref.npz /tmp/pkg.npz
python tests/parity/benchmark_arm_parity.py /tmp/fx /tmp/ref.npz /tmp/armcheck   # benchmark arm = notebook
```

`notebook_reference.py` executes the notebook's own code cells. It changes only
the paths, the epoch count, the seed and the forest size, and asserts that each
of those edits matches exactly once. On the synthetic data every array matches
bit for bit: QC scores, per-FOV features and labels, trimmed and scaled sets,
oversampled set, network probabilities, training history, metrics, confusion
matrix and RFECV support. The one exception is the forest's predictions, which
agree to ~1e-15. `RandomForestRegressor.predict` with `n_jobs>1` sums tree
outputs in thread order, so repeated calls on one fitted model differ by that
much too. `benchmark_arm_parity.py` runs the benchmark's `published_insample`
arm on the same fixture and checks it against the notebook the same way, so
the benchmark's "published" row is the notebook's model, not a re-implementation
that might differ.
