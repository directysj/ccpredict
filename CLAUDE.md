# ccpredict

Package version of `vaeFeatCCinfoABwells-NNclfRFreg.ipynb`: predicts the cell-cycle
reporter readout (nucleus/cytoplasm ratio, N/C) of single cells from O2VAE morphology
embeddings along their tracked trajectories. Preprint: doi 10.1101/2025.01.23.634593.
MCF10A + EGF; A wells = EGF1, B wells = EGF2, biological replicates.

Two published stages: a CNN classifying N/C into 3 bins, then an RFECV random forest on
[features | class probabilities]. `ccpredict/` reproduces that exactly; the benchmark
compares it with baselines, with honest protocols, and with one-stage alternatives.

## Ground rules

- **Never commit or push unless asked.** Leave changes in the working tree for review.
- **Defaults reproduce the notebook, defects included.** Every defect that changes numbers
  gets an off-by-default switch and a `NOTEBOOK DEFECT` comment. Never silently fix one.
- **No access to the real data.** Everything is validated on `ccpredict/simulate.py`. Say
  "on the simulator" rather than implying a result transfers to real cells.
- **Don't claim a gain smaller than the seed spread.** Run `tests/seed_spread.py` before
  ranking two arms. The ~0.01 R figure quoted below was measured with `nn_mlp` at
  `trajl=1`; **at `trajl=8` the spread is unmeasured** and the transformer arms look far
  more seed-sensitive than that. A single fit per arm ranks nothing.
- **Bootstrap CIs in the report resample rows, not training runs.** Two arms with
  non-overlapping intervals can still be indistinguishable once refitted. The interval
  answers "how much of this is which test cells", never "how much is the seed".
- **Nothing is scored on rows it was fitted on**, except the arms whose names say
  `_insample` (they exist to measure that effect).
- Device default is `"cpu"` (this Mac). Switch to `"auto"` only at handover to the
  Zuckerman-lab HPC.
- Blunt mechanistic critique over diplomatic summary.

## Layout

| path | what |
|---|---|
| `ccpredict/data.py`, `preprocessing.py` | load tracked FOVs + VAE + reporter, QC, snippets, labels, trimming |
| `ccpredict/cnn.py` | networks and the training loop (PyTorch), Keras-equivalent defaults |
| `ccpredict/models.py` | `StageOneClassifier`, `StageTwoRegressor`, `DirectRegressor` |
| `ccpredict/arms.py` | benchmark arms + registry |
| `ccpredict/benchmark.py` | splits, caching, metrics, `report.md` |
| `ccpredict/pipeline.py` | the notebook end to end |
| `ccpredict/simulate.py` | calibrated simulator (writes the real file layout) |
| `tests/parity/` | notebook parity and Keras-vs-PyTorch checks |
| `tests/ceiling.py`, `tests/seed_spread.py` | how much signal is left; is a gap bigger than the seed |

`README.md` explains the switches and the arms, `RUNNING.md` is the handover guide for
whoever has the real data, `results/simulator_tweaks.md` has the measurements below.

## Environment

Conda env `torch_290` (torch 2.9, Python 3.12) on this Mac. `pip install -e .` then
`pytest tests/`: 64 passed with TensorFlow present, 57 passed + 1 skipped without it
(the Keras parity file skips itself). TensorFlow is only needed for `tests/parity`;
install it with `pip install -e .[parity]`.

## State

Branch `ccp_torch_reg`. Stage 1 was ported from Keras to PyTorch, the one-stage `nn_*`
arms were added, and on this branch the sequence encoders for `trajl > 1` landed:
`snapshot_shared`, `temporal_conv` and `transformer` (self-attention among the snapshots
of a snippet), each selectable via `net.encoder` / `classifier.encoder` with `pool`
deciding the read-out. Changes are uncommitted on purpose.

**Branch defaults differ from `ccp_torch`**: `features.label_frame = "snippet_end"` and
`features.include_last_window = True`, because the ground truth N/C belongs to the *last
snapshot of the snippet*. `tests/parity/*` pin the notebook's old values explicitly so
parity still means parity.

## What has been measured (simulator, EGF1 -> EGF2, Pearson R)

The simulator reproduces the paper's three honest baselines (0.510 / 0.530 / 0.516 here
against 0.510 / 0.531 / 0.539 published), so its difficulty is about right.

- `published_insample` 0.894 vs `published_crossfit` 0.522. The paper's 0.755 sits
  between the two: it is an in-sample number, because the notebook fits the stage-2
  forest on the test FOV and scores the same rows.
- `rf_vae_insample` 0.911 vs `rf_vae` 0.516: in-sample fitting alone explains the gap.
- Stage 1 is worse than guessing: accuracy 0.82-0.86 against a majority class of 0.975+,
  balanced accuracy ~0.43. The middle bin holds ~97% of rows.
- The notebook's validation slice is 100% oversampled duplicates of class 2. Fixing it
  (`published_crossfit_groupedval`) changes nothing: 0.521 vs 0.522.
- One-stage nets (5 seeds): `nn_mlp_hist` 0.526 +- 0.008 > `nn_mlp_log` 0.514 >
  `nn_mlp_huber` 0.508 > `nn_mlp` 0.497 > `nn_mlp_weighted` 0.478. Inverse-density
  weighting hurts, which is the regression analogue of the notebook's oversampling.
- `nn_conv` 0.457 at 18x the runtime: convolving along the VAE latent index assumes an
  ordering that does not exist. BatchNorm order is cosmetic (0.457 vs 0.455) but before
  the activation the conv bias is provably dead, hence `conv_bias=False`.
- `nn_mlp_heavy_reg` R ~ 0: the notebook's l2 0.01 / l1 0.02 collapse a regression net to
  a constant (penalty ~13 at init against a loss of ~1).
- **The largest gain was not a model change.** `features.label_frame` reads the label one
  frame past the snippet; `"snippet_end"` adds 0.03-0.05 R to every arm. Best
  configuration so far: `nn_mlp_hist` + that fix, 0.578 +- 0.003.
- `tests/ceiling.py`: from noise-free embeddings the best model manages 0.589, from the
  generative state 0.959, and label noise caps R at 0.972. **These are not upper bounds.**
  Every row but `nc_true` is the better of a forest and the default-MLP `DirectRegressor`
  at `trajl=1`, so it bounds two model classes, not what is achievable: at `trajl=8` a
  transformer over the snippet's snapshots reaches 0.609 on one fit, past the 0.561
  "noisy embeddings" row. Read a gap as suggestive, never as proof nothing is left.
- At `trajl=8` with the corrected label, attention among a snippet's snapshots leads on a
  single fit: `nn_transformer_hist` 0.609 vs `lasso_vae` 0.552, dropping to 0.510 with the
  attention ablated and 0.560 for a cheap attention pool. One seed per arm, so unranked;
  see `results/regression_trajl8.md`, including why the earlier "the simulator cannot
  reward a temporal architecture" argument was wrong (snapshots are redundant given the
  latent state, not given noisy observations of it).

## Verification that must keep passing

- `tests/parity/notebook_reference.py` + `package_run.py --stage1-from` + `compare.py`:
  every array matches the notebook bit for bit except the CNN itself (forest predictions
  agree to ~1e-15; `RandomForestRegressor.predict` with `n_jobs>1` sums in thread order).
- `tests/parity/test_torch_port.py` (needs TensorFlow): with Keras weights copied in, the
  forward pass, penalty, gradients, 10 Adam steps and a 12-epoch fit all match Keras, and
  the early-stopping rule makes Keras' decisions. Rerun it after touching `cnn.py` or
  stage 1 of `models.py`.
- `tests/parity/torch_vs_keras.py`: 10 seeds per framework, statistical equivalence
  (`tests/parity/results/torch_vs_keras_sim.md`).

## Open

- Fix 3 from the earlier list is not done: use the trajectory. `trajl=1` throws away the
  time axis, and N/C is smooth in time, so a sequence model over the snippet is the
  remaining idea with a mechanism behind it.
- Handover: `RUNNING.md` tells whoever has the real data what to run. The one addition
  worth insisting on is a second run with `label_frame: "snippet_end"`.
- Before handover, switch `classifier.device` and `net.device` to `"auto"`.
- `runs/` holds local benchmark outputs (~440 MB, gitignored); `runs/*/cache` is the
  prepared features and can be deleted at any time.
