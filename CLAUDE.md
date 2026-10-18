# ccpredict — branch `ccp_torch_reg`

Predicts the cell-cycle reporter readout (nucleus/cytoplasm ratio, N/C) of single cells
from O2VAE morphology embeddings along their tracked trajectories. Preprint:
doi 10.1101/2025.01.23.634593. MCF10A + EGF; A wells = EGF1, B wells = EGF2, biological
replicates.

**One stage only.** A single network regresses N/C directly from a trajectory snippet.
The published two-stage model (CNN classifying binned N/C, then an RFECV forest on the
class probabilities), the `published_*` and `clf_*` arms, `pipeline.py` and the whole
notebook-parity suite were removed from this branch and live on **`ccp_torch`**. Do not
re-add them here; check that branch out instead.

What remains beside the networks: the LASSO and forest baselines (`lasso_vae`,
`lasso_classical`, `rf_vae`, `rf_vae_insample`, `constant`), which are the reference a
one-stage model has to beat.

## Ground rules

- **Never commit or push unless asked.** Leave changes in the working tree for review.
- **No access to the real data.** Everything is validated on `ccpredict/simulate.py`. Say
  "on the simulator" rather than implying a result transfers to real cells.
- **Don't claim a gain smaller than the seed spread.** Run `tests/seed_spread.py` before
  ranking two arms. The ~0.01 R figure from earlier work was measured with `nn_mlp` at
  `trajl=1`; **at `trajl=8` the spread is unmeasured** and the transformer arms look far
  more seed-sensitive. A single fit per arm ranks nothing.
- **Bootstrap CIs in the report resample rows, not training runs.** Two arms with
  non-overlapping intervals can still be indistinguishable once refitted. The interval
  answers "how much of this is which test cells", never "how much is the seed".
- **R is not R².** The report's `r2` is the coefficient of determination and also charges
  for calibration error; R does not. The nets lose 0.02–0.05, the LASSO ~0.006.
- **Nothing is scored on rows it was fitted on**, except arms whose names say `_insample`.
- Device default is `"cpu"` (this Mac). Switch `net.device` to `"auto"` only at handover
  to the Zuckerman-lab HPC.
- Blunt mechanistic critique over diplomatic summary.

## Layout

| path | what |
|---|---|
| `ccpredict/data.py`, `preprocessing.py` | load tracked FOVs + VAE + reporter, QC, snippets, labels, trimming |
| `ccpredict/cnn.py` | the network and the training loop (PyTorch) |
| `ccpredict/models.py` | `DirectRegressor`, `build_net` |
| `ccpredict/arms.py` | benchmark arms + registry |
| `ccpredict/benchmark.py` | splits, caching, metrics, `report.md` |
| `ccpredict/simulate.py` | calibrated simulator (writes the real file layout) |
| `tests/test_equivalence.py` | data-prep rewrites vs verbatim notebook functions |
| `tests/ceiling.py`, `tests/seed_spread.py` | how much signal is left; is a gap bigger than the seed |

`README.md` explains the encoders, heads and arms; `RUNNING.md` is the handover guide.

## Branch defaults that differ from the notebook

`features.label_frame = "snippet_end"` and `features.include_last_window = True`, because
the ground truth N/C belongs to the **last snapshot of the snippet**. Worth 0.03–0.05 R
to every arm. Confirmed against the real `jcTrajectory_CP`: `get_cell_trajectory` ends in
`np.flip`, so trajectories run forward in time, and `get_traj_segments` iterates
`range(traj_len - seg_length + 1)` — it includes the last window.

## Environment

Conda env `torch_290` (torch 2.9, Python 3.12) on this Mac. `pip install -e .` then
`pytest tests/`: 69 passed, 2 skipped in the cloud container (the 2 skips are the MPS and
CUDA accelerator tests, which run on the Mac). TensorFlow is no longer needed anywhere on
this branch.

## What has been measured (simulator, EGF1 → EGF2, Pearson R)

At `trajl=1` the simulator reproduced the paper's three honest baselines (0.510 / 0.530 /
0.516 here against 0.510 / 0.531 / 0.539 published), so its difficulty is about right.

- One-stage nets at `trajl=1` (5 seeds): `nn_mlp_hist` 0.526 ± 0.008 > `nn_mlp_log` 0.514
  > `nn_mlp_huber` 0.508 > `nn_mlp` 0.497 > `nn_mlp_weighted` 0.478. Inverse-density
  weighting hurts.
- `nn_conv` 0.457 at 18× the runtime: convolving along the VAE latent index assumes an
  ordering that does not exist. BatchNorm order is cosmetic (0.457 vs 0.455) but before
  the activation the conv bias is provably dead, hence `conv_bias=False`.
- `nn_mlp_heavy_reg` R ≈ 0: the notebook's l2 0.01 / l1 0.02 collapse a regression net to
  a constant (penalty ~13 at init against a loss of ~1).
- **The largest gain was not a model change**: fixing `features.label_frame` added
  0.03–0.05 R to every arm. Best at `trajl=1`: `nn_mlp_hist` + that fix, 0.578 ± 0.003.
- At `trajl=8` with the corrected label, **one fit per arm**: `nn_transformer_hist` 0.609,
  `nn_transformer` 0.603, `nn_snapshot_attn_hist` 0.560, `lasso_vae` 0.552,
  `nn_snapshot_hist` / `nn_tconv_hist` 0.545, `nn_transformer_noattn` 0.510. Unranked
  until the seeds are run. See `results/regression_trajl8.md`.
- `tests/ceiling.py` reports 0.589 from noise-free embeddings, 0.959 from the generative
  state, 0.972 as the label-noise cap. **These are not upper bounds**: every row but
  `nc_true` is the better of a forest and the default-MLP regressor at `trajl=1`, so it
  bounds two model classes. The transformer already passed the 0.561 "noisy embeddings"
  row.
- The earlier claim that the simulator *cannot* reward a temporal architecture was wrong.
  Snapshots 0..L-2 are redundant given the noise-free latent state, not given noisy
  observations of it; recovering the current state from a noisy trajectory is a filtering
  problem and there is signal in the time axis. See `results/regression_trajl8.md`.

## Open

- **The seed sweep at `trajl=8`.** The one thing that decides whether the transformer is
  real. ~15 fits at 200–900 s each; needs more than two cores.
- Before handover: switch `net.device` to `"auto"`.
- `runs/` holds local benchmark outputs (gitignored); `runs/*/cache` is the prepared
  features and can be deleted at any time. `runs/sim/jcTrajectory_CP.py` must stay the
  simulator's stand-in — the real tracker module is reference-only at
  `tests/reference/jcTrajectory_CP_real.py` and importing it needs pyemma/btrack/mahotas.
