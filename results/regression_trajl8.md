# One-stage regression at trajl=8, label at the last snapshot

Cross-replicate (train on A wells, test on B2_4), Pearson R on raw N/C, nothing
fitted on the rows it is scored on. Simulated data, `trajl=8` (8 x 15 min = 2 h),
`label_frame="snippet_end"`, `include_last_window=True`, 33,979 training rows
(A2_1 + A2_2) and 9,461 test rows. Two training FOVs rather than three because
`trajl=8` at 2,048 features exceeded the 6 GB of the machine this ran on.

Every arm here replaces both published stages with one network: no class bins,
no forest.

| arm | encoder | pool | head | R | 95% CI | MSE | per-traj R | restored epoch | s |
|---|---|---|---|---|---|---|---|---|---|
| `lasso_vae` | – | – | – | **0.552** | [0.503, 0.593] | 0.139 | 0.651 | – | 248 |
| `nn_tconv_hist` | temporal conv | last | 32 bins | 0.545 | [0.483, 0.590] | 0.142 | 0.635 | 2 of 17 | 62 |
| `nn_snapshot_hist` | shared per-snapshot | last | 32 bins | 0.545 | [0.495, 0.586] | 0.144 | 0.655 | 7 of 22 | 30 |
| `rf_vae` | – | – | – | 0.537 | [0.481, 0.584] | 0.153 | 0.664 | – | 48 |
| `nn_mlp` | flat MLP | – | squared | 0.528 | [0.472, 0.584] | 0.146 | 0.688 | 77 of 80 | 108 |
| `nn_snapshot` | shared per-snapshot | last | squared | 0.511 | – | 0.152 | 0.607 | 29 of 44 | 58 |
| `nn_tconv` | temporal conv | last | squared | 0.508 | – | 0.158 | 0.598 | 10 of 25 | 100 |
| `nn_tconv_concat` | temporal conv | concat | squared | 0.488 | – | 0.153 | 0.600 | 42 of 57 | 223 |
| `constant` | – | – | – | – | – | 0.200 | – | – | 0 |

## What this does and does not establish

- **No architecture beats the LASSO.** Every interval overlaps every other; the
  spread across all eight arms (0.064) is smaller than one interval's width
  (~0.10). `lasso_vae` at 0.552 also sits at the ceiling measured in
  `tests/ceiling.py` (0.561 for a model on the noisy embeddings with clean
  labels), so there is nothing left on this data for a better model to take.
- **The head matters more than the encoder.** The 32-bin histogram head is worth
  +0.035 over squared loss on both sequence encoders (0.545 vs 0.508 / 0.511),
  the same effect size measured at `trajl=1` (0.526 vs 0.497). Among the
  squared-loss arms the flat MLP is the best of them.
- **The convolution over time adds nothing over a shared per-snapshot encoder**
  (0.545 vs 0.545), and `nn_snapshot_hist` reaches it in half the time.
- **`pool="last"` beats `pool="concat"`** (0.508 vs 0.488 at 2.2x the runtime),
  which is what the label semantics predict: the ground truth belongs to the
  last snapshot, so reading out that step is better than flattening all eight
  and letting the head sort it out.
- **Budget matters as much as architecture.** `nn_tconv_hist` restored epoch 2
  of 17 and `nn_mlp` epoch 77 of 80. Early stopping watches trajectories held
  out of the *training replicate*, which cannot see the EGF1 -> EGF2 shift; in
  the classification runs, training to convergence made every architecture
  worse on the test replicate while raising its accuracy.

**Why the ranking cannot transfer.** In `simulate.py` the N/C ratio is a
deterministic function of the cell's *current* CDK2 activity. The ground truth
is the last snapshot's N/C, so it is fully determined by that one snapshot's
latent state, and the preceding seven snapshots are redundant by construction.
A temporal architecture cannot win here because there is no temporal signal to
find. What transfers is the cost (parameters, seconds) and the fact that the
code is correct; which architecture predicts real cell cycles is not answerable
on this simulator without first making N/C depend on the cell's history.
