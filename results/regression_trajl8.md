# One-stage regression at trajl=8, label at the last snapshot

Cross-replicate (train on A wells, test on B2_4), Pearson R on raw N/C, nothing
fitted on the rows it is scored on. Simulated data, `trajl=8` (8 x 15 min = 2 h),
`label_frame="snippet_end"`, `include_last_window=True`. Two training FOVs
(A2_1 + A2_2) rather than three because `trajl=8` at 2,048 features exceeded the
memory of the machine this ran on. 33,979 snippets survive labelling, of which
training trimming keeps 24,809: the nets fit on 20,707 and hold out 4,102 whole
trajectories for early stopping. 9,461 test rows.

Every arm here replaces both published stages with one network: no class bins,
no forest.

> **Read the whole table as one fit per arm.** Every number below is a single
> seed, and the seed spread at `trajl=8` has not been measured (see "What is not
> established"). The `+-0.01` figure quoted elsewhere in this repo was measured
> with `nn_mlp` at `trajl=1` and does not transfer. Differences smaller than a
> few hundredths should be treated as unranked.

| arm | encoder | pool | head | R | 95% CI | MSE | per-traj R | restored epoch | s |
|---|---|---|---|---|---|---|---|---|---|
| `nn_transformer_hist` | transformer | last | 32 bins | **0.609** | [0.561, 0.658] | 0.134 | 0.736 | – | 198 |
| `nn_transformer` | transformer | last | squared | 0.603 | [0.558, 0.655] | 0.135 | 0.736 | – | 228 |
| `nn_snapshot_attn_hist` | shared per-snapshot | attention | 32 bins | 0.560 | [0.499, 0.607] | 0.146 | 0.719 | – | 28 |
| `lasso_vae` | – | – | – | 0.552 | [0.503, 0.593] | 0.139 | 0.651 | – | 248 |
| `nn_tconv_hist` | temporal conv | last | 32 bins | 0.545 | [0.483, 0.590] | 0.142 | 0.635 | 2 of 17 | 62 |
| `nn_snapshot_hist` | shared per-snapshot | last | 32 bins | 0.545 | [0.495, 0.586] | 0.144 | 0.655 | 7 of 22 | 30 |
| `rf_vae` | – | – | – | 0.537 | [0.481, 0.584] | 0.153 | 0.664 | – | 48 |
| `nn_mlp` | flat MLP | – | squared | 0.528 | [0.472, 0.584] | 0.146 | 0.688 | 77 of 80 | 108 |
| `nn_snapshot` | shared per-snapshot | last | squared | 0.511 | – | 0.152 | 0.607 | 29 of 44 | 58 |
| `nn_transformer_noattn` | transformer, attention off | last | squared | 0.510 | [0.455, 0.555] | 0.156 | 0.596 | 42 of 57 | 326 |
| `nn_tconv` | temporal conv | last | squared | 0.508 | – | 0.158 | 0.598 | 10 of 25 | 100 |
| `nn_tconv_concat` | temporal conv | concat | squared | 0.488 | – | 0.153 | 0.600 | 42 of 57 | 223 |
| `constant` | – | – | – | – | – | 0.200 | – | – | 0 |

The transformer rows and `nn_snapshot_attn_hist` come from a later run on the
same split (`lasso_vae` and `nn_snapshot_hist` reproduced 0.552 and 0.545 there
to three decimals, which is what makes the two runs comparable).

## R and R-squared are not interchangeable here

The benchmark's `r2` is the coefficient of determination, `1 - SS_res/SS_tot`,
not the square of R. R is invariant to any offset or rescaling of the
predictions; `r2` additionally charges for being wrong in slope or intercept.

| arm | R | R^2 | reported `r2` | calibration loss |
|---|---|---|---|---|
| `lasso_vae` | 0.552 | 0.305 | 0.299 | 0.006 |
| `nn_snapshot_hist` | 0.545 | 0.297 | 0.276 | 0.021 |
| `nn_snapshot_attn_hist` | 0.560 | 0.314 | 0.265 | 0.049 |
| `nn_transformer` | 0.603 | 0.364 | 0.321 | 0.043 |
| `nn_transformer_hist` | 0.609 | 0.371 | 0.327 | 0.044 |
| `nn_transformer_noattn` | 0.509 | 0.260 | 0.215 | 0.045 |

Least squares lands slope ~1 on its own training set, so `lasso_vae` loses
almost nothing; every net loses 0.02-0.05. The mechanism is the replicate shift
appearing a second time: B2_4's N/C quantiles are [0.996, 1.464, 2.417] against
A2_2's [0.997, 1.426, 2.269], so the test replicate's upper tail is wider than
anything in training and predictions come out compressed. A recalibration
fitted on held-out *training* rows cannot correct a shift specific to the test
replicate, and fitting one on the test replicate is the in-sample error this
benchmark exists to measure. So:

- **R** is the fairer statistic for comparing models across replicates, being
  invariant to exactly the nuisance that differs.
- **`r2`** is what to quote for the error you would make on a fresh replicate,
  and it is the lower number. The paper's R = 0.755 squares to 0.570 against a
  reported 0.54, about 0.03 of the same calibration loss.

## What this does and does not establish

- **The histogram head, the pooling choice and the encoder ranking are all
  unreplicated.** The 32-bin head appears worth +0.035 over squared loss on both
  non-attention sequence encoders, `pool="last"` appears to beat `concat`
  (0.508 vs 0.488), and the temporal convolution appears to add nothing over a
  shared per-snapshot encoder (0.545 vs 0.545) at half the runtime. Each of
  those is one fit against one fit. The first was also seen at `trajl=1`
  (0.526 vs 0.497), which is weak corroboration; the others have none.
- **The transformer leads on this single fit, by about one interval's width.**
  `nn_transformer_hist` 0.609 against `lasso_vae` 0.552, and its interval
  [0.561, 0.658] clears the LASSO's point estimate. Stripping the attention
  while holding the rest of the block fixed (`transformer_attention="none"`)
  drops it to 0.510, and the cheap control that weights snapshots without a
  pairwise query-key term reaches only 0.560. Taken at face value that says the
  gain is the attention and not the block around it. **It is one seed per arm,
  so it is not yet a result** -- and note the bootstrap intervals resample rows,
  not training runs, so non-overlapping intervals say nothing about seed
  variance.
- **Cost.** At `trajl=8`, nf=256: the transformer is 100,673 parameters and
  ~57 ms per batch of 256 (fwd+bwd), against 33,089 / 5.7 ms for the shared
  encoder, 78,465 / 25 ms for the temporal conv, and 4,209,281 / 2,035 ms for
  the notebook's flat convolution.
- **Budget matters as much as architecture.** `nn_tconv_hist` restored epoch 2
  of 17 and `nn_mlp` epoch 77 of 80. Early stopping watches trajectories held
  out of the *training replicate*, which cannot see the EGF1 -> EGF2 shift; in
  the classification runs, training to convergence made every architecture
  worse on the test replicate while raising its accuracy.

## Why a temporal architecture can win here after all

An earlier version of this file argued that it could not: in `simulate.py` the
N/C ratio is a deterministic function of the cell's *current* CDK2 activity, the
label is the last snapshot's N/C, so the preceding seven snapshots are redundant
by construction.

That argument is wrong, and the error is worth keeping. The earlier snapshots
are redundant *given the noise-free latent state* -- which no model sees. What a
model sees is embeddings carrying AR(1) track-persistent noise (50%, rho=0.95)
on top of `vae_noise_sd` 1.2. Snapshot 7's *observation* is noisy and CDK2 traces
are smooth in time, so snapshots 0-6 genuinely constrain snapshot 7's true
state. Recovering a current state from a noisy trajectory is a filtering
problem, and there was signal in the time axis the whole time. "Redundant given
the state" is not "redundant given noisy observations of the state".

What is still unexplained is why attention beats a temporal convolution at that
job (0.603 vs 0.508 on one fit each), since a width-3 kernel over 8 steps can
also combine neighbours. No mechanism is claimed for that.

## What is not established

- **The seed spread at `trajl=8`.** Unmeasured. A sweep was attempted and had to
  be discarded: `tests/seed_spread.py` built its `SplitContext` without `trajl`,
  which defaulted to 1, so every fit in it ran with `steps=1` on `trajl=8`
  features -- the sequence encoders saw one 2,048-feature snapshot instead of
  eight 256-feature ones, and the transformer got a single token with nothing to
  attend to. It trained and reported plausible numbers. `SplitContext.trajl` now
  has no usable default and `__post_init__` refuses an unset one; a valid sweep
  needs ~15 fits at 200-900 s each and belongs on a machine with more than two
  cores.
- **Whether the ranking transfers to real cells.** The simulator's noise model is
  calibrated but invented, and the filtering argument above means the ranking now
  depends on that noise model rather than being ruled out by it. A temporal
  architecture's advantage here is an advantage against AR(1) noise with
  rho=0.95, which is not a measurement of real O2VAE embeddings.
- **The ceiling.** `tests/ceiling.py` reports 0.561 for "noisy embeddings", and
  `nn_transformer_hist` exceeds it. That is not a contradiction: every row of
  that tool except `nc_true` is the better of a random forest and the default-MLP
  `DirectRegressor` at `trajl=1`, i.e. a bound on two model classes, not on what
  is achievable. An earlier version of this file called 0.561 the point where
  "there is nothing left on this data for a better model to take". That was
  wrong, and it is the kind of claim that stops further work.
