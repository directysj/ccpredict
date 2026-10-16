# Fixing the task and the architecture: what it bought on the simulator

Every number below is cross-replicate (train on the A wells, test on each B
well separately), Pearson R on the raw N/C ratio, nothing fitted on the rows it
is scored on. Simulated data (`python -m ccpredict.simulate sim/ --scale 0.25
--seed 0`), 28,529 trimmed training rows, 40,720 test rows, CPU.

The simulator reproduces the paper's three honest baselines (R 0.510 / 0.530 /
0.516 here against 0.510 / 0.531 / 0.539 published), so its difficulty is about
right. It cannot tell you whether a change helps on real cells. What it can do
is price choices that are wrong on any data, and the last section bounds how
much there is left to win.

## The short version

| | R (mean over the 3 test FOVs) |
|---|---|
| the published pipeline, honestly evaluated (`published_crossfit`) | 0.522 |
| the published pipeline as published (forest fitted on the test FOV) | 0.894 |
| best configuration here: `nn_mlp_hist` + label-frame fix | **0.578 ± 0.003** |
| ceiling of this simulator (best model, noise-free embeddings) | 0.589 |

The gain over the published model comes almost entirely from one line of data
preparation, not from the network: the label is read one frame past the end of
the snippet it belongs to. Fixing that is worth about +0.05 R to every arm.
Of the model changes, only the histogram head is worth more than the seed noise.

## Fix 1: regress the ratio instead of binning it

Five seeds per arm, mean ± sd over seeds (`tests/seed_spread.py`). Notebook
labels, so these are comparable with `published_crossfit` = 0.522:

| arm | R B2_1 | R B2_3 | R B2_4 | mean R | s/run |
|---|---|---|---|---|---|
| `nn_mlp_hist` | 0.558 ± 0.008 | 0.517 ± 0.009 | 0.504 ± 0.007 | **0.526 ± 0.008** | 21 |
| `nn_mlp_log` | 0.548 ± 0.004 | 0.501 ± 0.006 | 0.494 ± 0.005 | 0.514 ± 0.005 | 39 |
| `nn_mlp_huber` | 0.542 ± 0.003 | 0.494 ± 0.006 | 0.487 ± 0.007 | 0.508 ± 0.003 | 34 |
| `nn_mlp` | 0.534 ± 0.006 | 0.481 ± 0.014 | 0.477 ± 0.011 | 0.497 ± 0.009 | 180¹ |
| `nn_mlp_weighted` | 0.512 ± 0.004 | 0.464 ± 0.005 | 0.457 ± 0.006 | 0.478 ± 0.001 | 24 |

¹ competing with another job for both cores; the others are the real cost.

- **Cross-entropy over 32 equal-count bins, predicting the expectation, is the
  best of the five** (+0.029 over plain squared loss, against a seed spread of
  0.008). Equal-count bins remove the imbalance by construction and the
  expectation keeps the output continuous, so this keeps what the classifier
  stage was good at and drops what it was bad at.
- **Fitting log(N/C) helps**, consistent with a positive, right-skewed target.
  Huber helps less, so the upper tail is signal, not outliers.
- **Inverse-density weighting hurts** (−0.019 against plain squared loss). The
  notebook's minority-class oversampling is the same idea, and it is the same
  mistake: it buys balance by making the common cells fit worse.
- All five replace both published stages with one network, and all five run in
  20–40 s against 819 s for `published_crossfit`.

## Fix 2: the architecture

One seed, notebook labels:

| arm | mean R | s/run | what it changes |
|---|---|---|---|
| `nn_mlp` | 0.504 | 32 | MLP on the features |
| `nn_conv` | 0.457 | 576 | conv along the latent index, BatchNorm before the activation |
| `nn_conv_notebook_bn` | 0.455 | 541 | the same with the notebook's BatchNorm order and conv bias |
| `nn_mlp_heavy_reg` | **−0.001** | 21 | MLP with the notebook's l2 0.01 / l1 0.02 |

- **The convolution costs 0.05 R and 18× the runtime.** Convolving a (3,)
  kernel along the VAE latent index assumes dimension 7 neighbours dimension 8.
  It does not, and the kernel pays for the assumption. This is the one
  architecture change in the published model that is wrong on any dataset.
- **BatchNorm order makes no difference to accuracy** (0.457 vs 0.455). Putting
  it before the activation only removes dead parameters: afterwards the conv
  bias has an exactly zero gradient, which is why `conv_bias=False` goes with
  it. Worth doing, worth nothing.
- **The notebook's penalties alone destroy the model.** With l2 0.01 / l1 0.02
  the regression net collapses to a constant (R ≈ 0, calibration slope −1.3e6):
  at initialisation the penalty is ~13 against a loss of ~1, so the optimiser
  spends its budget shrinking weights. The published classifier survives this
  only because predicting the 97% majority class keeps its cross-entropy
  respectable.

## The defect that mattered more than any model change

`features.label_frame` reads the label at `frame(start) + trajl`, one frame
(15 min) past the last cell of the snippet. `"snippet_end"` reads the frame of
the cell the morphology came from. Every arm, same rows otherwise:

| arm | notebook label | label at the snippet's own frame | change |
|---|---|---|---|
| `lasso_vae` | 0.530 | 0.561 | +0.031 |
| `rf_vae` | 0.516 | 0.549 | +0.033 |
| `nn_mlp` | 0.497 ± 0.009 | 0.551 ± 0.005 | +0.054 |
| `nn_mlp_hist` | 0.526 ± 0.008 | **0.578 ± 0.003** | +0.052 |

- Every arm gains, so this is the task being mis-specified rather than a model
  being improved: the published pipeline predicts the reporter one frame into
  the future from the current morphology and reports the result as if it were
  the reporter now.
- **The two defects interact.** `include_last_window=True` on its own changes
  nothing at all (identical row counts and R to three decimals): the last
  window's label would be read past the end of the trajectory, so that row is
  dropped as unlabelled. Fixing the label frame recovers those rows too
  (93,270 against 91,026).
- Histogram trimming of the training labels (`trim_training`) is worth
  −0.004 ± 0.004, i.e. nothing.

## How much is left? (`tests/ceiling.py`)

The simulator's generative state is known, so the attainable R can be measured
instead of guessed. Same split, forest and MLP, per-cell rows:

| input | dims | forest | MLP |
|---|---|---|---|
| the true ratio vs the measured label (no model) | 1 | — | 0.972 |
| generative state (cycle position, mitosis, pause, size) | 4 | 0.959 | 0.955 |
| embeddings before feature noise | 256 | 0.583 | 0.589 |
| embeddings as the pipeline sees them | 256 | 0.544 | 0.561 |

- **0.578 is at the ceiling.** The best configuration here sits 0.011 below the
  best any model managed on noise-free embeddings. There is no modelling change
  left that this simulator could reward, so "tweak until the synthetic number
  goes up" stops being a useful loop at this point.
- The information is lost in the encoding, not in the label: a model given the
  four generative factors reaches 0.959, and label noise alone caps R at 0.972.
  In this simulator that encoding is a random nonlinear map, which is a
  stand-in for O2VAE, not a model of it. **Whether the real embeddings keep more
  of the cycle than the simulated ones is the one question this cannot answer**,
  and it is the question that decides whether anything here is worth pursuing.

## What to check on the real data

1. `published_insample` against `published_crossfit`: is 0.755 an in-sample number?
2. `label_frame="snippet_end"` against the default, on any arm: does the
   off-by-one cost ~0.05 R there too? This is the cheapest thing to try and the
   most likely to transfer.
3. `nn_mlp_hist` against `published_crossfit` and `lasso_vae`: does one network
   match or beat two stages at 1/40th the cost?
4. `nn_conv` against `nn_mlp`: does the convolution hurt on the real embeddings
   as it does here?

## Commands

```bash
python -m ccpredict.simulate sim/ --scale 0.25 --seed 0
python -m ccpredict.benchmark --write-config bench.json     # edit paths to sim/
python -m ccpredict.benchmark --config bench.json --output-dir runs/tweaks \
    --arms constant,lasso_classical,lasso_vae,rf_vae,rf_vae_insample,nn_mlp,nn_mlp_log,nn_mlp_huber,nn_mlp_hist,nn_mlp_weighted,nn_mlp_heavy_reg,nn_conv,nn_conv_notebook_bn
python tests/seed_spread.py bench.json runs/spread --arms nn_mlp,nn_mlp_hist --seeds 5
python tests/ceiling.py sim/simulation.json
```

For the label-frame comparison, set `"features": {"label_frame": "snippet_end"}`
in a copy of `bench.json` and run the same arms into a second output directory.
