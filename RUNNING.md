# Running the benchmark on the real data

This is for whoever has access to the LI204601 files. It compares the model
published in the preprint (VAE features + CNN class probabilities + RFECV
random forest) with baselines and with honest versions of the same model, all
on the same rows. Your run produces a results folder; send that back.

No code changes are needed, only a config file with the data paths.

## 1. Environment

The environment that ran `vaeFeatCCinfoABwells-NNclfRFreg.ipynb` works:
Python ≥ 3.9, numpy, pandas, scipy, scikit-learn, matplotlib, joblib, csaps,
imbalanced-learn, TensorFlow with Keras 3. No new libraries.

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

Also check `pipeline.classifier.device`: `"cpu"`, `"gpu"` (fails if TensorFlow
sees no GPU) or `"auto"` (GPU if available). Only the CNN uses it.

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
  `--arms published_crossfit_groupedval`.

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

The key comparisons: `published_insample` vs `published_crossfit` (is the
published R an in-sample number?), `rf_vae` vs `rf_vae_insample` (how much does
in-sample fitting add by itself?), and `published_crossfit` vs `rf_vae` (do the
CNN probabilities help on held-out data?). `diagnostics.json` also reports
whether trajectories share cells, how different the A and B label
distributions are, what the notebook's validation slice consisted of, and how
optimistic a random row split is compared with a held-out FOV.
