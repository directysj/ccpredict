# PyTorch vs Keras stage 1

Train FOVs ['A2_1', 'A2_2', 'A2_3'] (8010 rows), test FOVs ['B2_1', 'B2_3', 'B2_4'] (40720 rows); validation=keras_tail, epochs<=50, patience=15, batch=128.
Seeds: keras 10, torch 10.

Difference = torch - keras. 'd/sd' is the difference over the pooled within-framework sd: |d/sd| well below 1 means the framework changes the result less than the seed does.

| metric | keras mean ± sd | torch mean ± sd | difference [95% CI] | d/sd | Mann-Whitney p |
|---|---|---|---|---|---|
| accuracy | 0.9236 ± 0.0438 | 0.8714 ± 0.0574 | -0.0522 [-0.1004, -0.0039] | -1.02 | 0.031 |
| balanced_accuracy | 0.3795 ± 0.0258 | 0.4006 ± 0.0162 | +0.0212 [+0.0007, +0.0417] | +0.98 | 0.026 |
| f1_macro | 0.3431 ± 0.0065 | 0.3352 ± 0.0122 | -0.0079 [-0.0173, +0.0015] | -0.81 | 0.038 |
| log_loss | 0.2845 ± 0.0726 | 0.3719 ± 0.1049 | +0.0874 [+0.0019, +0.1729] | +0.97 | 0.017 |
| spearman_score_vs_nc | 0.2352 ± 0.0185 | 0.2412 ± 0.0141 | +0.0060 [-0.0095, +0.0215] | +0.37 | 0.910 |
| mean_p0 | 0.0501 ± 0.0354 | 0.0905 ± 0.0436 | +0.0404 [+0.0029, +0.0778] | +1.02 | 0.038 |
| mean_p1 | 0.9224 ± 0.0566 | 0.8591 ± 0.0685 | -0.0633 [-0.1224, -0.0041] | -1.01 | 0.031 |
| mean_p2 | 0.0275 ± 0.0240 | 0.0503 ± 0.0272 | +0.0229 [-0.0012, +0.0470] | +0.89 | 0.045 |
| epochs | 47.2000 ± 3.3267 | 44.9000 ± 6.2441 | -2.3000 [-7.1075, +2.5075] | -0.46 | 0.544 |
| best_epoch | 38.4000 ± 9.7662 | 33.3000 ± 9.7303 | -5.1000 [-14.2591, +4.0591] | -0.52 | 0.172 |
| final_val_loss | 0.3539 ± 0.0663 | 0.3364 ± 0.0168 | -0.0175 [-0.0655, +0.0306] | -0.36 | 0.791 |
| best_val_loss | 0.2963 ± 0.0104 | 0.3026 ± 0.0096 | +0.0064 [-0.0030, +0.0158] | +0.64 | 0.186 |
| seconds | 360.1649 ± 36.1187 | 428.1630 ± 64.6899 | +67.9982 [+17.7854, +118.2110] | +1.30 | 0.026 |

## Do the two frameworks learn the same function?

Pearson r of the seed-averaged score p(class 2) - p(class 0) over the test rows, each average over half of the seeds:

- keras half vs keras half: 0.8817
- torch half vs torch half: 0.8993
- keras half vs torch half: 0.8740

If the port were wrong in a way that matters, the last number would sit clearly below the first two.

## Run

Simulator data at scale 0.25 (`python -m ccpredict.simulate`), benchmark config with the notebook classifier settings, run on 2 CPU cores:

```bash
python tests/parity/torch_vs_keras.py bench.json out --seeds 10 --max-train-rows 8000
```

Per-run rows: `torch_vs_keras_sim_runs.csv` (same folder as this file).

## Restored epoch

Pooled over both frameworks, the epoch early stopping restored correlates with mean p(class 1) on the test replicate (Spearman +0.66) and with log-loss (-0.61). Regressing each metric on framework + restored epoch: framework term for mean p(class 1) -0.042 (p = 0.083), log-loss +0.063 (p = 0.11), accuracy -0.036 (p = 0.080), balanced accuracy +0.014 (p = 0.095).
