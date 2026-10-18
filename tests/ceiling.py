"""How well could anything predict N/C on the simulator? (An upper bound for the benchmark.)

    python tests/ceiling.py SIMULATION.json [--trees 100]

The benchmark says what each model scores. It cannot say whether a model that
scores less than another is leaving signal on the table or has already taken
all there is. On simulated data that is answerable, because the generative
state is known. This fits the same forest as the benchmark's rf_vae arm,
cross-replicate (EGF1 -> EGF2), on four different inputs:

    truth factors      the generative state (cycle position, mitosis, pause,
                       size): the best any feature extractor could recover
    clean embeddings   the 256-d encoding before feature noise is added: the
                       best any model could do from perfect images
    noisy embeddings   what the pipeline actually sees (the benchmark's input)
    nc_true            no model at all: the correlation between the true ratio
                       and the measured label, i.e. the reporter-noise cap

Read the gaps: noisy -> clean is what denoising could buy, clean -> truth what a
better encoder could buy, and truth -> nc_true what nothing can buy.

**These are not upper bounds, and the "noisy embeddings" row least of all.**
Every row except nc_true is "the better of a random forest and the default-MLP
DirectRegressor, fitted on *this* input at trajl=1". That is a bound on two
model classes, not on what is achievable: both read absolute feature levels,
and the MLP carries BatchNorm statistics frozen on EGF1 into an EGF2 test set.
A model outside those classes can exceed any row here without contradicting the
simulator's information content -- a transformer over the snippet's snapshots
does exactly that (results/regression_trajl8.md). Treat the numbers as "what
these two models extract", and read a *gap* as suggestive rather than as signal
that provably cannot be reached. Only nc_true is a real cap: it involves no
model, just the label against the ratio it is a noisy copy of.

In particular, arms sitting at the "noisy embeddings" number does NOT show the
simulator is exhausted. It shows those arms do no better than these two -- which
may mean they share a weakness rather than that there is nothing left.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FACTOR_KEYS = ("a", "phi", "mitotic", "paused")


def _noisy_embeddings(sims, cfg):
    """The embeddings the dataset on disk holds, by replaying write_dataset's RNG stream."""
    from ccpredict.simulate import add_feature_noise

    rng = np.random.default_rng(cfg.seed + 1)
    out = {}
    for fov, s in sims.items():
        out[fov] = add_feature_noise(s, cfg, rng)[0]
        rng.random(s.cell_frame.size)       # write_dataset draws this for the reporter file
    return out


def _groups(sim):
    """Trajectory id per labelled row, for trajectory-held-out early stopping."""
    traj_of = np.full(sim.cell_frame.size, -1)
    for k, cells in enumerate(sim.trajectories):
        traj_of[cells] = k
    return traj_of[~np.isnan(sim.nc_obs)]


def _rows(sim, noisy):
    """Per-cell rows with a measured label, as the pipeline's trajl=1 snippets are."""
    ok = ~np.isnan(sim.nc_obs)
    truth = np.column_stack([sim.truth[k][ok] for k in FACTOR_KEYS])
    return {"truth factors": truth,
            "clean embeddings": sim.embeddings_clean[ok],
            "noisy embeddings": noisy[ok],
            "nc_true": sim.truth["nc_true"][ok][:, None]}, sim.nc_obs[ok]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("simulation", help="simulation.json written next to a simulated dataset")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=-1, help="forest threads (1 to leave cores free)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the MLP (the forest has its own)")
    args = ap.parse_args(argv)

    from ccpredict.config import ForestConfig, _from_dict
    from ccpredict.simulate import ReplicateParams, SimConfig, simulate

    with open(args.simulation) as fh:
        meta = json.load(fh)
    meta.pop("summary", None)              # written by write_dataset, not a config field
    cfg = _from_dict(SimConfig, meta)
    # _from_dict does not recurse into a dict of dataclasses.
    cfg.replicates = {k: (v if not isinstance(v, dict) else _from_dict(ReplicateParams, v))
                      for k, v in cfg.replicates.items()}
    print(f"simulating (scale {cfg.scale}, seed {cfg.seed}) ...", flush=True)
    sims = simulate(cfg)

    train = [f for f, s in sims.items() if s.replicate == "EGF1"]
    test = [f for f, s in sims.items() if s.replicate == "EGF2"]
    noisy = _noisy_embeddings(sims, cfg)
    blocks = {f: _rows(s, noisy[f]) for f, s in sims.items()}
    groups = {f: _groups(s) for f, s in sims.items()}
    y_train = np.concatenate([blocks[f][1] for f in train])

    from sklearn.ensemble import RandomForestRegressor

    from ccpredict.config import NetConfig
    from ccpredict.models import DirectRegressor
    from ccpredict.preprocessing import trajectory_groups
    reg = ForestConfig()
    # Trajectory ids restart in every FOV, so make them unique the same way the
    # package does (a fixed 1000-per-FOV offset collides past 1000 trajectories).
    g_train = np.concatenate([trajectory_groups(np.full(groups[f].size, i), groups[f])
                              for i, f in enumerate(train)])

    def fit_forest(X, y):
        model = RandomForestRegressor(
            n_estimators=args.trees, max_depth=reg.max_depth,
            min_samples_split=reg.min_samples_split, min_samples_leaf=reg.min_samples_leaf,
            max_features=reg.max_features, n_jobs=args.n_jobs, random_state=reg.random_state)
        model.fit(X, y)
        return model.predict

    def fit_mlp(X, y):
        # The nn_mlp arm, on MinMax-scaled inputs as the arms use.
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler().fit(X)
        model = DirectRegressor(NetConfig(verbose=0, seed=args.seed)).fit(
            scaler.transform(X), y, g_train)
        return lambda Z: model.predict(scaler.transform(Z))

    print(f"\ntrain {train} ({y_train.size} rows) -> test {test}")
    print(f"{'input':<18}{'model':<8}{'dims':>6}  " + "  ".join(f"R {f:<8}" for f in test) + "  mean R")
    rows = {}
    for name in ("nc_true", "truth factors", "clean embeddings", "noisy embeddings"):
        X_train = np.concatenate([blocks[f][0][name] for f in train])
        fitters = {"forest": fit_forest, "mlp": fit_mlp}
        if name == "nc_true":
            # No model: the measured label against the true ratio it is a noisy copy of.
            fitters = {"none": lambda X, y: (lambda Z: Z[:, 0])}
        for model_name, fitter in fitters.items():
            predict = fitter(X_train, y_train)
            rs = [float(np.corrcoef(predict(blocks[f][0][name]), blocks[f][1])[0, 1]) for f in test]
            rows[(name, model_name)] = rs
            print(f"{name:<18}{model_name:<8}{X_train.shape[1]:>6}  "
                  + "  ".join(f"{r:<10.3f}" for r in rs) + f"  {np.mean(rs):.3f}", flush=True)

    best = {k: max(np.mean(rows[(k, m)]) for m in ("forest", "mlp"))
            for k in ("noisy embeddings", "clean embeddings", "truth factors")}
    cap = np.mean(rows[("nc_true", "none")])
    print(f"\nbest of the two models: noisy {best['noisy embeddings']:.3f} -> "
          f"clean {best['clean embeddings']:.3f} -> truth factors {best['truth factors']:.3f} "
          f"-> label noise cap {cap:.3f}")
    print(f"headroom from here: denoising the features "
          f"{best['clean embeddings'] - best['noisy embeddings']:+.3f}, recovering the generative "
          f"state {best['truth factors'] - best['clean embeddings']:+.3f}, "
          f"unattainable {cap - best['truth factors']:+.3f}")


if __name__ == "__main__":
    main()
