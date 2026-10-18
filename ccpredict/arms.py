"""Models compared by the benchmark ("arms"). Each arm is fitted once per split and scored on
every test FOV of that split.

    constant                      training-set mean (anything must beat this)
    lasso_classical               LASSO on handcrafted features       (paper: R 0.510)
    lasso_vae                     LASSO on VAE embeddings              (paper: R 0.531)
    rf_vae                        random forest on VAE, fit on training FOVs  (paper: R 0.539)
    rf_vae_insample               same forest, fit AND scored on the test FOV (diagnostic)
    published_insample            CNN probabilities + RFECV forest exactly as in the notebook:
                                  forest fit and scored on the test FOV (paper: R 0.755)
    published_crossfit            same CNN, forest fit on training FOVs with out-of-fold CNN
                                  probabilities, scored on the test FOVs
    published_crossfit_groupedval as published_crossfit, but the CNN's early stopping uses held-out
                                  trajectories instead of the oversampled tail

One-stage neural regressors (no class bins, no forest; see NetConfig). All are fit
on the training FOVs and scored on the test FOVs:

    nn_mlp                        MLP, squared loss on the N/C ratio
    nn_mlp_log                    same, fitting log(N/C)
    nn_mlp_huber                  same, Huber loss (the ratio has a long upper tail)
    nn_mlp_hist                   MLP, cross-entropy over 32 equal-count bins of N/C,
                                  predicting the expectation
    nn_mlp_weighted               nn_mlp with rows weighted by 1 / label density
    nn_conv                       the notebook's conv encoder, BatchNorm before the
                                  activation and no conv bias: does the convolution help?
    nn_conv_notebook_bn           as nn_conv but with the notebook's BatchNorm order and
                                  conv bias, to price that choice on its own
    nn_mlp_heavy_reg              nn_mlp with the notebook's l2 0.01 / l1 0.02 penalties,
                                  to price those on their own
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import ForestConfig, NetConfig

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Data blocks
# ----------------------------------------------------------------------------

@dataclass
class Block:
    """Rows of one or more FOVs: features, labels and trajectory bookkeeping."""

    X: np.ndarray
    y: np.ndarray
    fold: np.ndarray     # index of the FOV within the block's FOV list
    traj: np.ndarray     # trajectory index within that FOV
    fovs: tuple

    @property
    def groups(self) -> np.ndarray:
        from .preprocessing import trajectory_groups
        return trajectory_groups(self.fold, self.traj)

    def __len__(self) -> int:
        return self.y.size


@dataclass
class ArmResult:
    y_pred: np.ndarray
    proba: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)


@dataclass
class SplitContext:
    """What an arm sees: the training block, the test blocks and shared caches."""

    split_name: str
    train: Block                   # trimmed if the benchmark trims
    tests: Dict[str, Block]
    columns: Callable[[str], np.ndarray]
    forest_cfg: ForestConfig
    net_cfg: NetConfig = field(default_factory=NetConfig)
    # Snapshots per row (features.trajl). No usable default: 1 silently collapses
    # the time axis, so a sequence encoder becomes a plain net over the flattened
    # row and a transformer gets a single token -- it still trains and still
    # reports a number, just not for the model you asked for. 0 forces the caller
    # to say. (tests/seed_spread.py omitted it once; the results looked ordinary.)
    trajl: int = 0
    cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trajl < 1:
            raise ValueError(
                "SplitContext.trajl must be the snippet length (pass trajl=meta['trajl'] / "
                "features.trajl); it decides how many snapshots the sequence encoders see")

    def X_train(self, feature_set: str) -> np.ndarray:
        return self.train.X[:, self.columns(feature_set)]

    def X_test(self, fov: str, feature_set: str) -> np.ndarray:
        return self.tests[fov].X[:, self.columns(feature_set)]


class Arm:
    name: str = "arm"
    feature_sets: tuple = ("vae",)

    def run(self, ctx: SplitContext) -> Dict[str, ArmResult]:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

class ConstantArm(Arm):
    name = "constant"
    feature_sets = ()

    def run(self, ctx):
        mean = float(ctx.train.y.mean())
        return {f: ArmResult(np.full(len(b), mean), info={"value": mean}) for f, b in ctx.tests.items()}


def _lasso_alphas(X: np.ndarray, y: np.ndarray, n: int = 30) -> np.ndarray:
    """Explicit alpha grid, so the code runs on old and new scikit-learn alike."""
    alpha_max = np.max(np.abs(X.T @ (y - y.mean()))) / X.shape[0]
    return np.logspace(np.log10(alpha_max * 1e-4), np.log10(alpha_max), n)[::-1]


class LassoArm(Arm):
    """StandardScaler + LassoCV, alpha chosen by trajectory-grouped 5-fold CV."""

    def __init__(self, feature_set: str):
        self.feature_set = feature_set
        self.feature_sets = (feature_set,)
        self.name = f"lasso_{feature_set.replace('+', '_')}"

    def run(self, ctx):
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(ctx.X_train(self.feature_set))
        cv = list(GroupKFold(n_splits=5).split(X, ctx.train.y, ctx.train.groups))
        model = LassoCV(alphas=_lasso_alphas(X, ctx.train.y), cv=cv, n_jobs=-1, max_iter=5000)
        model.fit(X, ctx.train.y)
        info = {"alpha": float(model.alpha_), "n_nonzero": int(np.sum(model.coef_ != 0))}
        return {f: ArmResult(model.predict(scaler.transform(ctx.X_test(f, self.feature_set))), info=info)
                for f in ctx.tests}


def _forest(cfg: ForestConfig):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features, n_jobs=cfg.n_jobs, random_state=cfg.random_state)


class ForestArm(Arm):
    """The paper's forest hyperparameters, no feature selection.

    fit_on="train": fit on the training FOVs, score on each test FOV.
    fit_on="test":  fit on each test FOV and score the same rows (in-sample).
    """

    def __init__(self, feature_set: str = "vae", fit_on: str = "train"):
        self.feature_set, self.fit_on = feature_set, fit_on
        self.feature_sets = (feature_set,)
        self.name = f"rf_{feature_set.replace('+', '_')}" + ("_insample" if fit_on == "test" else "")

    def run(self, ctx):
        out = {}
        if self.fit_on == "train":
            model = _forest(ctx.forest_cfg).fit(ctx.X_train(self.feature_set), ctx.train.y)
            for f in ctx.tests:
                out[f] = ArmResult(model.predict(ctx.X_test(f, self.feature_set)))
        else:
            for f, b in ctx.tests.items():
                X = ctx.X_test(f, self.feature_set)
                out[f] = ArmResult(_forest(ctx.forest_cfg).fit(X, b.y).predict(X))
        return out


# ----------------------------------------------------------------------------
# One-stage neural regressors
# ----------------------------------------------------------------------------

def scaled_vae_features(ctx):
    """MinMax-scaled VAE features, scaler fit on the (trimmed) training block.

    Cached on the split, so every network arm sees exactly the same matrices.
    """
    key = ("minmax",)
    if key not in ctx.cache:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler().fit(ctx.X_train("vae"))
        ctx.cache[key] = (scaler.transform(ctx.X_train("vae")),
                          {f: scaler.transform(ctx.X_test(f, "vae")) for f in ctx.tests})
    return ctx.cache[key]


class NeuralArm(Arm):
    """One network, MinMax-scaled VAE features in, N/C ratio out.

    Fit on the training block (minus the trajectories early stopping holds out)
    and scored on each test FOV. ``overrides`` are applied to the run's
    NetConfig, so one arm differs from another only by what its name says.
    """

    feature_sets = ("vae",)

    def __init__(self, name: str, **overrides):
        self.name = name
        self.overrides = overrides

    def cfg(self, ctx) -> NetConfig:
        cfg = copy.deepcopy(ctx.net_cfg)
        for key, value in self.overrides.items():
            if not hasattr(cfg, key):
                raise KeyError(f"{self.name}: NetConfig has no field {key!r}")
            setattr(cfg, key, value)
        return cfg

    def run(self, ctx):
        from .models import DirectRegressor

        Xs_train, Xs_test = scaled_vae_features(ctx)
        cfg = self.cfg(ctx)
        t0 = time.time()
        model = DirectRegressor(cfg, ctx.trajl).fit(Xs_train, ctx.train.y, ctx.train.groups)
        info = {"net_epochs": len(model.history.get("loss", [])),
                "net_best_epoch": int(model.history["best_epoch"][0]),
                "net_seconds": round(time.time() - t0, 1),
                "net_encoder": cfg.encoder, "net_head": cfg.head,
                "net_pool": cfg.pool,
                "n_parameters": sum(p.numel() for p in model.model.parameters())}
        return {f: ArmResult(model.predict(Xs_test[f]), info=dict(info)) for f in ctx.tests}


NOTEBOOK_REGULARIZERS = (("l2", 0.01), ("l1", 0.02))



# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

NEURAL_ARMS: Dict[str, dict] = {
    # Fix 1: the task. Regress the ratio instead of binning it.
    "nn_mlp": {},
    "nn_mlp_log": {"target_transform": "log"},
    "nn_mlp_huber": {"loss": "huber"},
    "nn_mlp_hist": {"head": "histogram"},
    "nn_mlp_weighted": {"label_weighting": "inverse_density"},
    # Fix 2: the architecture. Does the convolution along the latent index earn its cost?
    "nn_conv": {"encoder": "conv"},
    # Sequence encoders on the (trajl x nf) layout, regression only: no class bins,
    # no forest. "last" pooling reads out the snapshot the label belongs to.
    "nn_tconv": {"encoder": "temporal_conv", "pool": "last"},
    "nn_tconv_concat": {"encoder": "temporal_conv", "pool": "concat"},
    "nn_tconv_hist": {"encoder": "temporal_conv", "pool": "last", "head": "histogram"},
    "nn_snapshot": {"encoder": "snapshot_shared", "pool": "last"},
    "nn_snapshot_hist": {"encoder": "snapshot_shared", "pool": "last", "head": "histogram"},
    # Self-attention among the snapshots of a snippet: every snapshot can read
    # every other one, which neither encoder above can do. "last" reads out the
    # token for the snapshot the label belongs to, after it has attended to the
    # other seven.
    "nn_transformer": {"encoder": "transformer", "pool": "last"},
    "nn_transformer_hist": {"encoder": "transformer", "pool": "last", "head": "histogram"},
    # Prices the attention itself: identical block (token projection, both
    # LayerNorms, feed-forward sublayer), attention sublayer made the identity.
    # Whatever this arm scores is what the *block* is worth; only the gap above
    # it belongs to attention.
    "nn_transformer_noattn": {"encoder": "transformer", "pool": "last",
                              "transformer_attention": "none"},
    # The control nn_transformer_hist has to beat before its attention blocks can
    # be said to have earned anything: one learned weight per snapshot from that
    # snapshot's own content, with no pairwise query-key term and no extra depth.
    "nn_snapshot_attn_hist": {"encoder": "snapshot_shared", "pool": "attention",
                              "head": "histogram"},
    # Ablations of the notebook's two choices, one at a time.
    "nn_conv_notebook_bn": {"encoder": "conv", "norm_position": "after_activation",
                            "conv_bias": True},
    "nn_mlp_heavy_reg": {"regularizers": NOTEBOOK_REGULARIZERS},
}

ARMS: Dict[str, Callable[[], Arm]] = {
    "constant": ConstantArm,
    "lasso_classical": lambda: LassoArm("classical"),
    "lasso_vae": lambda: LassoArm("vae"),
    "rf_vae": lambda: ForestArm("vae", "train"),
    "rf_vae_insample": lambda: ForestArm("vae", "test"),
    **{name: (lambda n=name, o=over: NeuralArm(n, **o)) for name, over in NEURAL_ARMS.items()},
}

# Branch ccp_torch_reg predicts N/C with one network: no class bins, no forest.
# The default run is the cheap reference baselines, one honest two-stage
# reference to compare against (published_crossfit), and the one-stage arms.
#
# Deliberately NOT in the default, though all remain registered and runnable by
# name (--arms published_insample,clf_transformer,...):
#   published_insample / _crossfit_groupedval / _holdout / _holdout_groupedval
#       protocol ablations of the two-stage model; the point they make is made
#       in results/simulator_tweaks.md and each costs an RFECV fit.
#   rf_vae_insample      the in-sample demonstration, same reason.
#   clf_*                the classification task this branch exists to drop.
#   nn_mlp_log / _huber / _weighted, nn_conv*, nn_mlp_heavy_reg
#       loss and notebook ablations, already measured at trajl=1.
# At trajl=8 every arm here is expensive enough that "run them all" is not a
# sensible default: list the ones you want.
DEFAULT_ARMS: List[str] = [
    "constant", "lasso_classical", "lasso_vae", "rf_vae",     # baselines
    "published_crossfit",                                     # the two-stage reference
    "nn_mlp", "nn_mlp_hist",                                  # flat, no time axis
    "nn_snapshot", "nn_snapshot_hist",                        # shared per-snapshot encoder
    "nn_snapshot_attn_hist",                                  # + attention pooling (control)
    "nn_tconv", "nn_tconv_hist",                              # convolution along time
    "nn_transformer", "nn_transformer_hist",                  # attention among snapshots
    "nn_transformer_noattn",                                  # ... with the attention ablated
]

ALL_ARMS: List[str] = list(ARMS)


def make_arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; available: {sorted(ARMS)}")
    return ARMS[name]()
