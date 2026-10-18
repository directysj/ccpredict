"""Predict cell-cycle reporter readout (nucleus/cytoplasm ratio) from O2VAE
morphology embeddings along single-cell trajectories.

One stage: a single network regresses the N/C ratio directly from the snippet's
features. The published two-stage model (a CNN classifying the binned ratio,
then an RFECV random forest on the class probabilities) is not on this branch;
it is on ``ccp_torch``.
"""
from .config import (FeatureConfig, ForestConfig, NetConfig, PathsConfig, PipelineConfig,
                     QCConfig, SplitConfig)
from .data import (FOVSnippets, LabeledFOV, PreparedData, ReporterTracks, match_labels,
                   prepare_data, read_reporter_file, sliding_snippets)
from .models import DirectRegressor, build_net
from .preprocessing import TrainTestData, build_train_test, trim_by_histogram

__all__ = [
    "PipelineConfig", "PathsConfig", "QCConfig", "FeatureConfig", "SplitConfig",
    "ForestConfig", "NetConfig", "DirectRegressor", "build_net",
    "prepare_data", "PreparedData", "LabeledFOV", "FOVSnippets", "ReporterTracks",
    "sliding_snippets", "read_reporter_file", "match_labels", "build_train_test",
    "TrainTestData", "trim_by_histogram",
]
