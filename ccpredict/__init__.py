"""Predict cell-cycle reporter readout (nucleus/cytoplasm ratio) from O2VAE
morphology embeddings along single-cell trajectories.

Two stages: a CNN+DNN classifies the binned N/C ratio, then a random forest
(with RFECV feature selection) regresses the N/C ratio from the features plus
the classifier's class probabilities.
"""
from .config import (ClassifierConfig, FeatureConfig, PathsConfig, PipelineConfig, QCConfig,
                     RegressorConfig, SplitConfig)
from .data import (FOVSnippets, LabeledFOV, PreparedData, ReporterTracks, match_labels,
                   prepare_data, read_reporter_file, sliding_snippets)
from .models import StageOneClassifier, StageTwoRegressor, build_cnn_classifier, stack_features
from .pipeline import CellCyclePipeline, PipelineResults
from .preprocessing import TrainTestData, build_train_test, discretize, trim_by_histogram

__all__ = [
    "PipelineConfig", "PathsConfig", "QCConfig", "FeatureConfig", "SplitConfig",
    "ClassifierConfig", "RegressorConfig", "CellCyclePipeline", "PipelineResults",
    "prepare_data", "PreparedData", "LabeledFOV", "FOVSnippets", "ReporterTracks",
    "sliding_snippets", "read_reporter_file", "match_labels", "build_train_test",
    "TrainTestData", "trim_by_histogram", "discretize", "build_cnn_classifier",
    "StageOneClassifier", "StageTwoRegressor", "stack_features",
]
