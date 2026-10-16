"""CellCyclePipeline: the notebook as a sequence of callable stages.

    pipe = CellCyclePipeline(PipelineConfig.from_json("config.json"))
    data = pipe.prepare_data()                  # In [4]-[21]
    tt = pipe.build_train_test(data)            # In [22]-[23]
    clf, clf_out = pipe.run_classifier(tt)      # In [25]-[41]
    reg, reg_out = pipe.run_regressor(tt, clf)  # In [42], [44] + unexecuted cells

or pipe.run() for all of it. Each stage can be rerun alone, e.g. a new
regressor on a saved classifier via StageOneClassifier.load.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import evaluation as ev
from .config import PipelineConfig
from .data import PreparedData, prepare_data
from .models import StageOneClassifier, StageTwoRegressor, out_of_fold_proba, stack_features
from .preprocessing import TrainTestData, build_train_test, discretize

log = logging.getLogger(__name__)


@dataclass
class ClassifierOutput:
    y_train_class: np.ndarray
    y_test_class: np.ndarray
    proba_test: np.ndarray
    predicted_test: np.ndarray
    metrics: dict


@dataclass
class RegressorOutput:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    y_pred: np.ndarray
    metrics: dict
    fit_on: str


@dataclass
class PipelineResults:
    data: PreparedData
    train_test: TrainTestData
    classifier: StageOneClassifier
    classifier_output: ClassifierOutput
    regressor: Optional[StageTwoRegressor] = None
    regressor_output: Optional[RegressorOutput] = None
    figures: dict = field(default_factory=dict)


class CellCyclePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    # ---- naming -------------------------------------------------------------
    @property
    def tag(self) -> str:
        """Suffix used in output file names, as in the notebook's savefig paths."""
        c = self.config
        return (f"{c.split.test_fov}_trajl{c.features.trajl}_VAE{c.features.latent_dim}"
                f"Cls{len(c.classifier.class_boundaries) + 1}")

    def _out(self, *parts: str) -> Optional[str]:
        out = self.config.paths.output_dir
        if out is None:
            return None
        path = os.path.join(out, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _figure(self, results_figs: dict, name: str, fig) -> None:
        path = self._out("figs", f"{name}_{self.tag}.png")
        if path:
            ev.save_figure(fig, path)
            results_figs[name] = path
        else:
            results_figs[name] = fig

    # ---- stages -------------------------------------------------------------
    def prepare_data(self) -> PreparedData:
        c = self.config
        return prepare_data(c.fovs, c.paths, c.qc, c.features)

    def build_train_test(self, data: PreparedData) -> TrainTestData:
        return build_train_test(data.fovs, self.config.split)

    def run_classifier(self, tt: TrainTestData,
                       classifier: Optional[StageOneClassifier] = None):
        """Bin labels, fit (unless a fitted classifier is given), predict the test FOV."""
        cfg = self.config.classifier
        num_classes = len(cfg.class_boundaries) + 1
        y_train_class = discretize(tt.y_train, cfg.class_boundaries)
        y_test_class = discretize(tt.y_test, cfg.class_boundaries)

        if classifier is None:
            classifier = StageOneClassifier(cfg, num_classes).fit(tt.X_train, y_train_class,
                                                                  groups=tt.train_groups)
        proba = classifier.predict_proba(tt.X_test)
        predicted = np.argmax(proba, axis=1)
        metrics = ev.classification_metrics(y_test_class, predicted, num_classes)
        log.info("stage 1 on %s: accuracy %.3f, weighted F1 %.3f",
                 tt.test_fov, metrics["accuracy"], metrics["f1_weighted"])
        return classifier, ClassifierOutput(y_train_class, y_test_class, proba, predicted, metrics)

    def run_regressor(self, tt: TrainTestData, classifier: StageOneClassifier,
                      proba_test: Optional[np.ndarray] = None):
        cfg = self.config.regressor
        if proba_test is None:
            proba_test = classifier.predict_proba(tt.X_test)
        X_eval = stack_features(tt.X_test, proba_test)

        if cfg.fit_on == "test":
            log.warning("regressor fit_on='test': fitted and scored on the same test-FOV rows; "
                        "the metrics are in-sample (notebook behaviour)")
            X_fit, y_fit = X_eval, tt.y_test
        elif cfg.fit_on == "train":
            # Probabilities on the classifier's own training rows are in-sample,
            # so they are more confident than on new data. Cross-fitted
            # probabilities would remove that mismatch.
            X_fit = stack_features(tt.X_train, classifier.predict_proba(tt.X_train))
            y_fit = tt.y_train
        elif cfg.fit_on == "train_crossfit":
            clf_cfg = self.config.classifier
            y_class = discretize(tt.y_train, clf_cfg.class_boundaries)
            oof = out_of_fold_proba(tt.X_train, y_class, tt.train_fold, clf_cfg,
                                    len(clf_cfg.class_boundaries) + 1, groups=tt.train_groups)
            X_fit, y_fit = stack_features(tt.X_train, oof), tt.y_train
        else:
            raise ValueError(f"unknown fit_on {cfg.fit_on!r}")

        regressor = StageTwoRegressor(cfg).fit(X_fit, y_fit)
        y_pred = regressor.predict(X_eval)
        metrics = ev.regression_metrics(tt.y_test, y_pred)
        metrics["n_features_selected"] = regressor.n_features_used
        log.info("stage 2 on %s (fit_on=%s): MSE %.3f, R %.3f, R^2 %.3f", tt.test_fov,
                 cfg.fit_on, metrics["mse"], metrics["pearson_r"], metrics["r2"])
        return regressor, RegressorOutput(X_fit, y_fit, X_eval, tt.y_test, y_pred, metrics, cfg.fit_on)

    # ---- everything ---------------------------------------------------------
    def run(self, with_regressor: bool = True, make_figures: bool = True) -> PipelineResults:
        figs: dict = {}
        data = self.prepare_data()
        tt = self.build_train_test(data)
        if make_figures:
            self._figure(figs, "distN2Cratio", ev.plot_label_distributions(
                tt.y_train, tt.y_test, f"{'+'.join(tt.train_fovs)}: Training", f"{tt.test_fov}: Test Set"))

        classifier, clf_out = self.run_classifier(tt)
        if make_figures:
            self._figure(figs, "confMatrixNN", ev.plot_confusion_matrix(
                clf_out.metrics["confusion_matrix_true_normalized"]))
            self._figure(figs, "performanceNN", ev.plot_training_history(classifier.history))

        results = PipelineResults(data, tt, classifier, clf_out, figures=figs)
        if with_regressor:
            regressor, reg_out = self.run_regressor(tt, classifier, clf_out.proba_test)
            results.regressor, results.regressor_output = regressor, reg_out
            if make_figures:
                self._figure(figs, "ratioN2C_RF-NN", ev.plot_regression(
                    reg_out.y_eval, reg_out.y_pred, f"RF on VAE + NN probabilities: {tt.test_fov}"))
                if regressor.rfecv is not None:
                    self._figure(figs, "ratioN2C_RF-NN_score", ev.plot_rfecv_scores(regressor.rfecv))

        self.save(results)
        return results

    def save(self, results: PipelineResults) -> None:
        out = self.config.paths.output_dir
        if out is None:
            return
        models_dir = os.path.join(out, "models", self.tag)
        results.classifier.save(models_dir)
        if results.regressor is not None:
            results.regressor.save(models_dir)
        summary = {
            "config": self.config.to_dict(),
            "good_fovs": results.data.good_fovs,
            "cell_count_noise": results.data.cell_count_noise,
            "n_labelled_snippets": {f: int(len(v.y)) for f, v in results.data.fovs.items()},
            "n_train_after_trim": int(len(results.train_test.y_train)),
            "classifier": results.classifier_output.metrics,
            "regressor": None if results.regressor_output is None else {
                "fit_on": results.regressor_output.fit_on, **results.regressor_output.metrics},
        }
        path = self._out("metrics", f"performance_{self.tag}.json")
        with open(path, "w") as fh:
            json.dump(ev.as_jsonable(summary), fh, indent=2)
        log.info("wrote %s", path)
