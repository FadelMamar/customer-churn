from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
    TunedThresholdClassifierCV,
)
from sklearn.metrics import classification_report, PrecisionRecallDisplay

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import optuna
from optuna.integration import OptunaSearchCV

import logging

LOGGER = logging.getLogger("Model")


class Model(object):
    def __init__(self, name: str, preprocessor: Pipeline):
        # self.processor = preprocessor
        estimator = self._load_model(name=name)
        self.estimator = Pipeline(
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        self._tuner = None

    def _load_model(self, name: str, solver="liblinear"):
        if name == "svc":
            return SVC()
        elif name == "logisticreg":
            return LogisticRegression(solver=solver)
        elif name == "ridge":
            return RidgeClassifier()
        elif name == "gradboosting":
            return GradientBoostingClassifier()
        elif name == "rf":
            return RandomForestClassifier()
        elif name == "decisiontree":
            return DecisionTreeClassifier()
        else:
            raise NotImplementedError()

    def train(self, X: pd.DataFrame, y: np.ndarray):
        self.estimator.fit(X=X, y=y)
        return self

    def val(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        pr_curve: bool = False,
        tune_threshold: bool = False,
    ) -> float:
        
        estimator = self.estimator

        if tune_threshold:
            LOGGER.info("Tuning threshold using F1-score.")
            estimator = TunedThresholdClassifierCV(
                self.estimator, scoring="f1", cv=StratifiedKFold(n_splits=5), refit=True
            )
        y_pred = estimator.predict(X)

        report = classification_report(
            y_true=y, y_pred=y_pred, labels=[0, 1], target_names=["no_churn", "churn"]
        )

        if pr_curve:
            fig, ax = plt.subplots(1, 1)
            PrecisionRecallDisplay.from_estimator(estimator, X, y, pos_label=1, ax=ax)
            return fig, report

        return report

    def tune(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        param_grid: dict,
        algorithm: str = "grid",
        scoring: str = "f1",
        num_folds: int = 5,
        n_iter=20,
        verbose: int = 0,
        n_jobs: int = 1,
        save_path: str = None,
        study: optuna.study.Study = None,
        shuffle: bool = False,
    ):
        cfg = dict(
            estimator=self.estimator,
            scoring=scoring,
            refit=True,
            error_score="raise",
            cv=StratifiedKFold(n_splits=num_folds, shuffle=shuffle),
            verbose=verbose,
            n_jobs=n_jobs,
        )

        param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        if algorithm == "grid":
            self._tuner = GridSearchCV(param_grid=param_grid, **cfg)

        elif algorithm == "random":
            self._tuner = RandomizedSearchCV(
                param_distributions=param_grid, n_iter=n_iter, **cfg
            )

        elif algorithm == "optuna":
            param_grid = {
                k: optuna.distributions.CategoricalDistribution(v)
                for k, v in param_grid.items()
            }
            self._tuner = OptunaSearchCV(
                param_distributions=param_grid, n_trials=n_iter, study=study, **cfg
            )

        else:
            raise NotImplementedError()

        self._tuner.fit(X=X, y=y)

        self.estimator = self._tuner.best_estimator_

        LOGGER.info(f"Best {scoring}:{self._tuner.best_score_}")
        LOGGER.info(f"Best params:{self._tuner.best_params_}")

        try:
            if save_path:
                joblib.dump(self.estimator, save_path)
        except Exception as e:
            LOGGER.error(f"Failed to save the estimator.\n{e}")

        return self
