import fire
from churnee.models import Model
from churnee.inference import InferenceEngine
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data
import numpy as np
import pandas as pd

import logging

LOGGER = logging.getLogger("CLI")


class CLI(object):
    def __init__(
        self,
    ):
        self.numeric_cols = [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction",
        ]

        self.cols_to_drop = ["CustomerID"]

    def train(self, data_path: str):
        preprocessor = load_preprocessor(
            cols_to_drop=self.cols_to_drop,
            numeric_cols=self.numeric_cols,
            poly_degree=0,
            cat_encoder="onehot",
        )

        model = Model(name="logisticreg", preprocessor=preprocessor)

        X_tr, y_tr = load_data(data_path)

        model.train(X_tr, y_tr)
        score = model.val(X_tr, y_tr, pr_curve=False, tune_threshold=False)

        LOGGER.info(f"Train results: {score:.3f}")

    def tune(
        self,
        data_path: str,
        algorithm="optuna",
        num_folds: int = 5,
        shuffle_data: bool = False,
        poly_degree: int = 0,
        save_path: str = None,
    ):
        preprocessor = load_preprocessor(
            cols_to_drop=self.cols_to_drop,
            numeric_cols=self.numeric_cols,
            poly_degree=poly_degree,
            cat_encoder="onehot",
            add_interaction_only=False,
        )

        model = Model(name="logisticreg", preprocessor=preprocessor)

        X_tr, y_tr = load_data(data_path)

        param_grid = {
            "penalty": ["l1", "l2"],
            "tol": [
                1e-4,
            ],
            "C": np.logspace(-2, 2, 10).tolist(),
        }

        param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        model.tune(
            X=X_tr,
            y=y_tr,
            param_grid=param_grid,
            algorithm=algorithm,
            scoring="f1",
            num_folds=num_folds,
            shuffle=shuffle_data,
            verbose=1,
            save_path=save_path,
        )

    def predict(self, estimator_path: str, x: dict | str):
        if isinstance(x, str):
            x = pd.read_csv(x)
        else:
            assert isinstance(x, dict)
        y = InferenceEngine(estimator_path=estimator_path).predict(x=x)
        return y


if __name__ == "__main__":
    fire.Fire()
