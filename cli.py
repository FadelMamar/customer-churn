import fire
from churnee.models import Model
from churnee.inference import InferenceEngine
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data
import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime, date
LOGGER = logging.getLogger("CLI")


class CLI(object):
    def __init__(
        self,
        config_yaml:str=None,
        model_name:str="logisticreg",
        numeric_cols:list[str]=None,
        cols_to_drop:list[str]=None,
        param_grid:dict=None
    ):
        self.numeric_cols = numeric_cols
        self.cols_to_drop = cols_to_drop
        self.param_grid = param_grid
        self.model_name = model_name

        if config_yaml:
            with open(config_yaml,"r") as file:
                self.cfg = yaml.safe_load(file)
                # print(self.cfg)
            self.numeric_cols = self.cfg.get("numeric_cols")
            self.cols_to_drop = self.cfg.get("cols_to_drop")
            self.param_grid = dict()
            for param in self.cfg.get("param_grid"):
                self.param_grid.update(param)

        else:
            self.model_name = "logisticreg"
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
            self.param_grid = {
                "penalty": ["l1", "l2"],
                "tol": [
                    1e-4,
                ],
                "C": np.logspace(-2, 2, 10).tolist(),
            }

        LOGGER.info(f"Loading {self.model_name}")

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
        save_path: str = None,
    ):
        
        preprocessor = load_preprocessor(
            cols_to_drop=self.cols_to_drop,
            numeric_cols=self.numeric_cols,
            poly_degree=self.cfg.get("poly_degree",0),
            cat_encoder="onehot",
            add_interaction_only=self.cfg.get("poly_degree",False),
        )

        model = Model(name=self.model_name, preprocessor=preprocessor)

        X_tr, y_tr = load_data(data_path)

        if save_path is None:
            save_dir = Path(__file__).resolve().parent / "models"
            current_time = datetime.now().strftime("%H-%M")
            filename = self.model_name + f"_{str(date.today())}_{current_time}.joblib"
            save_path = save_dir / filename

        model.tune(
            X=X_tr,
            y=y_tr,
            param_grid=self.param_grid,
            algorithm=self.cfg.get("algorithm","optuna"),
            scoring=self.cfg.get("scoring","f1"),
            num_folds=self.cfg.get("num_folds",5),
            shuffle=self.cfg.get("shuffle_data",5),
            verbose=self.cfg.get("verbose",1),
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
    fire.Fire(CLI)
