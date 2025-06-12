# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 19:27:14 2025

@author: FADELCO
"""

from churnee.models import Model
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data
import numpy as np
from datetime import datetime, date
from pathlib import Path


def run(data_path:str=r"..\data\train.csv", 
        model_name:str = "logisticreg",
        algorithm="optuna"):
    
    numeric_cols = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
    ]

    cols_to_drop = ["CustomerID"]

    preprocessor = load_preprocessor(
        cols_to_drop=cols_to_drop,
        numeric_cols=numeric_cols,
        poly_degree=0,
        cat_encoder="onehot",
        add_interaction_only=True,
    )

    save_dir = Path(__file__).resolve().parent.parent / "models"
    model = Model(name=model_name, preprocessor=preprocessor)
    current_time = datetime.now().strftime("%H-%M")
    filename = model_name + f"_{str(date.today())}_{current_time}.joblib"
    save_path = save_dir / filename

    X_tr, y_tr = load_data(data_path)

    param_grid = {
        "penalty": ["l1", "l2"],
        "tol": [
            1e-4,
        ],
        "C": np.logspace(-2, 2, 10).tolist(),
    }

    model.tune(
        X=X_tr,
        y=y_tr,
        param_grid=param_grid,
        algorithm=algorithm,
        scoring="f1",
        num_folds=5,
        shuffle=False,
        verbose=1,
        n_jobs=1,
        save_path=save_path
    )

    estimator = model.estimator

    print(estimator)

if __name__ == "__main__":
    run()
