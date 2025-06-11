# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 19:27:14 2025

@author: FADELCO
"""

from churnee.models import Model
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data
import numpy as np

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

model = Model(name="logisticreg", preprocessor=preprocessor)


X_tr, y_tr = load_data(r"..\data\train.csv")

param_grid = {
    "penalty": ["l1", "l2"],
    "tol": [
        1e-4,
    ],
    "C": np.logspace(-2, 2, 10).tolist(),
}

param_grid = {f"model__{k}": v for k, v in param_grid.items()}

# param_dict.update({
#     })


model.tune(
    X=X_tr,
    y=y_tr,
    param_grid=param_grid,
    algorithm="optuna",
    scoring="f1",
    num_folds=5,
    shuffle=False,
    verbose=1,
)

estimator = model.estimator

print(estimator)
