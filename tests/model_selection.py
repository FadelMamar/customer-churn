# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 19:27:14 2025

@author: FADELCO
"""


from churnee.models import Model
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data
import numpy as np

numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
       'Payment Delay', 'Total Spend', 'Last Interaction']

cols_to_drop=["CustomerID"]

preprocessor = load_preprocessor(cols_to_drop=cols_to_drop,
                                 numeric_cols=numeric_cols,
                                 poly_degree=0,
                                 cat_encoder="onehot"
                                 )

model = Model(name="logisticreg",preprocessor=preprocessor)


X_tr, y_tr = load_data(r"..\data\train.csv")

param_dict = {"penalty":["l1","l2"],
              "tol":[1e-4,],
              "C":np.logspace(-2,2,10).tolist()
              }

param_dict = {f"model__{k}":v for k,v in param_dict.items()}


model.tune(X=X_tr,
            y=y_tr,
            param_grid=param_dict,
            algorithm="random",
            scoring="f1",
            num_folds=5,
            shuffle=False
            )

estimator = model.estimator
