# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 18:52:24 2025

@author: FADELCO
"""

from churnee.models import Model
from churnee.preprocess import load_preprocessor
from churnee.utils import load_data

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

X_preprocessed = preprocessor.fit_transform(X_tr,y_tr)

model.train(X_tr,y_tr)

score = model.val(X_tr,y_tr,scoring="f1")
