from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import get_scorer
import pandas as pd
import numpy as np



class Model(object):
    
    def __init__(self,name:str, preprocessor:Pipeline):
        # self.processor = preprocessor
        estimator = self._load_model(name=name)
        self.estimator = Pipeline([("preprocessor",preprocessor),("model",estimator)])
        
    
    def _load_model(self,name:str,solver="liblinear"):
        
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
            
    def train(self,X:pd.DataFrame,y:np.ndarray):
        self.model.fit(X=X,y=y)
        return self
    
    def val(self,X:pd.DataFrame,y:np.ndarray,scoring:str)->float:
        scorer = get_scorer(scoring=scoring)
        return scorer(self.model,X,y)
    
    def tune(self,
            X:pd.DataFrame,
            y:np.ndarray,
            param_grid:dict,
            algorithm:str="grid",
            scoring:str="f1",
            num_folds:int=5,
            n_iter=20,
            shuffle:bool=False):
        
        cfg = dict(estimator=self.estimator,
                          scoring=scoring,
                          refit=True,
                          cv=StratifiedKFold(n_splits=num_folds,
                                             shuffle=shuffle
                                             ),
                          )
        
        if algorithm == "grid":
            tuner = GridSearchCV(param_grid=param_grid,**cfg)
            
        elif algorithm == "random":
            tuner = RandomizedSearchCV(param_distributions=param_grid,
                                       n_iter=n_iter,
                                       **cfg)
        
        else:
            raise NotImplementedError()
            
        tuner.fit(X=X,y=y)
        
        self.estimator = tuner.best_estimator_
        
        return self
        