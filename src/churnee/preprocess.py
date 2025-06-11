from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropConstantFeatures, DropFeatures

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

def load_cat_encoding(name:str="onehot"):
    if name == "onehot":
        return OneHotEncoder()
    else:
        raise NotImplementedError()


def load_preprocessor(cols_to_drop:list[str], 
                      numeric_cols:list[str],
                      poly_degree:int=0,
                      add_interaction_only:bool=True,
                      cat_encoder:str='onehot'):
    
    
    scaler = ColumnTransformer([("std",StandardScaler(),numeric_cols)],remainder="passthrough")
    cat_encoder = load_cat_encoding(name=cat_encoder)
    steps = [DropFeatures(cols_to_drop),
            DropConstantFeatures(),
            cat_encoder,                         
            scaler]
    
    if poly_degree > 0 :
        step = PolynomialFeatures(degree=poly_degree,
                           interaction_only=add_interaction_only)
        steps.append(step)
    
    pipe = make_pipeline(*steps)
    
    return pipe

    
    
    
