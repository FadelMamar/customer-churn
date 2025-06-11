import pandas as pd
import numpy as np

def load_data(path:str)->tuple[pd.DataFrame,np.ndarray]:
    
    df = pd.read_csv(path)
    
    df = df.dropna()
    
    X = df.drop(columns=['Churn'])
    y = df['Churn'].to_numpy()
    
    return X,y