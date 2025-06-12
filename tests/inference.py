

def run_webservice(x:dict):
    import requests
    response = requests.post("http://127.0.0.1:4141/predict", json={"input": x})
    print(f"Status: {response.status_code}\nResponse:\n {response.text}")

def run(x:str|dict):
    from churnee.inference import InferenceEngine
    import pandas as pd

    estimator_path = r"D:\workspace\repos\customer-churn\models\logisticreg_2025-06-12_14-06.joblib"

    if isinstance(x, str):
            x = pd.read_csv(x)
    elif isinstance(x, dict):
        pass
    elif isinstance(x, pd.DataFrame):
        pass
    else:
        raise ValueError("'x' type is not supported. Supports 'str', 'dict', 'pd.DataFrame'.")
    
    y = InferenceEngine(estimator_path=estimator_path).predict(x=x)

    print("prediction:",y)


if __name__ == "__main__":
    from churnee.utils import load_data

    X,y = load_data(path=r"D:\workspace\repos\customer-churn\data\train.csv")

    x_pred = X.sample(frac=0.05).to_dict(orient='list')

    # run_webservice(x=x_pred)
    run(x=x_pred)