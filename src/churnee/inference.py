import pandas as pd
import numpy as np
import joblib
import logging

LOGGER = logging.getLogger("Inference")


class InferenceEngine(object):
    def __init__(self, estimator_path: str):
        self.estimator = joblib.load(estimator_path)

        LOGGER.info("Model loaded successfully.")

    def predict(self, x: pd.DataFrame | dict) -> np.ndarray:
        try:
            if isinstance(x, dict):
                x = pd.DataFrame.from_dict(x)
            y = self.estimator.predict(x)
            return y

        except Exception as e:
            LOGGER.error(f"Prediction failed -> {e}")
            return None
