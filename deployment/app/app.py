import litserve as ls
from litserve.mcp import MCP
from pydantic import BaseModel
import joblib
import logging
import pandas as pd
import numpy as np
from fastapi import HTTPException
import os


LOGGER = logging.getLogger("Inference")


class InferenceEngine(object):
    def __init__(self, estimator_path: str):
        self.estimator = joblib.load(estimator_path)

        LOGGER.info("Model loaded successfully.")

    def predict(self, x: pd.DataFrame | dict) -> list:
        try:
            if isinstance(x, dict):
                x = pd.DataFrame.from_dict(x)
            y = self.estimator.predict(x)
            return y.tolist()

        except Exception as e:
            LOGGER.error(f"Prediction failed -> {e}")
            raise HTTPException(status_code=400, detail=str(e))


class InputRequest(BaseModel):
    input: dict


class API(ls.LitAPI):
    def setup(self, device):
        path = os.environ.get("MODEL_PATH")

        if path is None:
            path = os.path.join("/models", os.environ["MODEL_NAME"])

        self.model = InferenceEngine(estimator_path=path)

    def decode_request(self, request: InputRequest):
        try:
            x = pd.DataFrame.from_dict(request.input, orient="columns")
            return x
        except Exception as e:
            LOGGER.error(f"Prediction failed -> {e}")
            raise HTTPException(status_code=400, detail=str(e))

    # def batch(self, inputs):
    #     return np.stack(inputs)

    def predict(self, x) -> list:
        return self.model.predict(x)

    # def unbatch(self, output):
    #     return list(output)

    def encode_response(self, output) -> dict:
        return {"output": output}


if __name__ == "__main__":
    mcp = MCP(name="churn_predictor", description="Churn prediction")
    api = API(mcp=mcp)
    server = ls.LitServer(
        api,
        accelerator="cpu",
        workers_per_device=1,
        max_batch_size=1,
        batch_timeout=os.environ.get("BATCH_TIMEOUT", 0),
    )
    server.run(
        port=4141,
    )
