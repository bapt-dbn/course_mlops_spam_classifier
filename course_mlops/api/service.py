import logging
from typing import TypedDict

import pandas as pd
from mlflow.pyfunc import PyFuncModel

from course_mlops.api.enums import ConfidenceEnum
from course_mlops.api.exceptions import ModelPredictionError
from course_mlops.api.loader import ModelLoader
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.train.preprocessing.data import DatasetColumn

logger = logging.getLogger(__name__)


def calculate_confidence(prob: float, spam: float = 0.8, ham: float = 0.2) -> ConfidenceEnum:
    if prob >= spam or prob <= ham:
        return ConfidenceEnum.HIGH
    return ConfidenceEnum.LOW


class PredictionResult(TypedDict):
    prediction: str
    probability: float | None


class PredictionService:
    def __init__(self, loader: ModelLoader) -> None:
        self.model: PyFuncModel | None = None
        self.loader = loader
        self.model_uri: str | None = None
        self.model_version: str | None = None
        self.model_type: str | None = None
        self.strategy: str | None = None
        self.run_id: str | None = None
        self.artifact_uri: str | None = None
        self.registered_at: str | None = None

    def load_model(self, model_name: str, strategy: str) -> None:
        self.model, metadata = self.loader.load(model_name, strategy)
        self.model_uri = metadata.get("model_uri")
        self.model_version = metadata.get("model_version")
        self.model_type = metadata.get("model_type")
        self.strategy = metadata.get("strategy")
        self.run_id = metadata.get("run_id")
        self.artifact_uri = metadata.get("artifact_uri")
        self.registered_at = metadata.get("registered_at")
        logger.info(f"Model loaded: {self.model_uri} (type: {self.model_type}, strategy: {strategy})")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, message: str) -> PredictionResult:
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedError("Model not loaded")

        df = pd.DataFrame({DatasetColumn.MESSAGE: [message]})

        try:
            result = self.model.predict(df)
        except Exception as e:
            raise ModelPredictionError(f"Model prediction failed: {e}") from e

        try:
            prediction_label = int(result["predictions"][0])
            probabilities = result.get("probabilities")
            spam_prob = float(probabilities[0][1]) if probabilities is not None else None
        except (KeyError, IndexError, TypeError) as e:
            raise ModelPredictionError(f"Unexpected model output format: {e}") from e

        return {
            "prediction": "spam" if prediction_label == 1 else "ham",
            "probability": spam_prob,
        }
