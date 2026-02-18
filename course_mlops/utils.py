import os
from enum import StrEnum
from typing import Self


class EnvironmentVariable(StrEnum):
    ENV = "CML_ENV"
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    MLFLOW_ARTIFACT_ROOT = "CML_MLFLOW_ARTIFACT_ROOT"
    MLFLOW_UI_BASE = "CML_MLFLOW_UI_BASE"
    MODEL_STRATEGY = "CML_MODEL_STRATEGY"
    API_HOST = "CML_API_HOST"
    API_PORT = "CML_API_PORT"
    LOG_LEVEL = "CML_LOG_LEVEL"
    LOG_FORMAT = "CML_LOG_FORMAT"
    FAIL_FAST = "CML_FAIL_FAST"
    DB_HOST = "CML_DB_HOST"
    DB_PORT = "CML_DB_PORT"
    DB_USER = "CML_DB_USER"
    DB_PASSWORD = "CML_DB_PASSWORD"
    DB_NAME = "CML_DB_NAME"
    DRIFT_WINDOW_SIZE = "CML_DRIFT_WINDOW_SIZE"
    DRIFT_MIN_SAMPLES = "CML_DRIFT_MIN_SAMPLES"

    def read(self: Self, default: str | None = None) -> str:
        try:
            return os.environ[self.value]
        except KeyError as e:
            if default is None:
                raise ValueError(
                    f"Environment variable {self.value} is not defined and no default value was provided"
                ) from e
            return default
