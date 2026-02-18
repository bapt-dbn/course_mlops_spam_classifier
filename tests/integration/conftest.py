import os
from collections.abc import Generator

import httpx
import pytest
from mlflow.tracking import MlflowClient

MODEL_NAME = "spam-classifier"
MLFLOW_TRACKING_URI = "http://localhost:5001"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI


@pytest.fixture(scope="session")
def client() -> Generator[httpx.Client, None, None]:
    with httpx.Client(base_url="http://localhost:8000", timeout=30) as c:
        yield c


@pytest.fixture(scope="session")
def mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
