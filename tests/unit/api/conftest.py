from collections.abc import Generator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from course_mlops.api.dependencies import get_prediction_service
from course_mlops.api.exception_handlers import course_mlops_exception_handler
from course_mlops.api.exception_handlers import general_exception_handler
from course_mlops.api.loader import ModelLoader
from course_mlops.api.routes import router
from course_mlops.exceptions import CourseMLOpsError
from course_mlops.train.config import Settings
from course_mlops.train.enums import ModelType


@pytest.fixture
def mock_prediction_service() -> Mock:
    service = Mock()
    service.is_loaded = True
    service.model = Mock()
    service.model_uri = "models:/spam-classifier/1"
    service.model_version = "1"
    service.model_type = ModelType.LOGISTIC_REGRESSION
    service.strategy = "latest"
    service.run_id = "abc123"
    service.artifact_uri = "s3://mlflow/artifacts"
    service.registered_at = "2024-01-15T10:30:00+00:00"
    service.predict.return_value = {"prediction": "ham", "probability": 0.1}
    return service


@pytest.fixture
def app(mock_prediction_service: Mock) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.add_exception_handler(CourseMLOpsError, course_mlops_exception_handler)
    test_app.add_exception_handler(Exception, general_exception_handler)
    test_app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    return test_app


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _mock_settings() -> Generator[None, None, None]:
    with patch("course_mlops.api.routes.get_settings", return_value=Settings()):
        yield


@pytest.fixture
def loader() -> Generator[ModelLoader, None, None]:
    """ModelLoader with tracking URI and MlflowClient already patched."""
    with (
        patch.object(ModelLoader, "_get_tracking_uri", return_value="http://mlflow:5000"),
        patch("course_mlops.api.loader.mlflow"),
        patch("course_mlops.api.loader.MlflowClient"),
    ):
        inst = ModelLoader()
        _ = inst.client
        yield inst
