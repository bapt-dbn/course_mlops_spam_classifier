from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from course_mlops.api.schemas import ErrorOutput
from course_mlops.api.schemas import ModelInfoOutput
from course_mlops.api.schemas import ModelLoadInput
from course_mlops.api.schemas import PredictInput
from course_mlops.train.enums import ModelType


def test_predict_input_valid() -> None:
    inp = PredictInput(message="Hello")
    assert inp.message == "Hello"


def test_predict_input_max_length() -> None:
    inp = PredictInput(message="x" * 5000)
    assert len(inp.message) == 5000


@pytest.mark.parametrize(
    "message",
    [
        "",
        "x" * 5001,
    ],
)
def test_predict_input_rejects_invalid_length(message: str) -> None:
    with pytest.raises(ValidationError):
        PredictInput(message=message)


def test_model_load_input_default_strategy() -> None:
    inp = ModelLoadInput()
    assert inp.strategy == "latest"


def test_model_load_input_custom_strategy() -> None:
    inp = ModelLoadInput(strategy="best")
    assert inp.strategy == "best"


def test_model_info_output_from_service_with_run_id() -> None:
    service = Mock()
    service.model_version = "1"
    service.model_type = ModelType.LOGISTIC_REGRESSION
    service.strategy = "latest"
    service.run_id = "abc123"
    service.artifact_uri = "s3://artifacts"
    service.registered_at = "2024-01-15T10:30:00+00:00"

    result = ModelInfoOutput.from_service(service, "spam-classifier", "http://mlflow:5000")

    assert result.model_name == "spam-classifier"
    assert result.mlflow_ui_url == "http://mlflow:5000/#/runs/abc123"
    assert result.run_id == "abc123"


def test_model_info_output_from_service_without_run_id() -> None:
    service = Mock()
    service.model_version = "1"
    service.model_type = None
    service.strategy = "latest"
    service.run_id = None
    service.artifact_uri = None
    service.registered_at = None

    result = ModelInfoOutput.from_service(service, "spam-classifier", "http://mlflow:5000")

    assert result.mlflow_ui_url == "http://mlflow:5000"
    assert result.run_id is None


def test_error_output_default_timestamp() -> None:
    output = ErrorOutput(detail="Something went wrong")

    assert output.detail == "Something went wrong"
    assert output.error_code is None
    assert output.timestamp is not None
