from unittest.mock import Mock

import numpy as np
import pytest

from course_mlops.api.enums import ConfidenceEnum
from course_mlops.api.exceptions import ModelPredictionError
from course_mlops.api.service import PredictionService
from course_mlops.api.service import calculate_confidence
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.train.enums import ModelType


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.95, ConfidenceEnum.HIGH),
        (0.8, ConfidenceEnum.HIGH),
        (0.05, ConfidenceEnum.HIGH),
        (0.2, ConfidenceEnum.HIGH),
        (0.5, ConfidenceEnum.LOW),
        (0.45, ConfidenceEnum.LOW),
        (0.75, ConfidenceEnum.LOW),
        (0.25, ConfidenceEnum.LOW),
    ],
)
def test_calculate_confidence(prob: float, expected: ConfidenceEnum) -> None:
    assert calculate_confidence(prob) == expected


def test_calculate_confidence_custom_thresholds() -> None:
    assert calculate_confidence(0.85, spam=0.9, ham=0.1) == ConfidenceEnum.LOW
    assert calculate_confidence(0.95, spam=0.9, ham=0.1) == ConfidenceEnum.HIGH


def test_prediction_service_initial_state() -> None:
    service = PredictionService(Mock())

    assert service.model is None
    assert service.model_uri is None
    assert service.model_version is None
    assert service.model_type is None
    assert service.strategy is None
    assert service.run_id is None
    assert service.artifact_uri is None
    assert service.registered_at is None


def test_is_loaded_false_when_no_model() -> None:
    service = PredictionService(Mock())
    assert service.is_loaded is False


def test_is_loaded_true_when_model_set() -> None:
    service = PredictionService(Mock())
    service.model = Mock()
    assert service.is_loaded is True


def test_load_model_sets_metadata() -> None:
    loader = Mock()
    mock_model = Mock()
    metadata = {
        "model_uri": "models:/spam/1",
        "model_version": "1",
        "model_type": ModelType.LOGISTIC_REGRESSION,
        "strategy": "latest",
        "run_id": "run123",
        "artifact_uri": "s3://artifacts",
        "registered_at": "2024-01-15T10:30:00+00:00",
    }
    loader.load.return_value = (mock_model, metadata)

    service = PredictionService(loader)
    service.load_model("spam", "latest")

    assert service.model is mock_model
    assert service.model_uri == "models:/spam/1"
    assert service.model_version == "1"
    assert service.model_type == ModelType.LOGISTIC_REGRESSION
    assert service.strategy == "latest"
    assert service.run_id == "run123"
    assert service.artifact_uri == "s3://artifacts"
    assert service.registered_at == "2024-01-15T10:30:00+00:00"
    loader.load.assert_called_once_with("spam", "latest")


@pytest.mark.parametrize(
    ("predictions", "probabilities", "expected_label", "expected_prob"),
    [
        (np.array([1]), np.array([[0.08, 0.92]]), "spam", 0.92),
        (np.array([0]), np.array([[0.9, 0.1]]), "ham", 0.1),
    ],
)
def test_predict_returns_correct_result(
    predictions: np.ndarray, probabilities: np.ndarray, expected_label: str, expected_prob: float
) -> None:
    service = PredictionService(Mock())
    service.model = Mock()
    service.model.predict.return_value = {
        "predictions": predictions,
        "probabilities": probabilities,
    }

    result = service.predict("test message")

    assert result["prediction"] == expected_label
    assert result["probability"] == pytest.approx(expected_prob)


def test_predict_without_probabilities() -> None:
    service = PredictionService(Mock())
    service.model = Mock()
    service.model.predict.return_value = {"predictions": np.array([0])}

    result = service.predict("Hello")

    assert result["prediction"] == "ham"
    assert result["probability"] is None


def test_predict_raises_when_model_not_loaded() -> None:
    service = PredictionService(Mock())

    with pytest.raises(ModelNotLoadedError):
        service.predict("Hello")


def test_predict_raises_on_model_error() -> None:
    service = PredictionService(Mock())
    service.model = Mock()
    service.model.predict.side_effect = RuntimeError("Model crashed")

    with pytest.raises(ModelPredictionError, match="Model prediction failed"):
        service.predict("Hello")


def test_predict_raises_on_bad_output_format() -> None:
    service = PredictionService(Mock())
    service.model = Mock()
    service.model.predict.return_value = {"wrong_key": []}

    with pytest.raises(ModelPredictionError, match="Unexpected model output format"):
        service.predict("Hello")
