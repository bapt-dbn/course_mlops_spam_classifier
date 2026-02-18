from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from course_mlops.api.routes import get_prediction_service
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.train.enums import ModelType


def test_get_prediction_service_returns_service() -> None:
    mock_request = Mock()
    mock_service = Mock()
    mock_request.app.state.prediction_service = mock_service

    assert get_prediction_service(mock_request) is mock_service


def test_get_prediction_service_raises_when_none() -> None:
    mock_request = Mock()
    mock_request.app.state.prediction_service = None

    with pytest.raises(ModelNotLoadedError, match="Prediction service not initialized"):
        get_prediction_service(mock_request)


def test_get_prediction_service_raises_when_attr_missing() -> None:
    mock_request = Mock()
    mock_request.app.state = object()  # no prediction_service attribute

    with pytest.raises(ModelNotLoadedError, match="Prediction service not initialized"):
        get_prediction_service(mock_request)


def test_predict_ham(client: TestClient, mock_prediction_service: Mock) -> None:
    mock_prediction_service.predict.return_value = {"prediction": "ham", "probability": 0.1}

    response = client.post("/api/v1/predict", json={"message": "Hello friend"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["prediction"] == "ham"
    assert data["probability"] == pytest.approx(0.1)
    assert data["confidence"] == "high"
    assert data["message"] == "Hello friend"


def test_predict_spam(client: TestClient, mock_prediction_service: Mock) -> None:
    mock_prediction_service.predict.return_value = {"prediction": "spam", "probability": 0.95}

    response = client.post("/api/v1/predict", json={"message": "Buy now!"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["prediction"] == "spam"
    assert response.json()["confidence"] == "high"


def test_predict_probability_none_defaults_to_half(client: TestClient, mock_prediction_service: Mock) -> None:
    mock_prediction_service.predict.return_value = {"prediction": "ham", "probability": None}

    response = client.post("/api/v1/predict", json={"message": "Hello"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["probability"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("body", "expected_status"),
    [
        ({"message": ""}, status.HTTP_422_UNPROCESSABLE_CONTENT),
        ({}, status.HTTP_422_UNPROCESSABLE_CONTENT),
        ({"message": "x" * 5001}, status.HTTP_422_UNPROCESSABLE_CONTENT),
    ],
)
def test_predict_validation_errors(client: TestClient, body: dict, expected_status: int) -> None:
    response = client.post("/api/v1/predict", json=body)
    assert response.status_code == expected_status


def test_health_ok(client: TestClient, mock_prediction_service: Mock) -> None:
    mock_prediction_service.is_loaded = True

    response = client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_uri"] == "models:/spam-classifier/1"
    assert "timestamp" in data


def test_health_model_not_loaded(client: TestClient, mock_prediction_service: Mock) -> None:
    mock_prediction_service.is_loaded = False

    response = client.get("/api/v1/health")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_model_info(client: TestClient, mock_prediction_service: Mock) -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == "spam-classifier"
    assert data["model_version"] == "1"
    assert data["model_type"] == ModelType.LOGISTIC_REGRESSION
    assert data["run_id"] == "abc123"


def test_load_model(client: TestClient, mock_prediction_service: Mock) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "best"})

    assert response.status_code == status.HTTP_200_OK
    mock_prediction_service.load_model.assert_called_once_with("spam-classifier", "best")


def test_load_model_default_strategy(client: TestClient, mock_prediction_service: Mock) -> None:
    response = client.post("/api/v1/model/load", json={})

    assert response.status_code == status.HTTP_200_OK
    mock_prediction_service.load_model.assert_called_once_with("spam-classifier", "latest")


def test_get_config(client: TestClient) -> None:
    response = client.get("/api/v1/config")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_type"] == ModelType.LOGISTIC_REGRESSION
    assert data["tfidf_max_features"] == 5000
    assert data["tfidf_ngram_range"] == [1, 2]
    assert data["tfidf_min_df"] == 2
    assert len(data["numerical_features"]) == 4
