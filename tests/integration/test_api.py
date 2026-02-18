import httpx
import pytest
from fastapi import status
from mlflow.tracking import MlflowClient

from .conftest import MODEL_NAME


@pytest.mark.integration
def test_root_returns_service_info(client: httpx.Client) -> None:
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"


@pytest.mark.integration
def test_health_with_model_loaded(client: httpx.Client) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_uri"] is not None
    assert "timestamp" in data


@pytest.mark.integration
def test_predict_ham_message(client: httpx.Client) -> None:
    response = client.post("/api/v1/predict", json={"message": "Hey, how are you doing today?"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["prediction"] == "ham"
    assert 0 <= data["probability"] <= 1
    assert data["confidence"] in ("high", "medium", "low")
    assert data["message"] == "Hey, how are you doing today?"


@pytest.mark.integration
def test_predict_spam_message(client: httpx.Client) -> None:
    response = client.post(
        "/api/v1/predict",
        json={"message": "FREE MONEY!!! Click here NOW to claim your $1000 prize!!!"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["prediction"] in ("spam", "ham")
    assert 0 <= data["probability"] <= 1
    assert data["confidence"] in ("high", "medium", "low")


@pytest.mark.integration
def test_predict_empty_message_returns_422(client: httpx.Client) -> None:
    response = client.post("/api/v1/predict", json={"message": ""})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.integration
def test_predict_missing_message_returns_422(client: httpx.Client) -> None:
    response = client.post("/api/v1/predict", json={})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.integration
def test_predict_too_long_message_returns_422(client: httpx.Client) -> None:
    response = client.post("/api/v1/predict", json={"message": "x" * 5001})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.integration
def test_predict_returns_original_message(client: httpx.Client) -> None:
    message = "Can we meet tomorrow at 10am?"
    response = client.post("/api/v1/predict", json={"message": message})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == message


@pytest.mark.integration
def test_model_info_returns_metadata(client: httpx.Client) -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert data["model_version"] is not None
    assert data["run_id"] is not None
    assert "mlflow_ui_url" in data
    assert data["artifact_uri"] is not None


@pytest.mark.integration
def test_model_load_latest_strategy(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "latest"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert data["strategy"] == "latest"


@pytest.mark.integration
def test_model_load_default_strategy(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["strategy"] == "latest"


@pytest.mark.integration
def test_model_load_best_strategy(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "best"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert data["strategy"] == "best"
    assert data["model_version"] is not None


@pytest.mark.integration
def test_model_load_version_strategy(client: httpx.Client, mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    version_number = versions[0].version

    response = client.post("/api/v1/model/load", json={"strategy": f"version:{version_number}"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_version"] == str(version_number)
    assert data["strategy"] == f"version:{version_number}"


@pytest.mark.integration
def test_model_load_type_strategy(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "type:logistic_regression"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert data["strategy"] == "type:logistic_regression"
    assert data["model_type"] == "logistic_regression"


@pytest.mark.integration
def test_api_works_after_model_reload(client: httpx.Client) -> None:
    client.post("/api/v1/model/load", json={"strategy": "latest"})

    response = client.post("/api/v1/predict", json={"message": "Hello friend"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["prediction"] in ("spam", "ham")


@pytest.mark.integration
def test_config_returns_training_config(client: httpx.Client) -> None:
    response = client.get("/api/v1/config")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "model_type" in data
    assert "tfidf_max_features" in data
    assert "tfidf_ngram_range" in data
    assert "tfidf_min_df" in data
    assert len(data["numerical_features"]) == 4


@pytest.mark.integration
def test_model_load_nonexistent_version_returns_500(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "version:99999"})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "error_code" in response.json()


@pytest.mark.integration
def test_model_load_nonexistent_type_returns_500(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "type:random_forest"})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "error_code" in response.json()


@pytest.mark.integration
def test_model_load_invalid_strategy_returns_500(client: httpx.Client) -> None:
    response = client.post("/api/v1/model/load", json={"strategy": "garbage"})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "error_code" in response.json()
