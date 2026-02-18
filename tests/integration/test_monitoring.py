import httpx
import pytest
from fastapi import status


@pytest.mark.integration
def test_monitoring_stats_endpoint(client: httpx.Client) -> None:
    response = client.get("/api/v1/monitoring/stats")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "total_predictions" in data
    assert "spam_count" in data
    assert "ham_count" in data
    assert "model_version" in data


@pytest.mark.integration
def test_predictions_are_logged(client: httpx.Client) -> None:
    initial_count = client.get("/api/v1/monitoring/stats").json()["total_predictions"]

    for msg in ["Hello friend", "Buy now!", "Meeting tomorrow"]:
        client.post("/api/v1/predict", json={"message": msg})

    new_count = client.get("/api/v1/monitoring/stats").json()["total_predictions"]
    assert new_count >= initial_count + 3


@pytest.mark.integration
def test_drift_endpoint_insufficient_data(client: httpx.Client) -> None:
    response = client.get("/api/v1/monitoring/drift")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
