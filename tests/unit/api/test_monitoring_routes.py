from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi import status
from fastapi.testclient import TestClient

from course_mlops.api.dependencies import get_model_version
from course_mlops.api.dependencies import require_drift_detector
from course_mlops.api.dependencies import require_monitoring
from course_mlops.api.exception_handlers import course_mlops_exception_handler
from course_mlops.api.exception_handlers import general_exception_handler
from course_mlops.exceptions import CourseMLOpsError
from course_mlops.monitoring.routes import monitoring_router
from course_mlops.monitoring.schemas import ColumnDrift
from course_mlops.monitoring.schemas import DriftResult
from course_mlops.monitoring.schemas import PredictionStats
from course_mlops.monitoring.schemas import RecentPrediction


@pytest.fixture
def mock_drift_detector() -> Mock:
    detector = Mock()
    detector._reference_df = Mock()
    detector._reference_df.__len__ = Mock(return_value=100)
    return detector


@pytest.fixture
def mock_dal() -> Generator[AsyncMock, None, None]:
    with patch("course_mlops.monitoring.routes.dal") as mock:
        mock.get_recent_predictions = AsyncMock()
        mock.get_stats = AsyncMock()
        yield mock


@pytest.fixture
def monitoring_client(mock_drift_detector: Mock, mock_dal: AsyncMock) -> Generator[TestClient, None, None]:
    app = FastAPI()

    app.include_router(monitoring_router)
    app.add_exception_handler(CourseMLOpsError, course_mlops_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    app.dependency_overrides[require_monitoring] = lambda: Mock(
        drift_window_size=500, drift_min_samples=50, enabled=True
    )
    app.dependency_overrides[require_drift_detector] = lambda: mock_drift_detector
    app.dependency_overrides[get_model_version] = lambda: "1"
    with TestClient(app) as c:
        yield c


def test_stats_returns_ok(monitoring_client: TestClient, mock_dal: AsyncMock) -> None:
    mock_dal.get_stats.return_value = PredictionStats(
        total_predictions=100,
        spam_count=30,
        ham_count=70,
        avg_probability=0.3,
        first_prediction="2024-01-01T00:00:00",
        last_prediction="2024-01-02T00:00:00",
    )

    response = monitoring_client.get("/api/v1/monitoring/stats")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_predictions"] == 100
    assert data["spam_ratio"] == pytest.approx(0.3)
    assert data["model_version"] == "1"


def test_stats_zero_predictions(monitoring_client: TestClient, mock_dal: AsyncMock) -> None:
    mock_dal.get_stats.return_value = PredictionStats(
        total_predictions=0,
        spam_count=0,
        ham_count=0,
        avg_probability=None,
        first_prediction=None,
        last_prediction=None,
    )

    response = monitoring_client.get("/api/v1/monitoring/stats")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["spam_ratio"] == pytest.approx(0.0)


def test_drift_returns_ok(
    monitoring_client: TestClient,
    mock_dal: AsyncMock,
    mock_drift_detector: Mock,
) -> None:
    prediction = RecentPrediction(
        text_length=10.0, word_count=2.0, caps_ratio=0.0, special_chars_count=0.0, prediction="ham", probability=0.1
    )
    mock_dal.get_recent_predictions.return_value = [prediction] * 60
    mock_drift_detector.detect.return_value = DriftResult(
        dataset_drift=False,
        drift_share=0.0,
        column_drifts={"text_length": ColumnDrift(drift_detected=False, drift_score=0.1, stattest_name="ks")},
    )

    response = monitoring_client.get("/api/v1/monitoring/drift")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["dataset_drift"] is False
    assert data["model_version"] == "1"


def test_drift_insufficient_data(monitoring_client: TestClient, mock_dal: AsyncMock) -> None:
    prediction = RecentPrediction(
        text_length=10.0, word_count=2.0, caps_ratio=0.0, special_chars_count=0.0, prediction="ham", probability=0.1
    )
    mock_dal.get_recent_predictions.return_value = [prediction] * 10

    response = monitoring_client.get("/api/v1/monitoring/drift")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
