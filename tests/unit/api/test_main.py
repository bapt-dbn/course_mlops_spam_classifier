from contextlib import asynccontextmanager
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi import status
from fastapi.testclient import TestClient

from course_mlops.api.main import _is_fail_fast_enabled
from course_mlops.api.main import app


@asynccontextmanager
async def _noop_context(app: FastAPI):
    yield


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_is_fail_fast_enabled(monkeypatch: pytest.MonkeyPatch, env_value: str, expected: bool) -> None:
    monkeypatch.setenv("CML_FAIL_FAST", env_value)
    assert _is_fail_fast_enabled() is expected


def test_is_fail_fast_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CML_FAIL_FAST", raising=False)
    assert _is_fail_fast_enabled() is True


def test_lifespan_loads_model_successfully() -> None:
    with (
        patch("course_mlops.api.main.ModelLoader"),
        patch("course_mlops.api.main.PredictionService") as mock_svc_cls,
        patch("course_mlops.api.main._monitoring_db", _noop_context),
        patch("course_mlops.api.main._drift_detector", _noop_context),
    ):
        mock_svc = Mock()
        mock_svc_cls.return_value = mock_svc

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == status.HTTP_200_OK

        mock_svc.load_model.assert_called_once()


def test_lifespan_fail_fast_raises_on_load_failure() -> None:
    with (
        patch("course_mlops.api.main.ModelLoader"),
        patch("course_mlops.api.main.PredictionService") as mock_svc_cls,
        patch("course_mlops.api.main._is_fail_fast_enabled", return_value=True),
        patch("course_mlops.api.main._monitoring_db", _noop_context),
        patch("course_mlops.api.main._drift_detector", _noop_context),
    ):
        mock_svc = Mock()
        mock_svc.load_model.side_effect = Exception("MLflow unavailable")
        mock_svc_cls.return_value = mock_svc

        with pytest.raises(RuntimeError, match="Failed to load model"), TestClient(app):
            pass


def test_lifespan_continues_without_fail_fast() -> None:
    with (
        patch("course_mlops.api.main.ModelLoader"),
        patch("course_mlops.api.main.PredictionService") as mock_svc_cls,
        patch("course_mlops.api.main._is_fail_fast_enabled", return_value=False),
        patch("course_mlops.api.main._monitoring_db", _noop_context),
        patch("course_mlops.api.main._drift_detector", _noop_context),
    ):
        mock_svc = Mock()
        mock_svc.load_model.side_effect = Exception("MLflow unavailable")
        mock_svc_cls.return_value = mock_svc

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == status.HTTP_200_OK


def test_root_returns_service_info() -> None:
    with (
        patch("course_mlops.api.main.ModelLoader"),
        patch("course_mlops.api.main.PredictionService") as mock_svc_cls,
        patch("course_mlops.api.main._monitoring_db", _noop_context),
        patch("course_mlops.api.main._drift_detector", _noop_context),
    ):
        mock_svc_cls.return_value = Mock()

        with TestClient(app) as client:
            response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data
