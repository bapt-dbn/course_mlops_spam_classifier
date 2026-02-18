from typing import Annotated

from fastapi import Depends
from fastapi import Request

from course_mlops.api.service import PredictionService
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.monitoring.config import MonitoringConfig
from course_mlops.monitoring.drift import DriftDetector
from course_mlops.monitoring.exceptions import MonitoringDatabaseError
from course_mlops.monitoring.exceptions import ReferenceDataNotFoundError


def get_prediction_service(request: Request) -> PredictionService:
    service: PredictionService | None = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise ModelNotLoadedError("Prediction service not initialized")
    return service


def get_model_version(
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> str:
    return getattr(prediction_service, "model_version", None) or "unknown"


def get_monitoring_config(request: Request) -> MonitoringConfig | None:
    return getattr(request.app.state, "monitoring_config", None)


def require_monitoring(request: Request) -> MonitoringConfig:
    config: MonitoringConfig | None = getattr(request.app.state, "monitoring_config", None)
    if not config or not config.enabled:
        raise MonitoringDatabaseError("Monitoring is not enabled")
    return config


def get_drift_detector(request: Request) -> DriftDetector | None:
    return getattr(request.app.state, "drift_detector", None)


def require_drift_detector(request: Request) -> DriftDetector:
    detector: DriftDetector | None = getattr(request.app.state, "drift_detector", None)
    if detector is None:
        raise ReferenceDataNotFoundError("Drift detector not available (reference dataset missing)")
    return detector
