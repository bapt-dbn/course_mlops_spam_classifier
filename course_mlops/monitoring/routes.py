import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter
from fastapi import Depends
from fastapi import status

from course_mlops.api.dependencies import get_model_version
from course_mlops.api.dependencies import require_drift_detector
from course_mlops.api.dependencies import require_monitoring
from course_mlops.monitoring import dal
from course_mlops.monitoring.config import MonitoringConfig
from course_mlops.monitoring.drift import DriftDetector
from course_mlops.monitoring.exceptions import InsufficientDataError
from course_mlops.monitoring.schemas import DriftOutput
from course_mlops.monitoring.schemas import StatsOutput

logger = logging.getLogger(__name__)

monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@monitoring_router.get("/drift", response_model=DriftOutput, status_code=status.HTTP_200_OK)
async def drift(
    monitoring_config: Annotated[MonitoringConfig, Depends(require_monitoring)],
    drift_detector: Annotated[DriftDetector, Depends(require_drift_detector)],
    model_version: Annotated[str, Depends(get_model_version)],
) -> DriftOutput:
    records = await dal.get_recent_predictions(model_version, limit=monitoring_config.drift_window_size)

    if len(records) < monitoring_config.drift_min_samples:
        raise InsufficientDataError(
            f"Need at least {monitoring_config.drift_min_samples} predictions for drift analysis, got {len(records)}"
        )

    current_df = pd.DataFrame([r.model_dump() for r in records])
    current_df["prediction_label"] = current_df["prediction"]

    result = drift_detector.detect(current_df)

    logger.info(
        "Drift analysis completed",
        extra={
            "x-dataset_drift": result.dataset_drift,
            "x-drift_share": result.drift_share,
            "x-n_samples": len(records),
        },
    )

    return DriftOutput(
        dataset_drift=result.dataset_drift,
        drift_share=result.drift_share,
        column_drifts=result.column_drifts,
        n_reference_samples=len(drift_detector._reference_df),
        n_current_samples=len(records),
        model_version=model_version,
    )


@monitoring_router.get(
    "/stats",
    response_model=StatsOutput,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_monitoring)],
)
async def stats(
    model_version: Annotated[str, Depends(get_model_version)],
) -> StatsOutput:
    stats = await dal.get_stats(model_version)

    return StatsOutput(
        total_predictions=stats.total_predictions,
        spam_count=stats.spam_count,
        ham_count=stats.ham_count,
        spam_ratio=stats.spam_count / stats.total_predictions if stats.total_predictions > 0 else 0.0,
        avg_probability=stats.avg_probability,
        first_prediction=stats.first_prediction,
        last_prediction=stats.last_prediction,
        model_version=model_version,
    )
