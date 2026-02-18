import logging
from datetime import UTC
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import status

from course_mlops.api.config import MLFLOW_UI_BASE
from course_mlops.api.config import MODEL_NAME
from course_mlops.api.dependencies import get_monitoring_config
from course_mlops.api.dependencies import get_prediction_service
from course_mlops.api.enums import HealthStatusEnum
from course_mlops.api.enums import PredictionEnum
from course_mlops.api.schemas import ConfigOutput
from course_mlops.api.schemas import HealthOutput
from course_mlops.api.schemas import ModelInfoOutput
from course_mlops.api.schemas import ModelLoadInput
from course_mlops.api.schemas import PredictInput
from course_mlops.api.schemas import PredictOutput
from course_mlops.api.service import PredictionService
from course_mlops.api.service import calculate_confidence
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.monitoring.config import MonitoringConfig
from course_mlops.monitoring.service import log_prediction_to_store
from course_mlops.train.config import get_settings
from course_mlops.train.enums import NumericalFeature

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictOutput, status_code=status.HTTP_200_OK)
async def predict(
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
    request: PredictInput,
    monitoring_config: Annotated[MonitoringConfig | None, Depends(get_monitoring_config)],
) -> PredictOutput:
    settings = get_settings()
    thresholds = settings.confidence_thresholds
    result = prediction_service.predict(request.message)

    probability = result["probability"] if result["probability"] is not None else 0.5
    confidence = calculate_confidence(probability, spam=thresholds.spam, ham=thresholds.ham)

    if monitoring_config and monitoring_config.enabled:
        await log_prediction_to_store(
            prediction_service.model_version,
            request.message,
            result["prediction"],
            probability,
            confidence.value,
        )

    return PredictOutput(
        message=request.message,
        prediction=PredictionEnum(result["prediction"]),
        probability=probability,
        confidence=confidence,
    )


@router.get("/health", response_model=HealthOutput, status_code=status.HTTP_200_OK)
async def health(
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> HealthOutput:
    if not prediction_service.is_loaded:
        raise ModelNotLoadedError("Model not loaded")

    return HealthOutput(
        status=HealthStatusEnum.HEALTHY,
        model_loaded=True,
        model_uri=prediction_service.model_uri,
        timestamp=datetime.now(UTC),
    )


@router.get("/model/info", response_model=ModelInfoOutput, status_code=status.HTTP_200_OK)
async def model_info(
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> ModelInfoOutput:
    return ModelInfoOutput.from_service(prediction_service, MODEL_NAME, MLFLOW_UI_BASE)


@router.post("/model/load", response_model=ModelInfoOutput, status_code=status.HTTP_200_OK)
async def load_model(
    body: ModelLoadInput,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> ModelInfoOutput:
    """Load a model with the specified strategy.

    Strategies:
    - `latest`: Most recent version
    - `best`: Best F1 score
    - `version:N`: Specific version (e.g., version:3)
    - `type:X`: Latest of a type (e.g., type:xgboost)
    """
    prediction_service.load_model(MODEL_NAME, body.strategy)
    return ModelInfoOutput.from_service(prediction_service, MODEL_NAME, MLFLOW_UI_BASE)


@router.get("/config", response_model=ConfigOutput, status_code=status.HTTP_200_OK)
async def get_config() -> ConfigOutput:
    settings = get_settings()

    return ConfigOutput(
        model_type=settings.model.type,
        tfidf_max_features=settings.features.tfidf.max_features,
        tfidf_ngram_range=settings.features.tfidf.ngram_range,
        tfidf_min_df=settings.features.tfidf.min_df,
        numerical_features=list(NumericalFeature),
    )
