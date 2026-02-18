from datetime import UTC
from datetime import datetime
from typing import Annotated
from typing import Self

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import Field

from course_mlops.api.enums import ConfidenceEnum
from course_mlops.api.enums import HealthStatusEnum
from course_mlops.api.enums import ModelStrategy
from course_mlops.api.enums import PredictionEnum
from course_mlops.api.service import PredictionService
from course_mlops.train.enums import ModelType
from course_mlops.train.enums import NumericalFeature


class PredictInput(BaseModel):
    message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=5000,
            description="Message text",
        ),
    ]


class PredictOutput(BaseModel):
    message: Annotated[str, Field(description="Original message text")]
    prediction: Annotated[PredictionEnum, Field(description="Spam or ham")]
    probability: Annotated[float, Field(ge=0, le=1, description="Spam probability")]
    confidence: Annotated[ConfidenceEnum, Field(description="Confidence level")]


class HealthOutput(BaseModel):
    status: Annotated[HealthStatusEnum, Field(description="Service health status")]
    model_loaded: Annotated[bool, Field(description="Whether a model is loaded")]
    model_uri: Annotated[str | None, Field(default=None, description="Loaded model URI")]
    timestamp: Annotated[AwareDatetime, Field(description="Health check timestamp")]


class ModelLoadInput(BaseModel):
    strategy: Annotated[
        str,
        Field(
            description="Strategy: 'latest', 'best', 'version:N', 'type:X'",
            examples=["latest", "best", "version:3", "type:xgboost"],
        ),
    ] = ModelStrategy.LATEST


class ModelInfoOutput(BaseModel):
    model_name: Annotated[str, Field(description="Model name")]
    model_version: Annotated[str | None, Field(default=None, description="Model version")]
    model_type: Annotated[str | None, Field(default=None, description="Model type (logistic_regression, xgboost)")]
    strategy: Annotated[str | None, Field(default=None, description="Strategy used to load model")]
    run_id: Annotated[str | None, Field(default=None, description="MLFlow run ID")]
    mlflow_ui_url: Annotated[str, Field(description="MLFlow UI URL")]
    artifact_uri: Annotated[str | None, Field(default=None, description="Model artifact URI")]
    registered_at: Annotated[str | None, Field(default=None, description="Registration timestamp (ISO 8601)")]

    @classmethod
    def from_service(
        cls,
        prediction_service: PredictionService,
        model_name: str,
        mlflow_ui_base: str,
    ) -> Self:
        return cls(
            model_name=model_name,
            model_version=prediction_service.model_version,
            model_type=prediction_service.model_type,
            strategy=prediction_service.strategy,
            run_id=prediction_service.run_id,
            mlflow_ui_url=f"{mlflow_ui_base}/#/runs/{prediction_service.run_id}"
            if prediction_service.run_id
            else mlflow_ui_base,
            artifact_uri=prediction_service.artifact_uri,
            registered_at=prediction_service.registered_at,
        )


class ConfigOutput(BaseModel):
    model_type: Annotated[ModelType, Field(description="Model type")]
    tfidf_max_features: Annotated[int, Field(description="TF-IDF max features")]
    tfidf_ngram_range: Annotated[tuple[int, int], Field(description="TF-IDF n-gram range")]
    tfidf_min_df: Annotated[int, Field(description="TF-IDF minimum document frequency")]
    numerical_features: Annotated[list[NumericalFeature], Field(description="Numerical features")]


class ErrorOutput(BaseModel):
    detail: Annotated[str, Field(description="Error detail message")]
    error_code: Annotated[str | None, Field(default=None, description="Error code")]
    timestamp: Annotated[AwareDatetime, Field(default_factory=lambda: datetime.now(UTC), description="Error timestamp")]
