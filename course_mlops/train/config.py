from functools import lru_cache
from pathlib import Path
from typing import Annotated
from typing import Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel
from pydantic import Field

from course_mlops.train.enums import ModelType
from course_mlops.train.enums import NumericalFeature
from course_mlops.train.exceptions import ConfigurationError
from course_mlops.train.models.logistic_regression import LogisticRegressionParams
from course_mlops.train.models.xgboost import XGBoostParams


class DataConfig(BaseModel):
    path: Annotated[str, Field(default="data/spam.csv", description="Path to dataset CSV")]
    test_size: Annotated[float, Field(default=0.2, ge=0, le=1, description="Test split ratio")]
    random_state: Annotated[int, Field(default=42, description="Random seed")]


class TfidfConfig(BaseModel):
    max_features: Annotated[int, Field(default=5000, gt=0, description="Max vocabulary size")]
    ngram_range: Annotated[tuple[int, int], Field(default=(1, 2), description="N-gram range")]
    min_df: Annotated[int, Field(default=2, ge=1, description="Min document frequency")]
    max_df: Annotated[float, Field(default=0.95, gt=0, le=1, description="Max document frequency")]
    stop_words: Annotated[str | None, Field(default="english", description="Stop words language")]


class FeaturesConfig(BaseModel):
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig, description="TF-IDF config")
    numerical: list[NumericalFeature] = Field(default=list(NumericalFeature), description="Numerical features")


class ModelConfig(BaseModel):
    type: Annotated[ModelType, Field(default=ModelType.LOGISTIC_REGRESSION, description="Model type")]
    params: LogisticRegressionParams | XGBoostParams = Field(
        default_factory=LogisticRegressionParams, description="Model hyperparameters"
    )


class LearningCurveConfig(BaseModel):
    cv: Annotated[int, Field(default=5, gt=1, description="Number of cross-validation folds")]
    n_jobs: Annotated[int, Field(default=-1, description="Number of parallel jobs (-1 for all cores)")]
    train_sizes_start: Annotated[float, Field(default=0.1, gt=0, le=1, description="Start of train sizes linspace")]
    train_sizes_end: Annotated[float, Field(default=1.0, gt=0, le=1, description="End of train sizes linspace")]
    train_sizes_steps: Annotated[int, Field(default=10, gt=1, description="Number of train size steps")]
    scoring: Annotated[str, Field(default="f1", description="Scoring metric")]


class MLFlowConfig(BaseModel):
    tracking_uri: Annotated[str, Field(default="mlruns", description="MLflow tracking URI")]
    experiment_name: Annotated[str, Field(default="spam-classifier", description="Experiment name")]
    registered_model_name: Annotated[str, Field(default="spam-classifier", description="Registered model name")]


class ConfidenceThresholds(BaseModel):
    spam: Annotated[float, Field(default=0.8, ge=0, le=1, description="Confident spam above this")]
    ham: Annotated[float, Field(default=0.2, ge=0, le=1, description="Confident ham below this")]


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    learning_curve: LearningCurveConfig = Field(default_factory=LearningCurveConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    confidence_thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)

    @classmethod
    def from_yaml(cls, path: str | Path = "config/config.yaml") -> Self:
        path = Path(path)
        if not path.exists():
            return cls()

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML syntax error in {path}: {e}") from e

        return cls(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()


settings = get_settings()
