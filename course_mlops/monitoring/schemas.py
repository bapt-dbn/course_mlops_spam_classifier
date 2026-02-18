from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class PredictionRecord(BaseModel):
    model_version: Annotated[str, Field(description="Model version used for prediction")]
    message_hash: Annotated[str, Field(description="SHA-256 hash of the original message")]
    text_length: Annotated[float, Field(description="Length of preprocessed text")]
    word_count: Annotated[float, Field(description="Number of words in preprocessed text")]
    caps_ratio: Annotated[float, Field(description="Ratio of uppercase characters")]
    special_chars_count: Annotated[float, Field(description="Count of special characters")]
    prediction: Annotated[str, Field(description="Predicted label")]
    probability: Annotated[float, Field(description="Prediction probability")]
    confidence: Annotated[str, Field(description="Confidence level")]


class ColumnDrift(BaseModel):
    drift_detected: Annotated[bool, Field(description="Whether drift was detected for this column")]
    drift_score: Annotated[float, Field(description="Statistical test p-value")]
    stattest_name: Annotated[str, Field(description="Name of the statistical test used")]


class DriftResult(BaseModel):
    dataset_drift: Annotated[bool, Field(description="Whether dataset-level drift was detected")]
    drift_share: Annotated[float, Field(description="Share of drifted columns")]
    column_drifts: Annotated[dict[str, ColumnDrift], Field(description="Per-column drift details")]


class DriftOutput(BaseModel):
    dataset_drift: Annotated[bool, Field(description="Whether dataset-level drift was detected")]
    drift_share: Annotated[float, Field(description="Share of drifted columns")]
    column_drifts: Annotated[dict[str, ColumnDrift], Field(description="Per-column drift details")]
    n_reference_samples: Annotated[int, Field(description="Number of reference samples")]
    n_current_samples: Annotated[int, Field(description="Number of current samples")]
    model_version: Annotated[str, Field(description="Model version analyzed")]


class RecentPrediction(BaseModel):
    text_length: Annotated[float, Field(description="Length of preprocessed text")]
    word_count: Annotated[float, Field(description="Number of words in preprocessed text")]
    caps_ratio: Annotated[float, Field(description="Ratio of uppercase characters")]
    special_chars_count: Annotated[float, Field(description="Count of special characters")]
    prediction: Annotated[str, Field(description="Predicted label")]
    probability: Annotated[float, Field(description="Prediction probability")]


class PredictionStats(BaseModel):
    total_predictions: Annotated[int, Field(description="Total number of predictions")]
    spam_count: Annotated[int, Field(description="Number of spam predictions")]
    ham_count: Annotated[int, Field(description="Number of ham predictions")]
    avg_probability: Annotated[float | None, Field(description="Average spam probability")]
    first_prediction: Annotated[Any, Field(description="Timestamp of first prediction")]
    last_prediction: Annotated[Any, Field(description="Timestamp of last prediction")]


class StatsOutput(BaseModel):
    total_predictions: Annotated[int, Field(description="Total number of predictions")]
    spam_count: Annotated[int, Field(description="Number of spam predictions")]
    ham_count: Annotated[int, Field(description="Number of ham predictions")]
    spam_ratio: Annotated[float, Field(description="Ratio of spam predictions")]
    avg_probability: Annotated[float | None, Field(description="Average spam probability")]
    first_prediction: Annotated[Any, Field(description="Timestamp of first prediction")]
    last_prediction: Annotated[Any, Field(description="Timestamp of last prediction")]
    model_version: Annotated[str, Field(description="Model version")]
