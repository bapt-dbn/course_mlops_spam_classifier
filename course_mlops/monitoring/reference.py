import json
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from course_mlops.monitoring.enums import MonitoredColumn
from course_mlops.monitoring.exceptions import ReferenceDataNotFoundError

logger = logging.getLogger(__name__)

_ARTIFACT_PATH = "monitoring/reference_dataset.json"


def build_reference_dataset(
    numerical_features: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    prediction_labels = ["spam" if p == 1 else "ham" for p in predictions]

    return pd.DataFrame(
        {
            MonitoredColumn.TEXT_LENGTH: numerical_features[:, 0],
            MonitoredColumn.WORD_COUNT: numerical_features[:, 1],
            MonitoredColumn.CAPS_RATIO: numerical_features[:, 2],
            MonitoredColumn.SPECIAL_CHARS_COUNT: numerical_features[:, 3],
            MonitoredColumn.PROBABILITY: probabilities[:, 1] if probabilities.ndim > 1 else probabilities,
            MonitoredColumn.PREDICTION_LABEL: prediction_labels,
        }
    )


def save_reference(reference_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "reference_dataset.json"
        reference_df.to_json(file_path, orient="records", indent=2)
        mlflow.log_artifact(str(file_path), artifact_path="monitoring")
    logger.info("Reference dataset saved to MLflow")


def load_reference(run_id: str, tracking_uri: str) -> pd.DataFrame:
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, _ARTIFACT_PATH, tmpdir)
            with open(local_path) as f:
                data = json.load(f)
            return pd.DataFrame(data)
    except Exception as e:
        raise ReferenceDataNotFoundError(f"Failed to load reference dataset for run {run_id}: {e}") from e
