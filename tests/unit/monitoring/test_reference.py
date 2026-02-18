import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from course_mlops.monitoring.exceptions import ReferenceDataNotFoundError
from course_mlops.monitoring.reference import build_reference_dataset
from course_mlops.monitoring.reference import load_reference
from course_mlops.monitoring.reference import save_reference


def test_build_reference_dataset_shape() -> None:
    numerical = np.array([[10.0, 2.0, 0.1, 1.0], [20.0, 4.0, 0.2, 2.0]])
    predictions = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])

    df = build_reference_dataset(numerical, predictions, probabilities)

    assert len(df) == 2
    assert set(df.columns) == {
        "text_length",
        "word_count",
        "caps_ratio",
        "special_chars_count",
        "probability",
        "prediction_label",
    }


def test_build_reference_dataset_values() -> None:
    numerical = np.array([[10.0, 2.0, 0.1, 1.0]])
    predictions = np.array([1])
    probabilities = np.array([[0.2, 0.8]])

    df = build_reference_dataset(numerical, predictions, probabilities)

    assert df["text_length"].iloc[0] == pytest.approx(10.0)
    assert df["probability"].iloc[0] == pytest.approx(0.8)
    assert df["prediction_label"].iloc[0] == "spam"


def test_build_reference_dataset_ham_label() -> None:
    numerical = np.array([[10.0, 2.0, 0.1, 1.0]])
    predictions = np.array([0])
    probabilities = np.array([[0.9, 0.1]])

    df = build_reference_dataset(numerical, predictions, probabilities)

    assert df["prediction_label"].iloc[0] == "ham"


@patch("course_mlops.monitoring.reference.mlflow")
def test_save_reference(mock_mlflow: MagicMock) -> None:
    df = pd.DataFrame(
        {
            "text_length": [10.0],
            "word_count": [2.0],
            "caps_ratio": [0.1],
            "special_chars_count": [1.0],
            "probability": [0.8],
            "prediction_label": ["spam"],
        }
    )

    save_reference(df)

    mock_mlflow.log_artifact.assert_called_once()


def test_load_reference_raises_on_missing() -> None:
    with (
        patch("course_mlops.monitoring.reference.MlflowClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client.download_artifacts.side_effect = Exception("Not found")
        mock_client_cls.return_value = mock_client

        with pytest.raises(ReferenceDataNotFoundError, match="Failed to load reference"):
            load_reference("run-id", "http://localhost:5001")


@patch("course_mlops.monitoring.reference.MlflowClient")
def test_load_reference_success(mock_client_cls: MagicMock, tmp_path: Path) -> None:
    data = [{"text_length": 10.0, "word_count": 2.0, "caps_ratio": 0.1}]
    artifact_file = tmp_path / "monitoring" / "reference_dataset.json"
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_text(json.dumps(data))

    mock_client = MagicMock()
    mock_client.download_artifacts.return_value = str(artifact_file)
    mock_client_cls.return_value = mock_client

    df = load_reference("run-123", "http://localhost:5001")

    assert len(df) == 1
    assert df["text_length"].iloc[0] == pytest.approx(10.0)
