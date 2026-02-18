import matplotlib

matplotlib.use("Agg")

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from course_mlops.train.config import DataConfig
from course_mlops.train.config import FeaturesConfig
from course_mlops.train.config import Settings
from course_mlops.train.config import TfidfConfig


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "Hey, how are you doing today?",
                "FREE MONEY!!! Click here now to claim $$$",
                "Can we meet tomorrow at 10am?",
                "WINNER! You have been selected for a prize!!!",
                "Please review the attached document",
                "Buy cheap meds online! Best prices $$$",
                "See you at the meeting later",
                "Congratulations! You won a free iPhone!!!",
            ],
        }
    )


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "hey how are you doing today",
        "free money click here now to claim",
        "can we meet tomorrow",
        "winner you have been selected for a prize",
        "please review the attached document",
        "buy cheap meds online best prices",
    ]


@pytest.fixture
def tiny_feature_matrix() -> csr_matrix:
    return csr_matrix(np.random.default_rng(42).random((6, 4)))


@pytest.fixture
def binary_labels() -> np.ndarray:
    return np.array([0, 1, 0, 1, 0, 1])


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(test_size=0.3, random_state=42)


@pytest.fixture
def features_config() -> FeaturesConfig:
    return FeaturesConfig(tfidf=TfidfConfig(min_df=1))


@pytest.fixture
def default_settings() -> Settings:
    return Settings()


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, Any, None]:
    with patch("course_mlops.train.pipeline.mlflow") as m:
        m.pyfunc = MagicMock()
        m.start_run.return_value.__enter__ = Mock(return_value=Mock(info=Mock(run_id="test-run-id")))
        m.start_run.return_value.__exit__ = Mock(return_value=False)
        yield m
