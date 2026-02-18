import numpy as np
import pandas as pd
import pytest

from course_mlops.monitoring.drift import DriftDetector


@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "text_length": rng.normal(50, 10, n),
            "word_count": rng.normal(10, 3, n),
            "caps_ratio": rng.uniform(0, 0.3, n),
            "special_chars_count": rng.poisson(2, n).astype(float),
            "probability": rng.uniform(0, 1, n),
            "prediction_label": rng.choice(["spam", "ham"], n),
        }
    )


@pytest.fixture
def detector(reference_df: pd.DataFrame) -> DriftDetector:
    return DriftDetector(reference_df)
