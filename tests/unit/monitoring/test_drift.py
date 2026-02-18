import numpy as np
import pandas as pd
import pytest

from course_mlops.monitoring.drift import DriftDetector
from course_mlops.monitoring.exceptions import DriftDetectionError
from course_mlops.monitoring.schemas import DriftResult


def test_detect_no_drift(detector: DriftDetector, reference_df: pd.DataFrame) -> None:
    # Current data similar to reference -> no drift
    result = detector.detect(reference_df)

    assert isinstance(result, DriftResult)
    assert isinstance(result.dataset_drift, bool)
    assert isinstance(result.drift_share, float)
    assert "text_length" in result.column_drifts


def test_detect_with_drift(detector: DriftDetector) -> None:
    # Current data very different -> drift
    rng = np.random.default_rng(99)
    n = 100
    current_df = pd.DataFrame(
        {
            "text_length": rng.normal(200, 10, n),  # very different
            "word_count": rng.normal(50, 3, n),  # very different
            "caps_ratio": rng.uniform(0.8, 1.0, n),  # very different
            "special_chars_count": rng.poisson(20, n).astype(float),  # very different
            "probability": rng.uniform(0.9, 1.0, n),  # very different
            "prediction_label": ["spam"] * n,  # all spam
        }
    )

    result = detector.detect(current_df)

    assert result.dataset_drift is True
    assert result.drift_share > 0


def test_detect_column_drifts_structure(detector: DriftDetector, reference_df: pd.DataFrame) -> None:
    result = detector.detect(reference_df)

    for col_data in result.column_drifts.values():
        assert isinstance(col_data.drift_detected, bool)
        assert isinstance(col_data.drift_score, float)
        assert isinstance(col_data.stattest_name, str)


def test_detect_raises_on_invalid_data(detector: DriftDetector) -> None:
    with pytest.raises(DriftDetectionError):
        detector.detect(pd.DataFrame())
