import pytest

from course_mlops.monitoring.schemas import ColumnDrift
from course_mlops.monitoring.schemas import DriftOutput
from course_mlops.monitoring.schemas import StatsOutput


def test_drift_output_valid() -> None:
    output = DriftOutput(
        dataset_drift=False,
        drift_share=0.0,
        column_drifts={
            "text_length": ColumnDrift(drift_detected=False, drift_score=0.1, stattest_name="ks"),
        },
        n_reference_samples=100,
        n_current_samples=50,
        model_version="1",
    )
    assert output.dataset_drift is False
    assert output.drift_share == pytest.approx(0.0)
    assert output.model_version == "1"


def test_stats_output_valid() -> None:
    output = StatsOutput(
        total_predictions=100,
        spam_count=30,
        ham_count=70,
        spam_ratio=0.3,
        avg_probability=0.3,
        first_prediction="2024-01-01T00:00:00",
        last_prediction="2024-01-02T00:00:00",
        model_version="1",
    )
    assert output.total_predictions == 100
    assert output.spam_ratio == pytest.approx(0.3)


def test_stats_output_with_none_avg_probability() -> None:
    output = StatsOutput(
        total_predictions=0,
        spam_count=0,
        ham_count=0,
        spam_ratio=0.0,
        avg_probability=None,
        first_prediction=None,
        last_prediction=None,
        model_version="1",
    )
    assert output.avg_probability is None


def test_column_drift_valid() -> None:
    cd = ColumnDrift(drift_detected=True, drift_score=0.95, stattest_name="ks")
    assert cd.drift_detected is True
    assert cd.drift_score == pytest.approx(0.95)
