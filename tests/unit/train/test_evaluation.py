import numpy as np
import pytest

from course_mlops.train.exceptions import EvaluationError
from course_mlops.train.reporting.evaluation import EvaluationMetrics
from course_mlops.train.reporting.evaluation import evaluate_model


def test_to_dict_keys() -> None:
    metrics = EvaluationMetrics(accuracy=0.9, precision=0.85, recall=0.8, f1=0.82, roc_auc=0.95)
    assert set(metrics.to_dict()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    metrics = evaluate_model(y, y)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_with_1d_proba() -> None:
    y = np.array([0, 1, 0, 1])
    metrics = evaluate_model(y, y, y_proba=np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics.roc_auc > 0


def test_with_2d_proba() -> None:
    y = np.array([0, 1, 0, 1])
    proba_2d = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = evaluate_model(y, y, y_proba=proba_2d)
    assert metrics.roc_auc > 0


def test_without_proba_roc_auc_is_zero() -> None:
    y = np.array([0, 1, 0, 1])
    assert evaluate_model(y, y).roc_auc == pytest.approx(0.0)


def test_empty_raises() -> None:
    with pytest.raises(EvaluationError):
        evaluate_model(np.array([]), np.array([]))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(EvaluationError):
        evaluate_model(np.array([0, 1]), np.array([0, 1, 0]))
