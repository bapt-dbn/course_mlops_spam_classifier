import pytest

from course_mlops.train.enums import ModelType
from course_mlops.train.models.base import BaseClassifier
from course_mlops.train.models.factory import get_classifier
from course_mlops.train.models.logistic_regression import LogisticRegressionClassifier
from course_mlops.train.models.xgboost import XGBoostClassifier


@pytest.mark.parametrize(
    ("model_type", "expected_class"),
    [
        (ModelType.LOGISTIC_REGRESSION, LogisticRegressionClassifier),
        (ModelType.XGBOOST, XGBoostClassifier),
    ],
)
def test_returns_correct_class(model_type: ModelType, expected_class: type[BaseClassifier]) -> None:
    assert get_classifier(model_type) is expected_class


def test_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="is not a valid ModelType"):
        get_classifier(ModelType("invalid_model"))
