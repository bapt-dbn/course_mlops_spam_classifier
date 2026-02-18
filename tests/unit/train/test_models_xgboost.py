import numpy as np
import pytest
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier

from course_mlops.train.enums import ModelType
from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.exceptions import ModelTrainingError
from course_mlops.train.models.xgboost import XGBoostClassifier
from course_mlops.train.models.xgboost import XGBoostParams


@pytest.fixture
def fast_params() -> XGBoostParams:
    return XGBoostParams(n_estimators=2, max_depth=2)


def test_params_defaults() -> None:
    params = XGBoostParams()
    assert params.n_estimators == 100
    assert params.max_depth == 6
    assert params.learning_rate == pytest.approx(0.1)
    assert params.subsample == pytest.approx(0.8)
    assert params.colsample_bytree == pytest.approx(0.8)
    assert params.random_state == 42


def test_name() -> None:
    assert XGBoostClassifier().name == ModelType.XGBOOST


def test_get_params_keys() -> None:
    assert set(XGBoostClassifier().get_params()) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "random_state",
    }


def test_fit_empty_data_raises(tiny_feature_matrix: csr_matrix, fast_params: XGBoostParams) -> None:
    with pytest.raises(ModelTrainingError):
        XGBoostClassifier(params=fast_params).fit(tiny_feature_matrix[:0], [])


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_before_fit_raises(tiny_feature_matrix: csr_matrix, method: str) -> None:
    with pytest.raises(ModelNotFittedError):
        getattr(XGBoostClassifier(), method)(tiny_feature_matrix)


def test_fit_returns_self(
    tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray, fast_params: XGBoostParams
) -> None:
    clf = XGBoostClassifier(params=fast_params)
    assert clf.fit(tiny_feature_matrix, binary_labels) is clf


def test_predict_shape(tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray, fast_params: XGBoostParams) -> None:
    clf = XGBoostClassifier(params=fast_params)
    clf.fit(tiny_feature_matrix, binary_labels)
    assert clf.predict(tiny_feature_matrix).shape == (6,)


def test_predict_proba_shape(
    tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray, fast_params: XGBoostParams
) -> None:
    clf = XGBoostClassifier(params=fast_params)
    clf.fit(tiny_feature_matrix, binary_labels)
    assert clf.predict_proba(tiny_feature_matrix).shape == (6, 2)


def test_to_sklearn_returns_xgb_classifier() -> None:
    params = XGBoostParams(n_estimators=50, max_depth=3)
    estimator = params.to_sklearn()
    assert isinstance(estimator, XGBClassifier)
    assert estimator.n_estimators == 50
    assert estimator.max_depth == 3
