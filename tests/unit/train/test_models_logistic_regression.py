import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from course_mlops.train.enums import ModelType
from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.exceptions import ModelTrainingError
from course_mlops.train.models.logistic_regression import LogisticRegressionClassifier
from course_mlops.train.models.logistic_regression import LogisticRegressionParams


def test_params_defaults() -> None:
    params = LogisticRegressionParams()
    assert pytest.approx(0.5) == params.C
    assert params.max_iter == 1000
    assert params.solver == "liblinear"


def test_name() -> None:
    assert LogisticRegressionClassifier().name == ModelType.LOGISTIC_REGRESSION


def test_get_params_keys() -> None:
    assert set(LogisticRegressionClassifier().get_params()) == {"C", "max_iter", "solver"}


def test_fit_empty_data_raises(tiny_feature_matrix: csr_matrix) -> None:
    with pytest.raises(ModelTrainingError):
        LogisticRegressionClassifier().fit(tiny_feature_matrix[:0], [])


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_before_fit_raises(tiny_feature_matrix: csr_matrix, method: str) -> None:
    with pytest.raises(ModelNotFittedError):
        getattr(LogisticRegressionClassifier(), method)(tiny_feature_matrix)


def test_fit_returns_self(tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray) -> None:
    clf = LogisticRegressionClassifier()
    assert clf.fit(tiny_feature_matrix, binary_labels) is clf


def test_predict_shape(tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray) -> None:
    clf = LogisticRegressionClassifier()
    clf.fit(tiny_feature_matrix, binary_labels)
    assert clf.predict(tiny_feature_matrix).shape == (6,)


def test_predict_proba_shape(tiny_feature_matrix: csr_matrix, binary_labels: np.ndarray) -> None:
    clf = LogisticRegressionClassifier()
    clf.fit(tiny_feature_matrix, binary_labels)
    assert clf.predict_proba(tiny_feature_matrix).shape == (6, 2)


def test_to_sklearn_returns_logistic_regression() -> None:
    params = LogisticRegressionParams(C=1.0, max_iter=500, solver="lbfgs")
    estimator = params.to_sklearn()
    assert isinstance(estimator, LogisticRegression)
    assert pytest.approx(1.0) == estimator.C
    assert estimator.max_iter == 500
    assert estimator.solver == "lbfgs"
