from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression

from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.exceptions import ModelTrainingError
from course_mlops.train.models.base import BaseClassifier
from course_mlops.train.models.logistic_regression.config import LogisticRegressionParams


class LogisticRegressionClassifier(BaseClassifier):
    params_class = LogisticRegressionParams

    def __init__(self, params: LogisticRegressionParams | None = None):
        self.params = params or self.params_class()
        self._model: LogisticRegression | None = None

    @property
    def name(self) -> str:
        return "logistic_regression"

    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        if X.shape[0] == 0:
            raise ModelTrainingError("Cannot train model with empty training data")

        self._model = LogisticRegression(
            C=self.params.C,
            max_iter=self.params.max_iter,
            solver=self.params.solver,
        )

        try:
            self._model.fit(X, y)
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {e}") from e

        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotFittedError
        return self._model.predict(X)

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotFittedError
        return self._model.predict_proba(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "C": self.params.C,
            "max_iter": self.params.max_iter,
            "solver": self.params.solver,
        }
