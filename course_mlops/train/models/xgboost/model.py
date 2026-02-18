from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from xgboost import XGBClassifier as XGBClassifierBase

from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.exceptions import ModelTrainingError
from course_mlops.train.models.base import BaseClassifier
from course_mlops.train.models.xgboost.config import XGBoostParams


class XGBoostClassifier(BaseClassifier):
    params_class = XGBoostParams

    def __init__(self, params: XGBoostParams | None = None):
        self.params = params or self.params_class()
        self._model: XGBClassifierBase | None = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        if X.shape[0] == 0:
            raise ModelTrainingError("Cannot train model with empty training data")

        self._model = XGBClassifierBase(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            learning_rate=self.params.learning_rate,
            subsample=self.params.subsample,
            colsample_bytree=self.params.colsample_bytree,
            random_state=self.params.random_state,
            eval_metric="logloss",
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
            "n_estimators": self.params.n_estimators,
            "max_depth": self.params.max_depth,
            "learning_rate": self.params.learning_rate,
            "subsample": self.params.subsample,
            "colsample_bytree": self.params.colsample_bytree,
            "random_state": self.params.random_state,
        }
