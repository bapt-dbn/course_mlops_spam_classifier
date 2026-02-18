from typing import Annotated

from pydantic import BaseModel
from pydantic import Field
from xgboost import XGBClassifier


class XGBoostParams(BaseModel):
    n_estimators: Annotated[int, Field(default=100, gt=0, description="Number of boosting rounds")]
    max_depth: Annotated[int, Field(default=6, gt=0, description="Max tree depth")]
    learning_rate: Annotated[float, Field(default=0.1, gt=0, le=1, description="Learning rate")]
    subsample: Annotated[float, Field(default=0.8, gt=0, le=1, description="Row subsample ratio")]
    colsample_bytree: Annotated[float, Field(default=0.8, gt=0, le=1, description="Column subsample ratio")]
    random_state: Annotated[int, Field(default=42, description="Random seed")]

    def to_sklearn(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
        )
