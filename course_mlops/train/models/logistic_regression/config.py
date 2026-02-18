from typing import Annotated

from pydantic import BaseModel
from pydantic import Field
from sklearn.linear_model import LogisticRegression


class LogisticRegressionParams(BaseModel):
    C: Annotated[float, Field(default=0.5, gt=0, description="Inverse regularization strength")]
    max_iter: Annotated[int, Field(default=1000, gt=0, description="Max iterations")]
    solver: Annotated[str, Field(default="liblinear", description="Optimization algorithm")]

    def to_sklearn(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
        )
