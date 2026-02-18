from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.sparse import spmatrix


class BaseClassifier(ABC):
    """Abstract base class for all classifiers in the project."""

    params_class: ClassVar[type[BaseModel]]

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name for logging/tracking."""
        ...

    @abstractmethod
    def fit(self, X: spmatrix | np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train the model on the given data."""
        ...

    @abstractmethod
    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Return class predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters for logging."""
        ...
