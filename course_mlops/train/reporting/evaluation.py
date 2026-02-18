from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from course_mlops.train.exceptions import EvaluationError


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> EvaluationMetrics:
    if len(y_true) == 0 or len(y_pred) == 0:
        raise EvaluationError("Cannot evaluate with empty predictions")

    if len(y_true) != len(y_pred):
        raise EvaluationError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    metrics = EvaluationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, pos_label=1),
        recall=recall_score(y_true, y_pred, pos_label=1),
        f1=f1_score(y_true, y_pred, pos_label=1),
        roc_auc=0.0,
    )

    if y_proba is not None:
        spam_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        metrics.roc_auc = roc_auc_score(
            y_true,
            spam_proba,
        )

    return metrics
