from course_mlops.train.enums import ModelType
from course_mlops.train.models.base import BaseClassifier
from course_mlops.train.models.logistic_regression import LogisticRegressionClassifier
from course_mlops.train.models.xgboost import XGBoostClassifier


def get_classifier(model_type: ModelType) -> type[BaseClassifier]:
    classifiers = {
        ModelType.LOGISTIC_REGRESSION: LogisticRegressionClassifier,
        ModelType.XGBOOST: XGBoostClassifier,
    }
    if model_type not in classifiers:
        raise ValueError(f"Unknown model type: {model_type}")
    return classifiers[model_type]
