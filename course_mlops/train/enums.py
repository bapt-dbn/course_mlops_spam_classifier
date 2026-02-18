from enum import StrEnum
from enum import auto


class ModelType(StrEnum):
    LOGISTIC_REGRESSION = auto()
    XGBOOST = auto()


class NumericalFeature(StrEnum):
    TEXT_LENGTH = auto()
    WORD_COUNT = auto()
    CAPS_RATIO = auto()
    SPECIAL_CHARS_COUNT = auto()
