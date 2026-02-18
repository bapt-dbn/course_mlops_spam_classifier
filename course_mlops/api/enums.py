from enum import StrEnum
from enum import auto


class PredictionEnum(StrEnum):
    SPAM = auto()
    HAM = auto()


class ConfidenceEnum(StrEnum):
    HIGH = auto()
    LOW = auto()


class HealthStatusEnum(StrEnum):
    HEALTHY = auto()
    DEGRADED = auto()


class ModelStrategy(StrEnum):
    """Strategy for selecting which model version to load.

    Basic strategies:
        LATEST: Most recent registered version
        BEST: Version with best F1 score

    Prefix strategies (use with .with_value()):
        VERSION: Specific version number (e.g., "version:3")
        TYPE: Latest of a model type (e.g., "type:xgboost")

    Example:
        >>> ModelStrategy.LATEST
        'latest'
        >>> ModelStrategy.VERSION.with_value("3")
        'version:3'
    """

    LATEST = auto()
    BEST = auto()
    VERSION = auto()  # e.g., version:N
    TYPE = auto()  # e.g., model:X

    def with_value(self, value: str) -> str:
        return f"{self.value}:{value}"

    @classmethod
    def parse(cls, strategy_str: str) -> tuple["ModelStrategy", str | None]:
        if ":" in strategy_str:
            prefix, value = strategy_str.split(":", 1)
            return cls(prefix), value
        return cls(strategy_str), None
