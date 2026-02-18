import pytest

from course_mlops.api.enums import ModelStrategy
from course_mlops.train.enums import ModelType


@pytest.mark.parametrize(
    ("strategy", "value", "expected"),
    [
        (ModelStrategy.VERSION, "3", "version:3"),
        (ModelStrategy.TYPE, ModelType.XGBOOST, "type:xgboost"),
        (ModelStrategy.LATEST, "foo", "latest:foo"),
    ],
)
def test_model_strategy_with_value(strategy: ModelStrategy, value: str, expected: str) -> None:
    assert strategy.with_value(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected_strategy", "expected_value"),
    [
        ("latest", ModelStrategy.LATEST, None),
        ("best", ModelStrategy.BEST, None),
        ("version:3", ModelStrategy.VERSION, "3"),
        ("type:xgboost", ModelStrategy.TYPE, ModelType.XGBOOST),
    ],
)
def test_model_strategy_parse(raw: str, expected_strategy: ModelStrategy, expected_value: str | None) -> None:
    strategy, value = ModelStrategy.parse(raw)
    assert strategy == expected_strategy
    assert value == expected_value


def test_model_strategy_parse_invalid_raises() -> None:
    with pytest.raises(ValueError, match="is not a valid"):
        ModelStrategy.parse("invalid_strategy")
