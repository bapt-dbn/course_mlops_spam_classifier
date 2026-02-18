import json
import logging
import logging.config
import sys
from datetime import UTC
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from course_mlops.logging import JsonFormatter
from course_mlops.logging import LogFormat
from course_mlops.logging import LogLevel
from course_mlops.logging import set_logging_config
from course_mlops.utils import EnvironmentVariable


@pytest.fixture
def caplog_json(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    set_logging_config()
    caplog.set_level(logging.DEBUG)
    yield caplog
    set_logging_config()


def test_json_formatter_basic_formatting() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC).timestamp()

    formatted_log = json.loads(formatter.format(record))
    assert formatted_log["timestamp"] == "2024-01-15T10:30:00+00:00"
    assert formatted_log["level"] == "INFO"
    assert formatted_log["message"] == "Test message"
    assert formatted_log["logger"] == "test_logger"
    assert formatted_log["lineno"] == 10
    assert formatted_log["exception"] is None


def test_json_formatter_formatting_with_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("Test error")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=20,
        msg="Error message",
        args=(),
        exc_info=exc_info,
    )
    record.created = datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC).timestamp()

    formatted_log = json.loads(formatter.format(record))
    assert formatted_log["level"] == "ERROR"
    assert formatted_log["message"] == "Error message"
    assert "ValueError: Test error" in formatted_log["exception"]
    assert "Traceback" in formatted_log["exception"]


def test_json_formatter_formatting_with_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=30,
        msg="Debug message",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC).timestamp()
    record.__dict__["x-custom_field"] = "custom_value"
    record.__dict__["x-run_id"] = "abc-123"

    formatted_log = json.loads(formatter.format(record))
    assert formatted_log["message"] == "Debug message"
    assert formatted_log["custom_field"] == "custom_value"
    assert formatted_log["run_id"] == "abc-123"


# Parametrized tests for set_logging_config
@pytest.mark.parametrize(
    ("env_vars", "expected_level", "expected_format"),
    [
        ({}, LogLevel.INFO, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_LEVEL.value: "DEBUG"}, LogLevel.DEBUG, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_LEVEL.value: "WARNING"}, LogLevel.WARNING, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_LEVEL.value: "ERROR"}, LogLevel.ERROR, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_LEVEL.value: "CRITICAL"}, LogLevel.CRITICAL, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_LEVEL.value: "invalid"}, LogLevel.INFO, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_FORMAT.value: "JSON"}, LogLevel.INFO, LogFormat.JSON),
        ({EnvironmentVariable.LOG_FORMAT.value: "HUMAN"}, LogLevel.INFO, LogFormat.HUMAN),
        ({EnvironmentVariable.LOG_FORMAT.value: "invalid"}, LogLevel.INFO, LogFormat.HUMAN),
        (
            {EnvironmentVariable.LOG_LEVEL.value: "DEBUG", EnvironmentVariable.LOG_FORMAT.value: "JSON"},
            LogLevel.DEBUG,
            LogFormat.JSON,
        ),
    ],
)
@patch("logging.config.dictConfig")
def test_set_logging_config_env_vars(
    mock_dict_config: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    env_vars: dict[str, str],
    expected_level: LogLevel,
    expected_format: LogFormat,
) -> None:
    # Clear env vars and set new ones for a clean state
    monkeypatch.delenv(EnvironmentVariable.LOG_LEVEL.value, raising=False)
    monkeypatch.delenv(EnvironmentVariable.LOG_FORMAT.value, raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = set_logging_config()
    mock_dict_config.assert_called_once_with(config)  # Verify dictConfig was called

    assert config["handlers"]["console"]["level"] == expected_level.value
    assert config["handlers"]["console"]["formatter"] == expected_format.value
    assert config["loggers"]["urllib3"]["level"] == "WARNING"
    assert config["loggers"]["mlflow"]["level"] == "WARNING"
    assert config["loggers"][""]["level"] == expected_level.value


@patch("logging.config.dictConfig")
def test_set_logging_config_applies_correct_default_config(mock_dict_config: MagicMock) -> None:
    config = set_logging_config()
    mock_dict_config.assert_called_once_with(config)

    assert config["formatters"][LogFormat.JSON]["()"] == "course_mlops.logging.JsonFormatter"
    assert (
        config["formatters"][LogFormat.HUMAN]["format"]
        == "%(levelname)1.1s: %(asctime)s - %(message)s - %(name)s[L%(lineno)s]"
    )

    assert config["handlers"]["console"]["class"] == "logging.StreamHandler"
    assert config["handlers"]["console"]["formatter"] == LogFormat.HUMAN.value  # Use .value
    assert config["handlers"]["console"]["level"] == LogLevel.INFO.value  # Use .value

    assert config["loggers"]["urllib3"]["level"] == "WARNING"
    assert config["loggers"]["mlflow"]["level"] == "WARNING"
    assert config["loggers"][""]["handlers"] == ["console"]
    assert config["loggers"][""]["level"] == LogLevel.INFO.value  # Use .value
