import json
import logging
import logging.config
from datetime import UTC
from datetime import datetime
from enum import IntEnum
from enum import StrEnum
from enum import auto
from typing import Self

from course_mlops.utils import EnvironmentVariable


class LogFormat(StrEnum):
    JSON = auto()
    HUMAN = auto()


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation systems.

    Output format:
        {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "level": "INFO",
            "message": "Model loaded",
            "logger": "course_mlops.api.main",
            "lineno": 42,
            "exception": null,
            "custom_field": "value"  # from extra={"x-custom_field": "value"}
        }

    Any key in the `extra` argument starting with "x-" will be included.
    """

    def format(self: Self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "lineno": record.lineno,
            "exception": self.formatException(record.exc_info) if record.exc_info else None,
            **{key[2:]: value for key, value in record.__dict__.items() if key.startswith("x-")},
        }
        return json.dumps(log_data)


def set_logging_config() -> dict:
    """Configure logging based on environment variables.

    Environment variables:
        CML_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        CML_LOG_FORMAT: JSON, HUMAN (default: HUMAN)

    Returns:
        The logging configuration dict that was applied.

    Example:
        >>> set_logging_config()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Hello", extra={"x-run_id": "abc123"})
    """
    log_level_str = EnvironmentVariable.LOG_LEVEL.read("INFO").upper()
    log_level = LogLevel.__members__.get(log_level_str, LogLevel.INFO)

    log_format_str = EnvironmentVariable.LOG_FORMAT.read("HUMAN").upper()
    log_format = LogFormat.__members__.get(log_format_str, LogFormat.HUMAN)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            LogFormat.JSON: {"()": "course_mlops.logging.JsonFormatter"},
            LogFormat.HUMAN: {
                "format": "%(levelname)1.1s: %(asctime)s - %(message)s - %(name)s[L%(lineno)s]",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": log_format,
                "level": log_level.value,
            }
        },
        "loggers": {
            "urllib3": {"level": "WARNING"},
            "mlflow": {"level": "WARNING"},
            "": {"handlers": ["console"], "level": log_level.value},
        },
    }

    logging.config.dictConfig(config)
    return config
