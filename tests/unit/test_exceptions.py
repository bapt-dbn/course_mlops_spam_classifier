import re

import pytest

from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import InvalidInputError
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.monitoring.exceptions import DriftDetectionError
from course_mlops.monitoring.exceptions import InsufficientDataError
from course_mlops.monitoring.exceptions import MonitoringDatabaseError
from course_mlops.monitoring.exceptions import ReferenceDataNotFoundError
from course_mlops.train.exceptions import ConfigurationError
from course_mlops.train.exceptions import DataLoadError
from course_mlops.train.exceptions import DataNotFoundError
from course_mlops.train.exceptions import DataValidationError
from course_mlops.train.exceptions import EvaluationError
from course_mlops.train.exceptions import FeatureExtractionError
from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.exceptions import ModelTrainingError

ALL_EXCEPTIONS: list[type[CourseMLOpsError]] = [
    ModelNotLoadedError,
    InvalidInputError,
    DataValidationError,
    FeatureExtractionError,
    ModelTrainingError,
    EvaluationError,
    ConfigurationError,
    DataNotFoundError,
    DataLoadError,
    ModelNotFittedError,
    MonitoringDatabaseError,
    DriftDetectionError,
    InsufficientDataError,
    ReferenceDataNotFoundError,
]


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_error_code_format(exc_cls: type[CourseMLOpsError]) -> None:
    assert re.match(r"^CML-[A-Z]{3}-[A-Z]{2,3}-\d{3}$", exc_cls().error_code)


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_default_message(exc_cls: type[CourseMLOpsError]) -> None:
    exc = exc_cls()
    assert exc.message == exc_cls.default_message


def test_custom_message_overrides_default() -> None:
    exc = DataValidationError("custom error")
    assert exc.message == "custom error"


def test_str_contains_error_code_and_message() -> None:
    exc = ModelNotLoadedError("Service down")
    assert str(exc) == f"[{exc.error_code}] Service down"


def test_init_raises_when_no_message_and_no_default() -> None:
    class BareError(CourseMLOpsError):
        origin = "TST"
        error_type = "VAL"
        code = 999
        default_message = None

    with pytest.raises(ValueError, match="requires a message"):
        BareError()
