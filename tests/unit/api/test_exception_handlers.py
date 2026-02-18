import json
from unittest.mock import Mock

import pytest
from fastapi import status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from course_mlops.api.exception_handlers import course_mlops_exception_handler
from course_mlops.api.exception_handlers import general_exception_handler
from course_mlops.api.exceptions import ModelPredictionError
from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import InvalidInputError
from course_mlops.exceptions import ModelNotLoadedError


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (ModelNotLoadedError("Not loaded"), status.HTTP_503_SERVICE_UNAVAILABLE),
        (InvalidInputError("Bad input"), status.HTTP_400_BAD_REQUEST),
        (ModelPredictionError("Failed"), status.HTTP_500_INTERNAL_SERVER_ERROR),
    ],
)
def test_course_mlops_handler_status_codes(exc: CourseMLOpsError, expected_status: int) -> None:
    response = course_mlops_exception_handler(Mock(), exc)
    assert response.status_code == expected_status


def test_course_mlops_handler_response_body() -> None:
    exc = ModelNotLoadedError("Not loaded")
    response = course_mlops_exception_handler(Mock(), exc)
    body = json.loads(response.body)

    assert body["detail"] == "Not loaded"
    assert body["error_code"] == exc.error_code


def test_general_handler_http_exception_server_error() -> None:
    exc = HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error")
    response = general_exception_handler(Mock(), exc)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body)["detail"] == "Server error"


def test_general_handler_http_exception_client_error() -> None:
    exc = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    response = general_exception_handler(Mock(), exc)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert json.loads(response.body)["detail"] == "Not found"


def test_general_handler_validation_error_with_messages() -> None:
    exc = RequestValidationError(
        errors=[{"msg": "field required", "loc": ("body", "field"), "type": "missing"}],
    )
    mock_request = Mock()
    mock_request.method = "POST"
    mock_request.url = "http://test/api/v1/predict"

    response = general_exception_handler(mock_request, exc)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "field required" in json.loads(response.body)["detail"]


def test_general_handler_validation_error_empty() -> None:
    exc = RequestValidationError(errors=[])
    mock_request = Mock()
    mock_request.method = "POST"
    mock_request.url = "http://test"

    response = general_exception_handler(mock_request, exc)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert json.loads(response.body)["detail"] == "Validation error"


def test_general_handler_generic_exception() -> None:
    exc = RuntimeError("Something broke")
    mock_request = Mock()
    mock_request.method = "GET"
    mock_request.url = "http://test"

    response = general_exception_handler(mock_request, exc)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body)["detail"] == "Internal server error"
