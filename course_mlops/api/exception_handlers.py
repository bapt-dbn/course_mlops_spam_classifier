import logging

from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from course_mlops.api.schemas import ErrorOutput
from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import InvalidInputError
from course_mlops.exceptions import ModelNotLoadedError
from course_mlops.monitoring.exceptions import InsufficientDataError
from course_mlops.monitoring.exceptions import ReferenceDataNotFoundError

logger = logging.getLogger(__name__)


def course_mlops_exception_handler(
    request: Request,
    exc: CourseMLOpsError,
) -> JSONResponse:
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, ModelNotLoadedError):
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, InvalidInputError | InsufficientDataError):
        http_status = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, ReferenceDataNotFoundError):
        http_status = status.HTTP_404_NOT_FOUND

    return JSONResponse(
        status_code=http_status,
        content=ErrorOutput(
            detail=exc.message,
            error_code=exc.error_code,
        ).model_dump(mode="json"),
    )


def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Internal server error"

    if isinstance(exc, HTTPException):
        http_status = exc.status_code
        detail = exc.detail
        if http_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error(f"{detail}")
        else:
            logger.warning(f"{detail}")

    elif isinstance(exc, RequestValidationError):
        http_status = status.HTTP_422_UNPROCESSABLE_CONTENT
        messages = [str(e.get("msg", "")) for e in exc.errors()]
        detail = "; ".join(messages) if messages else "Validation error"
        logger.error(f"Validation error on {request.method} {request.url}: {detail}")

    else:
        logger.error(f"Unhandled exception on {request.method} {request.url}")

    return JSONResponse(
        status_code=http_status,
        content=ErrorOutput(detail=detail).model_dump(mode="json"),
    )
