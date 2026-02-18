from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import OriginError
from course_mlops.exceptions import ReasonError


class MonitoringDatabaseError(CourseMLOpsError):
    origin = OriginError.MON
    error_type = ReasonError.IO
    code = 1
    default_message = "Monitoring database error."


class DriftDetectionError(CourseMLOpsError):
    origin = OriginError.MON
    error_type = ReasonError.INT
    code = 2
    default_message = "Drift detection failed."


class InsufficientDataError(CourseMLOpsError):
    origin = OriginError.MON
    error_type = ReasonError.VAL
    code = 3
    default_message = "Insufficient data to compute drift."


class ReferenceDataNotFoundError(CourseMLOpsError):
    origin = OriginError.MON
    error_type = ReasonError.NOT
    code = 4
    default_message = "Reference dataset not found."
