from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import OriginError
from course_mlops.exceptions import ReasonError


class DALError(CourseMLOpsError):
    origin = OriginError.MON
    error_type = ReasonError.IO
    code = 10
    default_message = "Error during database transaction."


class TransactionError(DALError):
    code = 11
    default_message = "Error during database transaction."


class NotFoundError(DALError):
    error_type = ReasonError.NOT
    code = 12
    default_message = "Object not found in database."


class IntegrityError(DALError):
    error_type = ReasonError.INT
    code = 13
    default_message = "Integrity error when updating database."
