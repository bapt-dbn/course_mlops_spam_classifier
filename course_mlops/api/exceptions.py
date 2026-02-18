from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import OriginError
from course_mlops.exceptions import ReasonError


class ModelPredictionError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.INT
    code = 3
    default_message = "Prediction failed. Please try again later."


class ModelMetadataFetchError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.IO
    code = 4
    default_message = "Failed to fetch model metadata from MLFlow."
