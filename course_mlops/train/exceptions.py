from course_mlops.exceptions import CourseMLOpsError
from course_mlops.exceptions import OriginError
from course_mlops.exceptions import ReasonError


class DataValidationError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.VAL
    code = 1
    default_message = "Data validation failed"


class FeatureExtractionError(CourseMLOpsError):
    origin = OriginError.FEA
    error_type = ReasonError.VAL
    code = 1
    default_message = "Feature extraction failed"


class ModelTrainingError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.VAL
    code = 1
    default_message = "Model training failed"


class EvaluationError(CourseMLOpsError):
    origin = OriginError.EVA
    error_type = ReasonError.VAL
    code = 1
    default_message = "Model evaluation failed"


class ConfigurationError(CourseMLOpsError):
    origin = OriginError.CFG
    error_type = ReasonError.VAL
    code = 1
    default_message = "Configuration validation failed"


class DataNotFoundError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.NOT
    code = 2
    default_message = "Data file not found"


class DataLoadError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.IO
    code = 3
    default_message = "Failed to load data"


class ModelNotFittedError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.VAL
    code = 5
    default_message = "Model must be fitted before calling predict"
