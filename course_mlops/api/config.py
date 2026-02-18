from course_mlops.api.enums import ModelStrategy
from course_mlops.utils import EnvironmentVariable

API_TITLE = "Spam Classifier API"
API_VERSION = "0.3.0"
MODEL_NAME = "spam-classifier"

MODEL_STRATEGY = EnvironmentVariable.MODEL_STRATEGY.read(default=ModelStrategy.LATEST)

MLFLOW_UI_BASE = EnvironmentVariable.MLFLOW_UI_BASE.read(default="http://localhost:5001")
API_HOST = EnvironmentVariable.API_HOST.read(default="0.0.0.0")
API_PORT = int(EnvironmentVariable.API_PORT.read(default="8000"))
