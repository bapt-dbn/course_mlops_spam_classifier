import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from course_mlops.api.config import API_TITLE
from course_mlops.api.config import API_VERSION
from course_mlops.api.config import MODEL_NAME
from course_mlops.api.config import MODEL_STRATEGY
from course_mlops.api.exception_handlers import course_mlops_exception_handler
from course_mlops.api.exception_handlers import general_exception_handler
from course_mlops.api.loader import ModelLoader
from course_mlops.api.routes import router
from course_mlops.api.service import PredictionService
from course_mlops.common.dal.transaction import Transaction
from course_mlops.exceptions import CourseMLOpsError
from course_mlops.logging import set_logging_config
from course_mlops.monitoring import dal
from course_mlops.monitoring.config import MonitoringConfig
from course_mlops.monitoring.drift import DriftDetector
from course_mlops.monitoring.reference import load_reference
from course_mlops.monitoring.routes import monitoring_router
from course_mlops.utils import EnvironmentVariable

set_logging_config()
logger = logging.getLogger(__name__)


def _is_fail_fast_enabled() -> bool:
    return EnvironmentVariable.FAIL_FAST.read(default="true").lower() in ("true", "1", "yes")


@asynccontextmanager
async def _prediction_service(app: FastAPI) -> AsyncGenerator[None, None]:
    loader = ModelLoader()
    service = PredictionService(loader)

    try:
        service.load_model(MODEL_NAME, MODEL_STRATEGY)
        logger.info(
            "Model loaded successfully",
            extra={"x-model_name": MODEL_NAME, "x-strategy": MODEL_STRATEGY},
        )
    except Exception as e:
        logger.error(
            f"Model loading failed: {e}",
            extra={"x-model_name": MODEL_NAME, "x-strategy": MODEL_STRATEGY},
        )
        if _is_fail_fast_enabled():
            raise RuntimeError(
                f"Failed to load model '{MODEL_NAME}' with strategy '{MODEL_STRATEGY}'. "
                "Train the model first with 'mlops_course train' or set CML_FAIL_FAST=false to start without model."
            ) from e
        logger.warning("Starting without model (CML_FAIL_FAST=false). Some endpoints will be unavailable.")

    app.state.prediction_service = service
    yield


@asynccontextmanager
async def _monitoring_db(app: FastAPI) -> AsyncGenerator[None, None]:
    monitoring_config = MonitoringConfig.from_env()
    app.state.monitoring_config = monitoring_config

    if not monitoring_config.enabled:
        logger.info("Monitoring disabled by configuration")
        yield
        return

    try:
        await dal.ping()
        logger.info("Monitoring database connected")
    except Exception as e:
        logger.warning(f"Failed to connect to monitoring database: {e}. Monitoring disabled.")
        monitoring_config.enabled = False

    try:
        yield
    finally:
        await Transaction.dispose_engine()


@asynccontextmanager
async def _drift_detector(app: FastAPI) -> AsyncGenerator[None, None]:
    service: PredictionService | None = getattr(app.state, "prediction_service", None)
    monitoring_config: MonitoringConfig | None = getattr(app.state, "monitoring_config", None)

    if monitoring_config and monitoring_config.enabled and service and service.run_id:
        try:
            tracking_uri = EnvironmentVariable.MLFLOW_TRACKING_URI.read(default="http://localhost:5001")
            reference_df = load_reference(service.run_id, tracking_uri)
            app.state.drift_detector = DriftDetector(reference_df)
            logger.info("Drift detector initialized with reference dataset")
        except Exception as e:
            logger.warning(f"Failed to load reference dataset: {e}. Drift detection unavailable.")

    yield


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    async with _prediction_service(app), _monitoring_db(app), _drift_detector(app):
        yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Spam Classifier API",
    lifespan=lifespan,
)

app.add_exception_handler(CourseMLOpsError, course_mlops_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(router)
app.include_router(monitoring_router)


@app.get("/", tags=["info"])
async def root() -> dict[str, str]:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
