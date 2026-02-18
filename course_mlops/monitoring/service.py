import hashlib
import logging

from course_mlops.monitoring import dal
from course_mlops.monitoring.schemas import PredictionRecord
from course_mlops.train.preprocessing.data import preprocess_message
from course_mlops.train.preprocessing.features import compute_numerical_features

logger = logging.getLogger(__name__)


async def log_prediction_to_store(
    model_version: str | None,
    message: str,
    prediction: str,
    probability: float,
    confidence: str,
) -> None:
    try:
        await dal.log_prediction(
            PredictionRecord(
                model_version=model_version or "unknown",
                message_hash=hashlib.sha256(message.encode()).hexdigest(),
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                **compute_numerical_features(preprocess_message(message)),
            )
        )
    except Exception:
        logger.warning("Failed to log prediction to monitoring store", exc_info=True)
