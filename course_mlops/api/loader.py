import logging
import os
from datetime import UTC
from datetime import datetime
from typing import NamedTuple
from typing import TypedDict

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from course_mlops.api.enums import ModelStrategy
from course_mlops.api.exceptions import ModelMetadataFetchError
from course_mlops.train.config import get_settings
from course_mlops.train.enums import ModelType

logger = logging.getLogger(__name__)

MS_THRESHOLD = 1e10
BEST_METRIC = "f1"  # metric used for "best" strategy


class ModelMetadata(TypedDict, total=False):
    model_uri: str
    model_version: str
    model_type: ModelType
    run_id: str
    artifact_uri: str
    registered_at: str
    strategy: str


class LoadResult(NamedTuple):
    model: PyFuncModel
    metadata: ModelMetadata


class ModelLoader:
    def __init__(self) -> None:
        self._tracking_uri = self._get_tracking_uri()
        self._client: MlflowClient | None = None

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            mlflow.set_tracking_uri(self._tracking_uri)
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    def load(self, model_name: str, strategy: str) -> LoadResult:
        try:
            model_version = self._resolve_strategy(model_name, strategy)
            model_uri = f"models:/{model_name}/{model_version.version}"

            model = mlflow.pyfunc.load_model(model_uri)
            metadata = self._build_metadata(model_version, model_uri, strategy)

            logger.info(f"Model loaded: {model_uri} (strategy: {strategy})")
            return LoadResult(model, metadata)

        except ModelMetadataFetchError:
            raise
        except Exception as e:
            raise ModelMetadataFetchError(f"Failed to load model: {e}") from e

    def _resolve_strategy(self, model_name: str, strategy: str) -> ModelVersion:
        try:
            parsed_strategy, value = ModelStrategy.parse(strategy)
        except ValueError as e:
            raise ModelMetadataFetchError(f"Invalid strategy: {strategy}") from e

        versions = self.client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ModelMetadataFetchError(f"No versions found for model {model_name}")

        if parsed_strategy == ModelStrategy.LATEST:
            return max(versions, key=lambda v: int(v.version))

        if parsed_strategy == ModelStrategy.BEST:
            return self._find_best_by_metric(versions)

        if parsed_strategy == ModelStrategy.VERSION:
            if value is None:
                raise ModelMetadataFetchError("VERSION strategy requires a value (e.g., 'version:3')")
            match = next((v for v in versions if v.version == value), None)
            if match is None:
                raise ModelMetadataFetchError(f"Version '{value}' not found for model '{model_name}'")
            return match

        if parsed_strategy == ModelStrategy.TYPE:
            if value is None:
                raise ModelMetadataFetchError("TYPE strategy requires a value (e.g., 'type:xgboost')")
            return self._find_latest_by_type(versions, value)

        return max(versions, key=lambda v: int(v.version))

    def _find_best_by_metric(self, versions: list[ModelVersion]) -> ModelVersion:
        """Find the model version with the best F1 score."""
        run_id_to_version = {v.run_id: v for v in versions if v.run_id}
        if not run_id_to_version:
            raise ModelMetadataFetchError(f"No model found with metric '{BEST_METRIC}'")

        best_version = None
        best_score = -1.0
        for run_id, version in run_id_to_version.items():
            run = self.client.get_run(run_id)
            score = run.data.metrics.get(BEST_METRIC, -1.0)
            if score > best_score:
                best_score = score
                best_version = version

        if best_version is None:
            raise ModelMetadataFetchError(f"No model found with metric '{BEST_METRIC}'")

        logger.info(f"Best model: version {best_version.version} with {BEST_METRIC}={best_score:.4f}")
        return best_version

    def _find_latest_by_type(self, versions: list[ModelVersion], model_type: str) -> ModelVersion:
        matching = [v for v in versions if v.tags and v.tags.get("model_type") == model_type]

        if not matching:
            raise ModelMetadataFetchError(f"No model found with type '{model_type}'")

        return max(matching, key=lambda v: int(v.version))

    def _build_metadata(
        self,
        version: ModelVersion,
        model_uri: str,
        strategy: str,
    ) -> ModelMetadata:
        metadata: ModelMetadata = {
            "model_uri": model_uri,
            "model_version": version.version,
            "run_id": version.run_id,
            "artifact_uri": version.source,
            "strategy": strategy,
            **({"model_type": version.tags["model_type"]} if version.tags and version.tags.get("model_type") else {}),
            **(
                {"registered_at": self._timestamp_to_iso(version.creation_timestamp)}
                if version.creation_timestamp
                else {}
            ),
        }
        return metadata

    @staticmethod
    def _get_tracking_uri() -> str:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            settings = get_settings()
            tracking_uri = settings.mlflow.tracking_uri
        return tracking_uri

    @staticmethod
    def _timestamp_to_iso(timestamp: int | float) -> str:
        ts_seconds = timestamp / 1000 if timestamp > MS_THRESHOLD else timestamp
        dt = datetime.fromtimestamp(ts_seconds, tz=UTC)
        return dt.isoformat()
