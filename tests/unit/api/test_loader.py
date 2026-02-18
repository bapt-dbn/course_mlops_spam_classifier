from dataclasses import dataclass
from dataclasses import field
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from course_mlops.api import loader as loader_module
from course_mlops.api.exceptions import ModelMetadataFetchError
from course_mlops.api.loader import ModelLoader
from course_mlops.train.enums import ModelType


@dataclass
class FakeModelVersion:
    version: str = "1"
    run_id: str | None = "run1"
    tags: dict[str, str] | None = None
    source: str = "s3://artifacts"
    creation_timestamp: int | None = None


@dataclass
class FakeRunInfo:
    run_id: str = "run1"


@dataclass
class FakeRunData:
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeRun:
    info: FakeRunInfo = field(default_factory=FakeRunInfo)
    data: FakeRunData = field(default_factory=FakeRunData)


@patch("course_mlops.api.loader.MlflowClient")
@patch("course_mlops.api.loader.mlflow")
@patch.object(ModelLoader, "_get_tracking_uri", return_value="http://mlflow:5000")
def test_init_sets_tracking_uri(mock_uri: Mock, mock_mlflow: Mock, mock_client_cls: Mock) -> None:
    inst = ModelLoader()

    assert inst._tracking_uri == "http://mlflow:5000"
    assert inst._client is None


@patch("course_mlops.api.loader.MlflowClient")
@patch("course_mlops.api.loader.mlflow")
@patch.object(ModelLoader, "_get_tracking_uri", return_value="http://mlflow:5000")
def test_client_lazy_creates_instance(mock_uri: Mock, mock_mlflow: Mock, mock_client_cls: Mock) -> None:
    inst = ModelLoader()
    client = inst.client

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
    mock_client_cls.assert_called_once_with(tracking_uri="http://mlflow:5000")
    assert client is mock_client_cls.return_value


@patch("course_mlops.api.loader.MlflowClient")
@patch("course_mlops.api.loader.mlflow")
@patch.object(ModelLoader, "_get_tracking_uri", return_value="http://mlflow:5000")
def test_client_returns_same_instance(mock_uri: Mock, mock_mlflow: Mock, mock_client_cls: Mock) -> None:
    inst = ModelLoader()
    client1 = inst.client
    client2 = inst.client

    mock_client_cls.assert_called_once()
    assert client1 is client2


def test_load_success(loader: ModelLoader) -> None:
    version = FakeModelVersion(
        version="1",
        tags={"model_type": ModelType.LOGISTIC_REGRESSION},
        creation_timestamp=1705312200000,
    )
    loader.client.search_model_versions.return_value = [version]

    mock_model = Mock()
    loader_module.mlflow.pyfunc.load_model.return_value = mock_model

    result = loader.load("spam-classifier", "latest")

    assert result.model is mock_model
    assert result.metadata["model_uri"] == "models:/spam-classifier/1"
    assert result.metadata["model_version"] == "1"


def test_load_reraises_metadata_fetch_error(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = []

    with pytest.raises(ModelMetadataFetchError, match="No versions found"):
        loader.load("spam-classifier", "latest")


def test_load_wraps_generic_exception(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [FakeModelVersion()]
    loader_module.mlflow.pyfunc.load_model.side_effect = RuntimeError("Connection refused")

    with pytest.raises(ModelMetadataFetchError, match="Failed to load model"):
        loader.load("spam-classifier", "latest")


def test_resolve_strategy_latest(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [
        FakeModelVersion(version="1"),
        FakeModelVersion(version="2"),
    ]

    assert loader._resolve_strategy("model", "latest").version == "2"


def test_resolve_strategy_best(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [
        FakeModelVersion(version="1", run_id="run1"),
        FakeModelVersion(version="2", run_id="run2"),
    ]
    runs = {
        "run1": FakeRun(info=FakeRunInfo(run_id="run1"), data=FakeRunData(metrics={"f1": 0.85})),
        "run2": FakeRun(info=FakeRunInfo(run_id="run2"), data=FakeRunData(metrics={"f1": 0.92})),
    }
    loader.client.get_run.side_effect = lambda run_id: runs[run_id]

    assert loader._resolve_strategy("model", "best").version == "2"


def test_resolve_strategy_version_found(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [
        FakeModelVersion(version="1"),
        FakeModelVersion(version="2"),
    ]

    assert loader._resolve_strategy("model", "version:2").version == "2"


def test_resolve_strategy_version_not_found(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [FakeModelVersion(version="1")]

    with pytest.raises(ModelMetadataFetchError, match="Version '99' not found"):
        loader._resolve_strategy("model", "version:99")


def test_resolve_strategy_version_without_value(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [FakeModelVersion()]

    with pytest.raises(ModelMetadataFetchError, match="VERSION strategy requires a value"):
        loader._resolve_strategy("model", "version")


def test_resolve_strategy_type_found(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [
        FakeModelVersion(version="1", tags={"model_type": ModelType.LOGISTIC_REGRESSION}),
        FakeModelVersion(version="2", tags={"model_type": ModelType.XGBOOST}),
    ]

    assert loader._resolve_strategy("model", "type:xgboost").version == "2"


def test_resolve_strategy_type_without_value(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = [FakeModelVersion()]

    with pytest.raises(ModelMetadataFetchError, match="TYPE strategy requires a value"):
        loader._resolve_strategy("model", "type")


def test_resolve_strategy_invalid(loader: ModelLoader) -> None:
    with pytest.raises(ModelMetadataFetchError, match="Invalid strategy"):
        loader._resolve_strategy("model", "invalid")


def test_resolve_strategy_no_versions(loader: ModelLoader) -> None:
    loader.client.search_model_versions.return_value = []

    with pytest.raises(ModelMetadataFetchError, match="No versions found"):
        loader._resolve_strategy("model", "latest")


@patch("course_mlops.api.loader.ModelStrategy.parse", return_value=(Mock(), None))
def test_resolve_strategy_fallback(mock_parse: Mock, loader: ModelLoader) -> None:
    """Cover the final fallback return when strategy matches no known branch."""
    loader.client.search_model_versions.return_value = [
        FakeModelVersion(version="1"),
        FakeModelVersion(version="3"),
    ]

    result = loader._resolve_strategy("model", "something")

    assert result.version == "3"


def test_find_best_by_metric_success(loader: ModelLoader) -> None:
    runs = {
        "run1": FakeRun(info=FakeRunInfo(run_id="run1"), data=FakeRunData(metrics={"f1": 0.85})),
        "run2": FakeRun(info=FakeRunInfo(run_id="run2"), data=FakeRunData(metrics={"f1": 0.92})),
    }
    loader.client.get_run.side_effect = lambda run_id: runs[run_id]

    versions = [
        FakeModelVersion(version="1", run_id="run1"),
        FakeModelVersion(version="2", run_id="run2"),
    ]

    assert loader._find_best_by_metric(versions).version == "2"


def test_find_best_by_metric_no_run_ids(loader: ModelLoader) -> None:
    with pytest.raises(ModelMetadataFetchError, match="No model found with metric"):
        loader._find_best_by_metric([FakeModelVersion(run_id=None)])


def test_find_best_by_metric_no_runs_returned(loader: ModelLoader) -> None:
    loader.client.get_run.return_value = FakeRun(info=FakeRunInfo(run_id="run1"), data=FakeRunData(metrics={}))

    with pytest.raises(ModelMetadataFetchError, match="No model found with metric"):
        loader._find_best_by_metric([FakeModelVersion(run_id="run1")])


def test_find_latest_by_type_success(loader: ModelLoader) -> None:
    versions = [
        FakeModelVersion(version="1", tags={"model_type": ModelType.XGBOOST}),
        FakeModelVersion(version="3", tags={"model_type": ModelType.XGBOOST}),
        FakeModelVersion(version="5", tags={"model_type": ModelType.LOGISTIC_REGRESSION}),
    ]

    assert loader._find_latest_by_type(versions, ModelType.XGBOOST).version == "3"


def test_find_latest_by_type_no_match(loader: ModelLoader) -> None:
    with pytest.raises(ModelMetadataFetchError, match="No model found with type"):
        loader._find_latest_by_type(
            [FakeModelVersion(tags={"model_type": ModelType.LOGISTIC_REGRESSION})], ModelType.XGBOOST
        )


def test_find_latest_by_type_no_tags(loader: ModelLoader) -> None:
    with pytest.raises(ModelMetadataFetchError, match="No model found with type"):
        loader._find_latest_by_type([FakeModelVersion(tags=None)], ModelType.XGBOOST)


def test_build_metadata_full(loader: ModelLoader) -> None:
    version = FakeModelVersion(
        version="2",
        run_id="run123",
        tags={"model_type": ModelType.XGBOOST},
        source="s3://bucket/model",
        creation_timestamp=1705312200000,
    )

    metadata = loader._build_metadata(version, "models:/spam/2", "best")

    assert metadata["model_uri"] == "models:/spam/2"
    assert metadata["model_version"] == "2"
    assert metadata["run_id"] == "run123"
    assert metadata["artifact_uri"] == "s3://bucket/model"
    assert metadata["strategy"] == "best"
    assert metadata["model_type"] == ModelType.XGBOOST
    assert "registered_at" in metadata


def test_build_metadata_without_tags(loader: ModelLoader) -> None:
    version = FakeModelVersion(tags=None, creation_timestamp=1705312200000)
    metadata = loader._build_metadata(version, "models:/spam/1", "latest")
    assert "model_type" not in metadata


def test_build_metadata_tags_without_model_type(loader: ModelLoader) -> None:
    version = FakeModelVersion(tags={"other": "value"}, creation_timestamp=1705312200000)
    metadata = loader._build_metadata(version, "models:/spam/1", "latest")
    assert "model_type" not in metadata


def test_build_metadata_without_timestamp(loader: ModelLoader) -> None:
    version = FakeModelVersion(creation_timestamp=None)
    metadata = loader._build_metadata(version, "models:/spam/1", "latest")
    assert "registered_at" not in metadata


def test_get_tracking_uri_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow:5000")
    assert ModelLoader._get_tracking_uri() == "http://env-mlflow:5000"


@patch("course_mlops.api.loader.get_settings")
def test_get_tracking_uri_from_settings(mock_get_settings: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mock_get_settings.return_value.mlflow.tracking_uri = "http://settings-mlflow:5000"

    assert ModelLoader._get_tracking_uri() == "http://settings-mlflow:5000"


@pytest.mark.parametrize(
    ("timestamp", "expected_date_fragment"),
    [
        (1705312200000, "2024-01-15"),
        (1705312200, "2024-01-15"),
    ],
)
def test_timestamp_to_iso(timestamp: int, expected_date_fragment: str) -> None:
    result = ModelLoader._timestamp_to_iso(timestamp)
    assert expected_date_fragment in result
