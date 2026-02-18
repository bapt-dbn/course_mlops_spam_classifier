import pytest
from mlflow.tracking import MlflowClient

from course_mlops.train.config import Settings
from course_mlops.train.pipeline import TrainingPipeline

from .conftest import MODEL_NAME


@pytest.fixture(scope="session")
def training_run_id(mlflow_client: MlflowClient) -> str:
    pipeline = TrainingPipeline(Settings())
    return pipeline.run()


@pytest.mark.integration
def test_training_pipeline_registers_model(training_run_id: str, mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")

    assert len(versions) >= 1
    assert any(v.run_id == training_run_id for v in versions)


@pytest.mark.integration
def test_training_pipeline_logs_metrics(training_run_id: str, mlflow_client: MlflowClient) -> None:
    metrics = mlflow_client.get_run(training_run_id).data.metrics

    assert metrics["accuracy"] > 0.8
    assert metrics["f1"] > 0.5
    assert metrics["precision"] > 0
    assert metrics["recall"] > 0
    assert metrics["roc_auc"] > 0.5


@pytest.mark.integration
def test_training_pipeline_logs_params(training_run_id: str, mlflow_client: MlflowClient) -> None:
    params = mlflow_client.get_run(training_run_id).data.params

    assert "model.type" in params
    assert "data.test_size" in params
    assert "features.tfidf.max_features" in params


@pytest.mark.integration
def test_training_pipeline_tags_model_type(training_run_id: str, mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))

    assert latest.tags["model_type"] == "logistic_regression"
