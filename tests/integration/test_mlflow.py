import pytest
from mlflow.tracking import MlflowClient

from .conftest import MODEL_NAME


@pytest.mark.integration
def test_experiments_exist(mlflow_client: MlflowClient) -> None:
    experiments = mlflow_client.search_experiments()

    assert len(experiments) >= 1


@pytest.mark.integration
def test_model_is_registered(mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")

    assert len(versions) >= 1


@pytest.mark.integration
def test_model_version_has_run_id(mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")

    assert versions[0].run_id is not None


@pytest.mark.integration
def test_run_has_metrics(mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    run = mlflow_client.get_run(versions[0].run_id)
    metrics = run.data.metrics

    assert "f1" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["f1"] > 0
    assert metrics["accuracy"] > 0


@pytest.mark.integration
def test_run_has_parameters(mlflow_client: MlflowClient) -> None:
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    run = mlflow_client.get_run(versions[0].run_id)

    assert len(run.data.params) > 0
