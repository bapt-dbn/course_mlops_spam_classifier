from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from course_mlops.cli import app
from course_mlops.train.enums import ModelType


@pytest.fixture(name="runner")
def runner_fixture() -> CliRunner:
    return CliRunner()


def test_version_command_found(runner: CliRunner) -> None:
    with patch("course_mlops.cli.get_version", return_value="1.2.3"):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "mlops_course version 1.2.3" in result.stdout


def test_version_command_not_found(runner: CliRunner) -> None:
    with patch("course_mlops.cli.get_version", side_effect=PackageNotFoundError):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "mlops_course version unknown (package not installed)" in result.stdout


@patch("course_mlops.cli.TrainingPipeline")
@patch("course_mlops.cli.Settings.from_yaml")
@patch("course_mlops.cli.typer.echo")
def test_train_command_default(
    mock_typer_echo: MagicMock,
    mock_settings_from_yaml: MagicMock,
    mock_training_pipeline: MagicMock,
    runner: CliRunner,
) -> None:
    mock_settings_instance = MagicMock()
    mock_settings_from_yaml.return_value = mock_settings_instance
    mock_training_pipeline_instance = MagicMock()
    mock_training_pipeline_instance.run.return_value = "mock_run_id"
    mock_training_pipeline.return_value = mock_training_pipeline_instance

    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    mock_typer_echo.assert_any_call(f"Models to train: {[ModelType.LOGISTIC_REGRESSION.value]}")
    mock_settings_from_yaml.assert_called_once_with()
    mock_training_pipeline.assert_called_once_with(mock_settings_instance)
    mock_training_pipeline_instance.run.assert_called_once()
    mock_typer_echo.assert_any_call("Run ID: mock_run_id")
    mock_typer_echo.assert_any_call("\n==================================================")
    mock_typer_echo.assert_any_call("Training complete!")


@patch("course_mlops.cli.TrainingPipeline")
@patch("course_mlops.cli.Settings.from_yaml")
@patch("course_mlops.cli.typer.echo")
def test_train_command_specific_models(
    mock_typer_echo: MagicMock,
    mock_settings_from_yaml: MagicMock,
    mock_training_pipeline: MagicMock,
    runner: CliRunner,
) -> None:
    mock_settings_instance = MagicMock()
    mock_settings_from_yaml.return_value = mock_settings_instance
    mock_training_pipeline_instance = MagicMock()
    mock_training_pipeline_instance.run.return_value = "mock_run_id"
    mock_training_pipeline.return_value = mock_training_pipeline_instance

    result = runner.invoke(app, ["train", "--model", "logistic_regression", "-m", "xgboost"])

    assert result.exit_code == 0
    mock_typer_echo.assert_any_call(
        f"Models to train: {[ModelType.LOGISTIC_REGRESSION.value, ModelType.XGBOOST.value]}"
    )
    assert mock_training_pipeline.call_count == 2
    assert mock_training_pipeline_instance.run.call_count == 2


@patch("course_mlops.cli.TrainingPipeline")
@patch("course_mlops.cli.Settings.from_yaml")
@patch("course_mlops.cli.typer.echo")
def test_train_command_all_models(
    mock_typer_echo: MagicMock,
    mock_settings_from_yaml: MagicMock,
    mock_training_pipeline: MagicMock,
    runner: CliRunner,
) -> None:
    mock_settings_instance = MagicMock()
    mock_settings_from_yaml.return_value = mock_settings_instance
    mock_training_pipeline_instance = MagicMock()
    mock_training_pipeline_instance.run.return_value = "mock_run_id"
    mock_training_pipeline.return_value = mock_training_pipeline_instance

    result = runner.invoke(app, ["train", "--all"])

    assert result.exit_code == 0
    expected_models = [m.value for m in ModelType]
    mock_typer_echo.assert_any_call(f"Models to train: {expected_models}")
    assert mock_training_pipeline.call_count == len(ModelType)
    assert mock_training_pipeline_instance.run.call_count == len(ModelType)


@patch("course_mlops.cli.TrainingPipeline")
@patch("course_mlops.cli.Settings.from_yaml")
@patch("course_mlops.cli.typer.echo")
def test_train_command_config_file(
    mock_typer_echo: MagicMock,
    mock_settings_from_yaml: MagicMock,
    mock_training_pipeline: MagicMock,
    runner: CliRunner,
) -> None:
    mock_settings_instance = MagicMock()
    mock_settings_from_yaml.return_value = mock_settings_instance
    mock_training_pipeline_instance = MagicMock()
    mock_training_pipeline_instance.run.return_value = "mock_run_id"
    mock_training_pipeline.return_value = mock_training_pipeline_instance

    config_path = "path/to/config.yaml"
    result = runner.invoke(app, ["train", "--config", config_path])

    assert result.exit_code == 0
    mock_settings_from_yaml.assert_called_once_with(config_path)


@patch("course_mlops.cli.get_classifier")
@patch("course_mlops.cli.TrainingPipeline")
@patch("course_mlops.cli.Settings.from_yaml")
@patch("course_mlops.cli.typer.echo")
def test_train_command_model_type_in_config_overridden(
    mock_typer_echo: MagicMock,
    mock_settings_from_yaml: MagicMock,
    mock_training_pipeline: MagicMock,
    mock_get_classifier: MagicMock,
    runner: CliRunner,
) -> None:
    mock_settings_instance = MagicMock()
    mock_settings_instance.model.type = ModelType.XGBOOST  # Config says XGBOOST
    mock_settings_instance.model.params = MagicMock()
    mock_settings_from_yaml.return_value = mock_settings_instance

    mock_training_pipeline_instance = MagicMock()
    mock_training_pipeline_instance.run.return_value = "mock_run_id"
    mock_training_pipeline.return_value = mock_training_pipeline_instance

    mock_classifier_class = MagicMock()
    mock_classifier_class.params_class.return_value = MagicMock()
    mock_get_classifier.return_value = mock_classifier_class

    result = runner.invoke(app, ["train", "--model", "logistic_regression"])

    assert result.exit_code == 0
    assert mock_settings_instance.model.type == ModelType.LOGISTIC_REGRESSION
    mock_get_classifier.assert_called_once_with(ModelType.LOGISTIC_REGRESSION)
    mock_classifier_class.params_class.assert_called_once()
    assert mock_settings_instance.model.params == mock_classifier_class.params_class.return_value


@patch("course_mlops.cli.uvicorn.run")
@patch("course_mlops.cli.typer.echo")
def test_serve_command_default(
    mock_typer_echo: MagicMock,
    mock_uvicorn_run: MagicMock,
    runner: CliRunner,
) -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    mock_typer_echo.assert_called_once_with("Starting server on 0.0.0.0:8000...")
    mock_uvicorn_run.assert_called_once_with(
        "course_mlops.api.main:app",
        host="0.0.0.0",
        port=8000,
    )


@patch("course_mlops.cli.uvicorn.run")
@patch("course_mlops.cli.typer.echo")
def test_serve_command_custom_host_port(
    mock_typer_echo: MagicMock,
    mock_uvicorn_run: MagicMock,
    runner: CliRunner,
) -> None:
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "-p", "5000"])
    assert result.exit_code == 0
    mock_typer_echo.assert_called_once_with("Starting server on 127.0.0.1:5000...")
    mock_uvicorn_run.assert_called_once_with(
        "course_mlops.api.main:app",
        host="127.0.0.1",
        port=5000,
    )
