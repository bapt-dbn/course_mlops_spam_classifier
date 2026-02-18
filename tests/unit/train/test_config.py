from pathlib import Path

import pytest

from course_mlops.train.config import ModelConfig
from course_mlops.train.config import Settings
from course_mlops.train.enums import ModelType
from course_mlops.train.exceptions import ConfigurationError


def test_model_config_default_is_logistic_regression() -> None:
    assert ModelConfig().type == ModelType.LOGISTIC_REGRESSION


def test_settings_from_yaml_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert Settings.from_yaml(tmp_path / "nonexistent.yaml") == Settings()


def test_settings_from_yaml_valid(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("data:\n  test_size: 0.4\n  random_state: 99\nmodel:\n  type: xgboost\n")
    settings = Settings.from_yaml(yaml_file)
    assert settings.data.test_size == pytest.approx(0.4)
    assert settings.data.random_state == 99
    assert settings.model.type == ModelType.XGBOOST


def test_settings_from_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(":\n  - :\n  bad: [unclosed")
    with pytest.raises(ConfigurationError):
        Settings.from_yaml(yaml_file)


def test_settings_from_yaml_empty_file_returns_defaults(tmp_path: Path) -> None:
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    assert Settings.from_yaml(yaml_file) == Settings()
