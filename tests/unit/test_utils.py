import pytest

from course_mlops.utils import EnvironmentVariable


def test_read_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CML_LOG_LEVEL", "DEBUG")
    assert EnvironmentVariable.LOG_LEVEL.read() == "DEBUG"


def test_read_returns_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CML_LOG_LEVEL", raising=False)
    assert EnvironmentVariable.LOG_LEVEL.read(default="INFO") == "INFO"


def test_read_raises_when_missing_and_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CML_LOG_LEVEL", raising=False)
    with pytest.raises(ValueError, match="CML_LOG_LEVEL"):
        EnvironmentVariable.LOG_LEVEL.read()
