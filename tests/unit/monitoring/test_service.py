from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from course_mlops.monitoring.service import log_prediction_to_store


@pytest.fixture
def mock_dal() -> Generator[AsyncMock, None, None]:
    with patch("course_mlops.monitoring.service.dal") as mock:
        mock.log_prediction = AsyncMock()
        yield mock


async def test_log_prediction_to_store_success(mock_dal: AsyncMock) -> None:
    await log_prediction_to_store("1", "free money", "spam", 0.95, "high")

    mock_dal.log_prediction.assert_awaited_once()
    record = mock_dal.log_prediction.call_args[0][0]
    assert record.prediction == "spam"
    assert record.model_version == "1"


async def test_log_prediction_to_store_defaults_model_version(mock_dal: AsyncMock) -> None:
    await log_prediction_to_store(None, "hello", "ham", 0.1, "high")

    record = mock_dal.log_prediction.call_args[0][0]
    assert record.model_version == "unknown"


async def test_log_prediction_to_store_swallows_exceptions(mock_dal: AsyncMock) -> None:
    mock_dal.log_prediction = AsyncMock(side_effect=RuntimeError("DB down"))
    await log_prediction_to_store("1", "hello", "ham", 0.1, "high")
