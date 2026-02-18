from pathlib import Path

import pandas as pd
import pytest

from course_mlops.train.config import DataConfig
from course_mlops.train.exceptions import DataLoadError
from course_mlops.train.exceptions import DataNotFoundError
from course_mlops.train.exceptions import DataValidationError
from course_mlops.train.preprocessing.data import DataProcessor
from course_mlops.train.preprocessing.data import preprocess_message


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello World", "hello world"),
        ("Visit http://example.com now", "visit now"),
        ("Email me@test.com please", "email please"),
        ("Call 12345 today", "call today"),
        ("Hello, world!", "hello world"),
        ("  too   many    spaces  ", "too many spaces"),
        ("", ""),
        (None, ""),
    ],
)
def test_preprocess_message(input_text: str | None, expected: str) -> None:
    assert preprocess_message(input_text) == expected


def test_load_file_not_found_raises(data_config: DataConfig) -> None:
    processor = DataProcessor(data_config)
    with pytest.raises(DataNotFoundError):
        processor.load("/nonexistent/path.csv")


def test_load_valid_csv(tmp_path: Path, data_config: DataConfig) -> None:
    csv_file = tmp_path / "spam.csv"
    csv_file.write_text("label,message\nham,hello\nspam,free money\n")
    df = DataProcessor(data_config).load(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ["label", "message"]


def test_load_missing_columns_raises(tmp_path: Path, data_config: DataConfig) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("col_a,col_b\n1,2\n")
    with pytest.raises(DataValidationError):
        DataProcessor(data_config).load(csv_file)


def test_load_invalid_csv_raises(tmp_path: Path, data_config: DataConfig) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises((DataLoadError, DataValidationError)):
        DataProcessor(data_config).load(csv_file)


def test_preprocess_drops_nan(data_config: DataConfig) -> None:
    df = pd.DataFrame({"label": ["ham", "spam", None, "ham"], "message": ["hi", None, "hello", "bye"]})
    result = DataProcessor(data_config).preprocess(df)
    assert len(result) == 2
    assert result["message"].isna().sum() == 0


def test_preprocess_copies_dataframe(data_config: DataConfig, sample_df: pd.DataFrame) -> None:
    result = DataProcessor(data_config).preprocess(sample_df)
    assert result is not sample_df


def test_split_returns_correct_proportions(data_config: DataConfig, sample_df: pd.DataFrame) -> None:
    train, test = DataProcessor(data_config).split(sample_df)
    expected_test = int(len(sample_df) * data_config.test_size)
    assert len(test) == pytest.approx(expected_test, abs=1)
    assert len(train) + len(test) == len(sample_df)


def test_split_preserves_stratification(sample_df: pd.DataFrame) -> None:
    config = DataConfig(test_size=0.5, random_state=42)
    train, test = DataProcessor(config).split(sample_df)
    train_ratio = (train["label"] == "spam").mean()
    test_ratio = (test["label"] == "spam").mean()
    assert train_ratio == pytest.approx(test_ratio, abs=0.15)
