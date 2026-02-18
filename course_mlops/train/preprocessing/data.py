import re
import string
from enum import StrEnum
from enum import auto
from pathlib import Path
from typing import Self

import pandas as pd
from sklearn.model_selection import train_test_split

from course_mlops.train.config import DataConfig
from course_mlops.train.exceptions import DataLoadError
from course_mlops.train.exceptions import DataNotFoundError
from course_mlops.train.exceptions import DataValidationError

_URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_DIGITS_PATTERN = re.compile(r"\d+")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


class DatasetColumn(StrEnum):
    LABEL = auto()
    MESSAGE = auto()


def preprocess_message(text: str | None) -> str:
    if not text:
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub("", text)
    text = _EMAIL_PATTERN.sub("", text)
    text = _DIGITS_PATTERN.sub("", text)
    text = text.translate(_PUNCTUATION_TABLE)
    return " ".join(text.split())


class DataProcessor:
    def __init__(self: Self, config: DataConfig) -> None:
        self.config = config

    def load(self: Self, path: str | Path | None = None) -> pd.DataFrame:
        path = Path(path or self.config.path)

        if not path.exists():
            raise DataNotFoundError(f"Data file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load data from {path}: {e}") from e

        required_columns = {col.value for col in DatasetColumn}
        if not required_columns.issubset(df.columns):
            raise DataValidationError(f"Dataset must contain columns: {required_columns}. Found: {list(df.columns)}")

        return df

    def preprocess(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return df.dropna(subset=[DatasetColumn.MESSAGE, DatasetColumn.LABEL])

    def split(
        self: Self,
        df: pd.DataFrame,
        stratify_column: str = DatasetColumn.LABEL,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(  # type: ignore[no-any-return]
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[stratify_column],
        )
