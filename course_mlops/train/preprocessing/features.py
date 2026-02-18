import re
from typing import Self

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from course_mlops.train.config import FeaturesConfig
from course_mlops.train.enums import NumericalFeature
from course_mlops.train.exceptions import FeatureExtractionError

# Special characters used for spam detection (matches NumericalFeature.SPECIAL_CHARS_COUNT)
SPECIAL_CHARS_PATTERN = re.compile(r"[!?$€£%]")
_UPPERCASE_PATTERN = re.compile(r"[A-Z]")


def compute_numerical_features(text: str) -> dict[str, float]:
    text_len = len(text)
    return {
        NumericalFeature.TEXT_LENGTH: float(text_len),
        NumericalFeature.WORD_COUNT: float(len(text.split())),
        NumericalFeature.CAPS_RATIO: float(len(_UPPERCASE_PATTERN.findall(text)) / max(text_len, 1)),
        NumericalFeature.SPECIAL_CHARS_COUNT: float(len(SPECIAL_CHARS_PATTERN.findall(text))),
    }


class FeatureEngineer:
    def __init__(self: Self, config: FeaturesConfig) -> None:
        self.config = config
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None

    def fit(self: Self, texts: list[str]) -> Self:
        if not texts:
            raise FeatureExtractionError("Cannot fit on empty text list")

        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf.max_features,
            ngram_range=self.config.tfidf.ngram_range,
            min_df=self.config.tfidf.min_df,
            max_df=self.config.tfidf.max_df,
            stop_words=self.config.tfidf.stop_words,
        )
        self.vectorizer.fit(texts)

        X_numerical = self._extract_numerical(texts)
        self.scaler = StandardScaler()
        self.scaler.fit(X_numerical)

        return self

    def transform(self: Self, texts: list[str]) -> spmatrix:
        """Transform texts to combined feature matrix."""
        if self.vectorizer is None or self.scaler is None:
            raise FeatureExtractionError("FeatureEngineer must be fitted before transform")

        X_tfidf = self.vectorizer.transform(texts)
        X_numerical = self._extract_numerical(texts)
        X_numerical_scaled = self.scaler.transform(X_numerical)

        return hstack([X_tfidf, csr_matrix(X_numerical_scaled)])

    def _extract_numerical(self: Self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise FeatureExtractionError("Cannot extract features from empty text list")

        result = np.empty((len(texts), len(NumericalFeature)), dtype=np.float64)

        for i, text in enumerate(texts):
            features = compute_numerical_features(text)
            for j, key in enumerate(NumericalFeature):
                result[i, j] = features[key]

        return result

    @property
    def vocabulary_size(self: Self) -> int:
        if self.vectorizer is None:
            raise FeatureExtractionError("FeatureEngineer must be fitted first")
        return len(self.vectorizer.vocabulary_)

    @property
    def numerical_features_count(self: Self) -> int:
        return len(NumericalFeature)
