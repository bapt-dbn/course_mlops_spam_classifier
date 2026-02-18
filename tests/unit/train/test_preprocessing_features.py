import pytest
from scipy.sparse import spmatrix

from course_mlops.train.config import FeaturesConfig
from course_mlops.train.enums import NumericalFeature
from course_mlops.train.exceptions import FeatureExtractionError
from course_mlops.train.preprocessing.features import FeatureEngineer
from course_mlops.train.preprocessing.features import compute_numerical_features


@pytest.mark.parametrize(
    ("text", "expected_length", "expected_words"),
    [
        ("hello world", 11, 2),
        ("a", 1, 1),
        ("one two three four", 18, 4),
    ],
)
def test_extract_numerical_length_and_word_count(
    features_config: FeaturesConfig, text: str, expected_length: int, expected_words: int
) -> None:
    result = FeatureEngineer(features_config)._extract_numerical([text])
    assert result[0, 0] == expected_length
    assert result[0, 1] == expected_words


def test_extract_numerical_caps_ratio(features_config: FeaturesConfig) -> None:
    result = FeatureEngineer(features_config)._extract_numerical(["HeLLo"])
    assert result[0, 2] == pytest.approx(3 / 5)


def test_extract_numerical_special_chars_count(features_config: FeaturesConfig) -> None:
    result = FeatureEngineer(features_config)._extract_numerical(["Buy now!!! $$$"])
    assert result[0, 3] == 6


def test_extract_numerical_empty_raises(features_config: FeaturesConfig) -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureEngineer(features_config)._extract_numerical([])


def test_fit_creates_vectorizer_and_scaler(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    fe = FeatureEngineer(features_config)
    fe.fit(sample_texts)
    assert fe.vectorizer is not None
    assert fe.scaler is not None


def test_fit_empty_raises(features_config: FeaturesConfig) -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureEngineer(features_config).fit([])


def test_fit_returns_self(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    fe = FeatureEngineer(features_config)
    assert fe.fit(sample_texts) is fe


def test_transform_before_fit_raises(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    with pytest.raises(FeatureExtractionError):
        FeatureEngineer(features_config).transform(sample_texts)


def test_transform_returns_sparse(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    fe = FeatureEngineer(features_config)
    fe.fit(sample_texts)
    assert isinstance(fe.transform(sample_texts), spmatrix)


def test_transform_shape(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    fe = FeatureEngineer(features_config)
    fe.fit(sample_texts)
    result = fe.transform(sample_texts)
    assert result.shape[0] == len(sample_texts)
    assert result.shape[1] == fe.vocabulary_size + len(NumericalFeature)


def test_vocabulary_size_before_fit_raises(features_config: FeaturesConfig) -> None:
    with pytest.raises(FeatureExtractionError):
        _ = FeatureEngineer(features_config).vocabulary_size


def test_vocabulary_size_after_fit(sample_texts: list[str], features_config: FeaturesConfig) -> None:
    fe = FeatureEngineer(features_config)
    fe.fit(sample_texts)
    assert fe.vocabulary_size > 0


def test_compute_numerical_features_keys() -> None:
    result = compute_numerical_features("hello world")
    assert set(result) == {f.value for f in NumericalFeature}


def test_compute_numerical_features_values() -> None:
    result = compute_numerical_features("Hello WORLD!")
    assert result[NumericalFeature.TEXT_LENGTH] == pytest.approx(12.0)
    assert result[NumericalFeature.WORD_COUNT] == pytest.approx(2.0)
    assert result[NumericalFeature.CAPS_RATIO] == pytest.approx(6 / 12)
    assert result[NumericalFeature.SPECIAL_CHARS_COUNT] == pytest.approx(1.0)


def test_compute_numerical_features_empty_string() -> None:
    result = compute_numerical_features("")
    assert result[NumericalFeature.TEXT_LENGTH] == pytest.approx(0.0)
    assert result[NumericalFeature.WORD_COUNT] == pytest.approx(0.0)
    assert result[NumericalFeature.CAPS_RATIO] == pytest.approx(0.0)
    assert result[NumericalFeature.SPECIAL_CHARS_COUNT] == pytest.approx(0.0)
