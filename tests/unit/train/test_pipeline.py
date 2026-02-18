from unittest.mock import DEFAULT
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from course_mlops.train.config import Settings
from course_mlops.train.enums import ModelType
from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.pipeline import SpamClassifierWrapper
from course_mlops.train.pipeline import TrainingPipeline
from course_mlops.train.preprocessing.data import DatasetColumn
from course_mlops.train.reporting.evaluation import EvaluationMetrics


def test_wrapper_predict_returns_dict() -> None:
    feature_engineer = Mock()
    model = Mock()
    feature_engineer.transform.return_value = csr_matrix(np.eye(2))
    model.predict.return_value = np.array([0, 1])
    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    result = SpamClassifierWrapper(feature_engineer, model).predict(
        None, pd.DataFrame({DatasetColumn.MESSAGE: ["hello", "free money"]})
    )
    assert "predictions" in result
    assert "probabilities" in result


def test_wrapper_predict_preprocesses() -> None:
    feature_engineer = Mock()
    model = Mock()
    feature_engineer.transform.return_value = csr_matrix(np.eye(1))
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.9, 0.1]])

    SpamClassifierWrapper(feature_engineer, model).predict(
        None, pd.DataFrame({DatasetColumn.MESSAGE: ["HELLO WORLD!!!"]})
    )
    call_args = feature_engineer.transform.call_args[0][0]
    assert call_args[0] == "hello world"


def test_pipeline_init() -> None:
    pipeline = TrainingPipeline(Settings())
    assert pipeline.model is None
    assert pipeline.data_processor is not None
    assert pipeline.feature_engineer is not None


def test_load_data(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["hi", "free"]})
    pipeline.data_processor = Mock()
    pipeline.data_processor.load.return_value = df
    pipeline.data_processor.preprocess.return_value = df

    result = pipeline.load_data()

    pipeline.data_processor.load.assert_called_once()
    pipeline.data_processor.preprocess.assert_called_once_with(df)
    mock_mlflow.log_metric.assert_any_call("data.raw_samples", 2)
    assert len(result) == 2


def test_build_features(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["hi there", "free money"]})
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.fit.return_value = pipeline.feature_engineer
    pipeline.feature_engineer.transform.return_value = csr_matrix(np.eye(2))
    pipeline.feature_engineer.vocabulary_size = 10
    pipeline.feature_engineer.numerical_features_count = 4

    X, y = pipeline.build_features(df)

    pipeline.feature_engineer.fit.assert_called_once()
    pipeline.feature_engineer.transform.assert_called_once()
    assert X.shape == (2, 2)
    assert len(y) == 2


def test_transform_features() -> None:
    pipeline = TrainingPipeline(Settings())
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["hi there", "free money"]})
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.transform.return_value = csr_matrix(np.eye(2))
    pipeline.label_encoder.fit(["ham", "spam"])

    X, _y = pipeline.transform_features(df)

    pipeline.feature_engineer.transform.assert_called_once()
    assert X.shape == (2, 2)


@patch("course_mlops.train.pipeline.get_classifier")
def test_train(mock_get_classifier: Mock) -> None:
    pipeline = TrainingPipeline(Settings())
    mock_cls = Mock()
    mock_model = Mock()
    mock_cls.return_value = mock_model
    mock_get_classifier.return_value = mock_cls

    X = csr_matrix(np.eye(4))
    y = np.array([0, 1, 0, 1])
    pipeline.train(X, y)

    mock_get_classifier.assert_called_once()
    mock_model.fit.assert_called_once_with(X, y)


def test_evaluate(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    pipeline.model = Mock()
    pipeline.model.predict.return_value = np.array([0, 1])
    pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    with (
        patch("course_mlops.train.pipeline.evaluate_model") as mock_eval,
        patch("course_mlops.train.pipeline.plot_confusion_matrix") as mock_cm,
        patch("course_mlops.train.pipeline.plot_roc_curve") as mock_roc,
    ):
        mock_eval.return_value = EvaluationMetrics(accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=1.0)
        mock_cm.return_value = Mock()
        mock_roc.return_value = Mock()
        metrics = pipeline.evaluate(csr_matrix(np.eye(2)), np.array([0, 1]))

    mock_mlflow.log_metrics.assert_called_once()
    assert metrics.accuracy == pytest.approx(1.0)


def test_evaluate_before_training_raises() -> None:
    pipeline = TrainingPipeline(Settings())
    with pytest.raises(ModelNotFittedError):
        pipeline.evaluate(csr_matrix(np.eye(2)), np.array([0, 1]))


def test_log_model(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    pipeline.model = Mock()
    pipeline.model.predict.return_value = np.array([0])
    pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1]])
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.vectorizer = Mock()
    pipeline.feature_engineer.transform.return_value = csr_matrix(np.eye(1))

    pipeline.log_model("test-run-id")

    mock_mlflow.pyfunc.log_model.assert_called_once()
    mock_mlflow.register_model.assert_called_once()


def test_log_model_before_training_raises() -> None:
    with pytest.raises(ModelNotFittedError):
        TrainingPipeline(Settings()).log_model("test-run-id")


def test_log_params(mock_mlflow: MagicMock) -> None:
    TrainingPipeline(Settings()).log_params()
    assert mock_mlflow.log_params.call_count == 2


def test_run_returns_run_id(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())

    train_df = pd.DataFrame({"label": ["ham", "spam"], "message": ["a", "b"]})
    test_df = pd.DataFrame({"label": ["ham", "spam"], "message": ["c", "d"]})
    features = (csr_matrix(np.eye(2)), np.array([0, 1]))

    with patch.multiple(
        pipeline,
        log_params=DEFAULT,
        load_data=DEFAULT,
        data_processor=DEFAULT,
        build_features=DEFAULT,
        transform_features=DEFAULT,
        train=DEFAULT,
        evaluate=DEFAULT,
        save_reference_dataset=DEFAULT,
        compute_learning_curve=DEFAULT,
        log_model=DEFAULT,
    ) as mocks:
        mocks["load_data"].return_value = pd.concat([train_df, test_df])
        mocks["data_processor"].split.return_value = (train_df, test_df)
        mocks["build_features"].return_value = features
        mocks["transform_features"].return_value = features

        assert pipeline.run() == "test-run-id"


def test_log_model_feature_engineer_not_fitted(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    pipeline.model = Mock()
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.vectorizer = None

    with pytest.raises(ModelNotFittedError, match="FeatureEngineer must be fitted"):
        pipeline.log_model("test-run-id")


def test_log_model_adds_xgboost_pip_requirement(mock_mlflow: MagicMock) -> None:
    settings = Settings()
    settings.model.type = ModelType.XGBOOST
    pipeline = TrainingPipeline(settings)
    pipeline.model = Mock()
    pipeline.model.predict.return_value = np.array([0])
    pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1]])
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.vectorizer = Mock()
    pipeline.feature_engineer.transform.return_value = csr_matrix(np.eye(1))

    pipeline.log_model("test-run-id")

    call_kwargs = mock_mlflow.pyfunc.log_model.call_args
    pip_reqs = call_kwargs.kwargs.get("pip_requirements") or call_kwargs[1].get("pip_requirements")
    assert any("xgboost" in r for r in pip_reqs)


def test_compute_learning_curve(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())

    with (
        patch("course_mlops.train.pipeline.learning_curve") as mock_lc,
        patch("course_mlops.train.pipeline.plot_learning_curve") as mock_plot,
    ):
        mock_lc.return_value = (np.array([10, 20]), np.array([[0.8], [0.9]]), np.array([[0.7], [0.8]]))
        mock_plot.return_value = Mock()

        pipeline.compute_learning_curve(csr_matrix(np.eye(4)), np.array([0, 1, 0, 1]))

    mock_lc.assert_called_once()
    mock_plot.assert_called_once()
    mock_mlflow.log_figure.assert_called_once()


def test_save_reference_dataset(mock_mlflow: MagicMock) -> None:
    pipeline = TrainingPipeline(Settings())
    pipeline.model = Mock()
    pipeline.model.predict.return_value = np.array([0, 1])
    pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer._extract_numerical.return_value = np.array([[10.0, 2.0, 0.1, 1.0], [20.0, 4.0, 0.2, 2.0]])

    df_test = pd.DataFrame({"label": ["ham", "spam"], "message": ["hello", "free money"]})

    with (
        patch("course_mlops.train.pipeline.build_reference_dataset") as mock_build,
        patch("course_mlops.train.pipeline.save_reference") as mock_save,
    ):
        mock_build.return_value = pd.DataFrame()
        pipeline.save_reference_dataset(df_test, csr_matrix(np.eye(2)), np.array([0, 1]))

    mock_build.assert_called_once()
    mock_save.assert_called_once()


def test_save_reference_dataset_before_training_raises() -> None:
    pipeline = TrainingPipeline(Settings())
    with pytest.raises(ModelNotFittedError):
        pipeline.save_reference_dataset(pd.DataFrame(), csr_matrix(np.eye(2)), np.array([0, 1]))
