import logging
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from scipy.sparse import spmatrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder

from course_mlops.monitoring.reference import build_reference_dataset
from course_mlops.monitoring.reference import save_reference
from course_mlops.train.config import Settings
from course_mlops.train.enums import ModelType
from course_mlops.train.exceptions import ModelNotFittedError
from course_mlops.train.models import get_classifier
from course_mlops.train.models.base import BaseClassifier
from course_mlops.train.preprocessing.data import DataProcessor
from course_mlops.train.preprocessing.data import DatasetColumn
from course_mlops.train.preprocessing.data import preprocess_message
from course_mlops.train.preprocessing.features import FeatureEngineer
from course_mlops.train.reporting.evaluation import EvaluationMetrics
from course_mlops.train.reporting.evaluation import evaluate_model
from course_mlops.train.reporting.plots import plot_confusion_matrix
from course_mlops.train.reporting.plots import plot_learning_curve
from course_mlops.train.reporting.plots import plot_roc_curve

logger = logging.getLogger(__name__)


class SpamClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        model: BaseClassifier,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.model = model

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        texts = model_input[DatasetColumn.MESSAGE].tolist()
        texts_clean = [preprocess_message(t) for t in texts]
        X = self.feature_engineer.transform(texts_clean)

        return {
            "predictions": self.model.predict(X),
            "probabilities": self.model.predict_proba(X),
        }


class TrainingPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_processor = DataProcessor(settings.data)
        self.feature_engineer = FeatureEngineer(settings.features)
        self.model: BaseClassifier | None = None
        self.label_encoder = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        logger.info("Loading data...")
        df = self.data_processor.load()
        mlflow.log_metric("data.raw_samples", len(df))

        logger.info("Preprocessing...")
        df = self.data_processor.preprocess(df)
        mlflow.log_metric("data.clean_samples", len(df))

        for label, count in df[DatasetColumn.LABEL].value_counts().to_dict().items():
            mlflow.log_metric(f"data.class_{label}", count)

        return df

    def build_features(self, df: pd.DataFrame) -> tuple[spmatrix, np.ndarray]:
        logger.info("Fitting feature engineer...")
        texts = [preprocess_message(t) for t in df[DatasetColumn.MESSAGE].tolist()]
        self.feature_engineer.fit(texts)
        X = self.feature_engineer.transform(texts)
        y = self.label_encoder.fit_transform(df[DatasetColumn.LABEL].values)

        mlflow.log_metric("features.vocabulary_size", self.feature_engineer.vocabulary_size)
        mlflow.log_metric("features.numerical_features_count", self.feature_engineer.numerical_features_count)

        return X, y

    def transform_features(self, df: pd.DataFrame) -> tuple[spmatrix, np.ndarray]:
        texts = [preprocess_message(t) for t in df[DatasetColumn.MESSAGE].tolist()]
        X = self.feature_engineer.transform(texts)
        y = self.label_encoder.transform(df[DatasetColumn.LABEL].values)
        return X, y

    def train(self, X_train: spmatrix, y_train: np.ndarray) -> BaseClassifier:
        logger.info(f"Training model: {self.settings.model.type}...")
        classifier_cls = get_classifier(self.settings.model.type)
        self.model = classifier_cls(params=self.settings.model.params)  # type: ignore[call-arg]
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: spmatrix, y_test: np.ndarray) -> EvaluationMetrics:
        if self.model is None:
            raise ModelNotFittedError("Model must be trained before evaluation")
        logger.info("Evaluating...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_metrics(metrics.to_dict())
        logger.info(f"Accuracy: {metrics.accuracy:.4f} | F1: {metrics.f1:.4f} | ROC-AUC: {metrics.roc_auc:.4f}")

        fig_cm = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")

        spam_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        fig_roc = plot_roc_curve(y_test, spam_proba)
        mlflow.log_figure(fig_roc, "plots/roc_curve.png")

        return metrics

    def compute_learning_curve(
        self,
        X: spmatrix,
        y: np.ndarray,
    ) -> None:
        logger.info("Computing learning curve...")

        estimator = self.settings.model.params.to_sklearn()

        lc_config = self.settings.learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            cv=lc_config.cv,
            n_jobs=lc_config.n_jobs,
            train_sizes=np.linspace(
                lc_config.train_sizes_start,
                lc_config.train_sizes_end,
                lc_config.train_sizes_steps,
            ),
            scoring=lc_config.scoring,
        )

        fig = plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            title=f"Learning Curve ({self.settings.model.type.value})",
        )
        mlflow.log_figure(fig, "plots/learning_curve.png")
        logger.info("Learning curve logged")

    def save_reference_dataset(self, df_test: pd.DataFrame, X_test: spmatrix, y_test: np.ndarray) -> None:
        if self.model is None:
            raise ModelNotFittedError("Model must be trained before saving reference dataset")
        logger.info("Saving reference dataset...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        texts = [preprocess_message(t) for t in df_test[DatasetColumn.MESSAGE].tolist()]
        numerical_features = self.feature_engineer._extract_numerical(texts)

        reference_df = build_reference_dataset(numerical_features, y_pred, y_proba)
        save_reference(reference_df)
        logger.info("Reference dataset saved")

    def log_model(self, run_id: str) -> None:
        logger.info("Logging model to MLflow...")
        if self.model is None:
            raise ModelNotFittedError("Model must be trained before logging")
        if self.feature_engineer.vectorizer is None:
            raise ModelNotFittedError("FeatureEngineer must be fitted before logging")
        wrapped_model = SpamClassifierWrapper(self.feature_engineer, self.model)

        example_input = pd.DataFrame({DatasetColumn.MESSAGE: ["Free money click here!"]})
        example_output = wrapped_model.predict(None, example_input)
        signature = infer_signature(example_input, example_output)

        pip_requirements = ["scikit-learn>=1.5.0", "pandas>=2.2.0"]
        if self.settings.model.type == ModelType.XGBOOST:
            pip_requirements.append("xgboost>=2.0.0")

        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapped_model,
            signature=signature,
            input_example=example_input,
            pip_requirements=pip_requirements,
        )

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=self.settings.mlflow.registered_model_name,
            tags={"model_type": self.settings.model.type.value},
        )
        logger.info(f"Model registered: {registered_model.name} v{registered_model.version}")

    def log_params(self) -> None:
        mlflow.log_params(
            {
                "data.path": self.settings.data.path,
                "data.test_size": self.settings.data.test_size,
                "data.random_state": self.settings.data.random_state,
                "features.tfidf.max_features": self.settings.features.tfidf.max_features,
                "features.tfidf.ngram_range": str(self.settings.features.tfidf.ngram_range),
                "features.tfidf.min_df": self.settings.features.tfidf.min_df,
                "model.type": self.settings.model.type,
            }
        )
        model_params = {f"model.{k}": v for k, v in self.settings.model.params.model_dump().items()}
        mlflow.log_params(model_params)

    def run(self) -> str:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.settings.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.settings.mlflow.experiment_name)

        if self.settings.model.type == ModelType.XGBOOST:
            mlflow.xgboost.autolog(log_models=False)
        else:
            mlflow.sklearn.autolog(log_models=False)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_id}")

            mlflow.set_tag("model_type", self.settings.model.type.value)
            mlflow.set_tag("dataset", self.settings.data.path)

            self.log_params()

            df = self.load_data()

            df_train, df_test = self.data_processor.split(df)
            mlflow.log_metric("data.train_samples", len(df_train))
            mlflow.log_metric("data.test_samples", len(df_test))

            X_train, y_train = self.build_features(df_train)
            X_test, y_test = self.transform_features(df_test)

            self.train(X_train, y_train)
            self.evaluate(X_test, y_test)
            self.save_reference_dataset(df_test, X_test, y_test)
            self.compute_learning_curve(X_train, y_train)
            self.log_model(run_id)

            logger.info(f"Training complete! Run ID: {run_id}")

            return str(run_id)
