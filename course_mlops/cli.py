from importlib.metadata import version as get_version
from typing import Annotated

import typer
import uvicorn
from alembic import command
from alembic.config import Config

from course_mlops.logging import set_logging_config
from course_mlops.train.config import Settings
from course_mlops.train.enums import ModelType
from course_mlops.train.models import get_classifier
from course_mlops.train.pipeline import TrainingPipeline

set_logging_config()

app = typer.Typer(
    name="mlops_course",
    help="CLI for MLOps Course - Spam classification model training",
    add_completion=False,
)


@app.command()
def train(
    models: Annotated[
        list[ModelType] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model(s) to train. Can be specified multiple times.",
        ),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Train all available models.",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
        ),
    ] = None,
) -> None:
    model_types = list(ModelType) if all_models else (models or [ModelType.LOGISTIC_REGRESSION])

    typer.echo(f"Models to train: {[m.value for m in model_types]}")

    for model_type in model_types:
        typer.echo(f"\n{'=' * 50}")
        typer.echo(f"Training: {model_type.value}")
        typer.echo("=" * 50)

        settings = Settings.from_yaml(config) if config else Settings.from_yaml()
        if model_type != settings.model.type:
            settings.model.type = model_type
            settings.model.params = get_classifier(model_type).params_class()

        pipeline = TrainingPipeline(settings)
        run_id = pipeline.run()
        typer.echo(f"Run ID: {run_id}")

    typer.echo(f"\n{'=' * 50}")
    typer.echo("Training complete!")
    typer.echo("View results: mlflow ui --port 5000")


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind to.",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to.",
        ),
    ] = 8000,
) -> None:
    typer.echo(f"Starting server on {host}:{port}...")
    uvicorn.run(
        "course_mlops.api.main:app",
        host=host,
        port=port,
    )


@app.command()
def migrate() -> None:
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    typer.echo("Database migrations applied successfully.")


@app.command()
def version() -> None:
    try:
        v = get_version("course-mlops")
        typer.echo(f"mlops_course version {v}")
    except Exception:
        typer.echo("mlops_course version unknown (package not installed)")


if __name__ == "__main__":
    app()
