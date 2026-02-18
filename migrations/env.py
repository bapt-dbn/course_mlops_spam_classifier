import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from course_mlops.common.db import Schema
from course_mlops.monitoring.db import PredictionLog  # noqa: F401 — register model

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Schema.metadata

# Override URL from env vars if available
_db_host = os.environ.get("CML_DB_HOST")
if _db_host:
    _db_port = os.environ.get("CML_DB_PORT", "5432")
    _db_user = os.environ.get("CML_DB_USER", "postgres")
    _db_password = os.environ.get("CML_DB_PASSWORD", "")
    _db_name = os.environ.get("CML_DB_NAME", "postgres")
    config.set_main_option(
        "sqlalchemy.url", f"postgresql+psycopg://{_db_user}:{_db_password}@{_db_host}:{_db_port}/{_db_name}"
    )


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
