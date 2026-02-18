"""Initial prediction_logs table.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("message_hash", sa.String(length=64), nullable=True),
        sa.Column("text_length", sa.Float(), nullable=True),
        sa.Column("word_count", sa.Float(), nullable=True),
        sa.Column("caps_ratio", sa.Float(), nullable=True),
        sa.Column("special_chars_count", sa.Float(), nullable=True),
        sa.Column("prediction", sa.String(length=10), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(length=10), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_prediction_logs_created_at", "prediction_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_prediction_logs_created_at", table_name="prediction_logs")
    op.drop_table("prediction_logs")
