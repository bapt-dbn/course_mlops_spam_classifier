from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from course_mlops.common.db import Created
from course_mlops.common.db import Schema


class PredictionLog(Schema, Created):
    __tablename__ = "prediction_logs"
    __table_args__ = (Index("idx_prediction_logs_created_at", "created_at", postgresql_using="btree"),)

    model_version: Mapped[str | None] = mapped_column(String(50), default=None)
    message_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    text_length: Mapped[float | None] = mapped_column(Float, default=None)
    word_count: Mapped[float | None] = mapped_column(Float, default=None)
    caps_ratio: Mapped[float | None] = mapped_column(Float, default=None)
    special_chars_count: Mapped[float | None] = mapped_column(Float, default=None)
    prediction: Mapped[str | None] = mapped_column(String(10), default=None)
    probability: Mapped[float | None] = mapped_column(Float, default=None)
    confidence: Mapped[str | None] = mapped_column(String(10), default=None)
