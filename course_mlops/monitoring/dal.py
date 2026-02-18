from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from course_mlops.common.dal.transaction import db_transaction
from course_mlops.monitoring.db import PredictionLog
from course_mlops.monitoring.schemas import PredictionRecord
from course_mlops.monitoring.schemas import PredictionStats
from course_mlops.monitoring.schemas import RecentPrediction


@db_transaction()
async def ping(session: AsyncSession) -> None:
    await session.execute(text("SELECT 1"))


@db_transaction(commit=True)
async def log_prediction(session: AsyncSession, record: PredictionRecord) -> None:
    session.add(PredictionLog(**record.model_dump()))


@db_transaction()
async def get_recent_predictions(session: AsyncSession, model_version: str, limit: int = 500) -> list[RecentPrediction]:
    stmt = (
        select(
            PredictionLog.text_length,
            PredictionLog.word_count,
            PredictionLog.caps_ratio,
            PredictionLog.special_chars_count,
            PredictionLog.prediction,
            PredictionLog.probability,
        )
        .where(PredictionLog.model_version == model_version)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
    )
    return [RecentPrediction.model_validate(row, from_attributes=True) for row in (await session.execute(stmt)).all()]


@db_transaction()
async def get_stats(session: AsyncSession, model_version: str) -> PredictionStats:
    stmt = select(
        func.count().label("total_predictions"),
        func.count().filter(PredictionLog.prediction == "spam").label("spam_count"),
        func.count().filter(PredictionLog.prediction == "ham").label("ham_count"),
        func.avg(PredictionLog.probability).label("avg_probability"),
        func.min(PredictionLog.created_at).label("first_prediction"),
        func.max(PredictionLog.created_at).label("last_prediction"),
    ).where(PredictionLog.model_version == model_version)

    result: PredictionStats = PredictionStats.model_validate((await session.execute(stmt)).one(), from_attributes=True)
    return result
