from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import declarative_mixin
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql import func


@declarative_mixin
class Schema(MappedAsDataclass, DeclarativeBase):
    id: Mapped[int] = mapped_column(Integer, primary_key=True, init=False)


@declarative_mixin
class Created(MappedAsDataclass):
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), init=False)
