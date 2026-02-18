from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import Concatenate
from typing import Self

from sqlalchemy import exc
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from course_mlops.common.dal import exceptions
from course_mlops.utils import EnvironmentVariable


class Transaction:
    __engine: AsyncEngine | None = None

    @classmethod
    def _get_engine(cls) -> AsyncEngine:
        if cls.__engine is None:
            cls.__engine = create_async_engine(
                f"postgresql+psycopg_async://{EnvironmentVariable.DB_USER.read()}:{EnvironmentVariable.DB_PASSWORD.read()}@{EnvironmentVariable.DB_HOST.read()}:{EnvironmentVariable.DB_PORT.read()}/{EnvironmentVariable.DB_NAME.read()}",
            )
        return cls.__engine

    @classmethod
    async def dispose_engine(cls) -> None:
        if cls.__engine is not None:
            await cls.__engine.dispose()
            cls.__engine = None

    def __init__(self, commit: bool) -> None:
        self.session = AsyncSession(autocommit=False, autoflush=False, expire_on_commit=False, bind=self._get_engine())
        self.commit = commit

    async def __aenter__(self) -> Self:
        await self.session.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.commit and exc_type is None:
            try:
                await self.session.commit()
            except exc.IntegrityError as e:
                await self.session.rollback()
                raise exceptions.IntegrityError from e
            except exc.DatabaseError as e:
                await self.session.rollback()
                raise exceptions.TransactionError from e
        if exc_type is not None:
            await self.session.rollback()
        await self.session.__aexit__(exc_type, exc_value, traceback)


def db_transaction[**P, T](
    commit: bool = False,
) -> Callable[[Callable[Concatenate[AsyncSession, P], Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[Concatenate[AsyncSession, P], Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with Transaction(commit=commit) as transaction:
                try:
                    result = await func(transaction.session, *args, **kwargs)
                except exc.NoResultFound as e:
                    raise exceptions.NotFoundError from e
                except exc.IntegrityError as e:
                    raise exceptions.IntegrityError from e
                except exc.SQLAlchemyError as e:
                    raise exceptions.TransactionError from e
            return result

        return wrapper

    return decorator
