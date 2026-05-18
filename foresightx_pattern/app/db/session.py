from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _normalize_database_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    normalized = url.strip()
    if normalized.startswith(("\"", "'")) and normalized.endswith(("\"", "'")):
        normalized = normalized[1:-1].strip()
    lower = normalized.lower()
    if lower.startswith("postgresql+asyncpg://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgresql+asyncpg://") :]
    elif lower.startswith("postgresql+psycopg2://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgresql+psycopg2://") :]
    elif lower.startswith("postgres://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgres://") :]
    elif lower.startswith("postgresql://"):
        normalized = "postgresql+psycopg://" + normalized[len("postgresql://") :]
    return normalized


def get_engine(database_url: str) -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(_normalize_database_url(database_url), pool_pre_ping=True, future=True)
    return _engine


def get_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(database_url), expire_on_commit=False)
    return _session_factory


async def check_database_connection(database_url: str) -> None:
    async with get_engine(database_url).connect() as connection:
        await connection.execute(text("SELECT 1"))


async def close_database() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
