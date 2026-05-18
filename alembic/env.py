from logging.config import fileConfig
import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from foresightx_pattern.app.db.base import Base
from foresightx_pattern.app.db import models  # noqa: F401

load_dotenv()


def _normalize_database_url(value: str) -> str:
    normalized = value.strip()
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


config = context.config
database_url = _normalize_database_url(
    os.getenv("PATTERN_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/foresightx_pattern")
)
config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
