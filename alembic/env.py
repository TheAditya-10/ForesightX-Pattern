from logging.config import fileConfig
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from foresightx_pattern.app.db.base import Base
from foresightx_pattern.app.db import models  # noqa: F401


def _normalize_database_url(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith(("\"", "'")) and normalized.endswith(("\"", "'")):
        normalized = normalized[1:-1].strip()
    lower = normalized.lower()
    if lower.startswith("postgresql+psycopg2://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgresql+psycopg2://") :]
    elif lower.startswith("postgresql+psycopg://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgresql+psycopg://") :]
    elif lower.startswith("postgres://"):
        normalized = "postgresql+asyncpg://" + normalized[len("postgres://") :]
    elif lower.startswith("postgresql://") and "asyncpg" not in lower:
        normalized = "postgresql+asyncpg://" + normalized[len("postgresql://") :]

    parsed = urlsplit(normalized)
    if parsed.scheme != "postgresql+asyncpg":
        return normalized

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    sslmode = query.pop("sslmode", None)
    if sslmode and "ssl" not in query:
        query["ssl"] = sslmode
    query.pop("channel_binding", None)
    query.pop("gssencmode", None)
    query.pop("target_session_attrs", None)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


config = context.config
database_url = _normalize_database_url(
    os.getenv("PATTERN_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/foresightx_pattern")
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
