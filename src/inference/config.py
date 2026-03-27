from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_database_url(value: str) -> str:
    normalized = value
    if normalized.startswith("postgresql://"):
        normalized = normalized.replace("postgresql://", "postgresql+asyncpg://", 1)

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


@dataclass(slots=True)
class InferenceSettings:
    service_name: str = field(default_factory=lambda: os.getenv("PATTERN_SERVICE_NAME", "foresightx-pattern"))
    environment: str = field(default_factory=lambda: os.getenv("PATTERN_ENV", "development"))
    host: str = field(default_factory=lambda: os.getenv("PATTERN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PATTERN_PORT", "8003")))
    log_level: str = field(default_factory=lambda: os.getenv("PATTERN_LOG_LEVEL", "INFO"))
    database_url: str = field(
        default_factory=lambda: _normalize_database_url(
            os.getenv(
                "PATTERN_DATABASE_URL",
                "postgresql+asyncpg://postgres:postgres@localhost:5432/foresightx_pattern",
            )
        )
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _parse_csv(
            os.getenv("PATTERN_CORS_ORIGINS"),
            ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
        )
    )
    project_root: Path = field(default_factory=lambda: Path(os.getenv("PATTERN_PROJECT_ROOT", PROJECT_ROOT)))

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def metadata_dir(self) -> Path:
        return self.project_root / "metadata"

    @property
    def features_dir(self) -> Path:
        return self.project_root / "data" / "features"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"
