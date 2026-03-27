from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class InferenceSettings:
    service_name: str = field(default_factory=lambda: os.getenv("PATTERN_SERVICE_NAME", "foresightx-pattern"))
    environment: str = field(default_factory=lambda: os.getenv("PATTERN_ENV", "development"))
    host: str = field(default_factory=lambda: os.getenv("PATTERN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PATTERN_PORT", "8003")))
    log_level: str = field(default_factory=lambda: os.getenv("PATTERN_LOG_LEVEL", "INFO"))
    cors_origins: list[str] = field(
        default_factory=lambda: _parse_csv(
            os.getenv("PATTERN_CORS_ORIGINS"),
            ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
        )
    )
    project_root: Path = field(
        default_factory=lambda: Path(os.getenv("PATTERN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    )

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
