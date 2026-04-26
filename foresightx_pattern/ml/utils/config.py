from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    config_path: Path
    data_service_url: str
    raw_data_path: Path
    processed_data_path: Path
    features_path: Path
    model_dir: Path
    reports_dir: Path
    mlruns_dir: Path
    tickers: list[str]
    data: dict[str, Any]
    features: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    tracking: dict[str, Any]
    service: dict[str, Any]
    cache: dict[str, Any]


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    path = Path(os.getenv("FORESIGHTX_CONFIG_PATH", config_path or DEFAULT_CONFIG_PATH))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_settings(config_path: Path | None = None) -> AppSettings:
    config = load_yaml_config(config_path)
    project_root = PROJECT_ROOT
    artifacts_dir = project_root / config["artifacts"]["root_dir"]
    model_dir = Path(os.getenv("FORESIGHTX_ARTIFACTS_DIR", artifacts_dir / "model"))
    raw_data_path = project_root / config["artifacts"]["raw_data_path"]
    processed_data_path = project_root / config["artifacts"]["processed_data_path"]
    features_path = project_root / config["artifacts"]["features_path"]
    reports_dir = project_root / config["artifacts"]["reports_path"]
    mlruns_dir = project_root / config["artifacts"]["mlruns_path"]
    for path in [
        artifacts_dir,
        model_dir,
        raw_data_path.parent,
        processed_data_path.parent,
        features_path.parent,
        reports_dir,
        mlruns_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return AppSettings(
        project_root=project_root,
        config_path=Path(os.getenv("FORESIGHTX_CONFIG_PATH", config_path or DEFAULT_CONFIG_PATH)),
        data_service_url=os.getenv("PATTERN_DATA_SERVICE_URL", "http://localhost:8001"),
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        features_path=features_path,
        model_dir=model_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        tickers=[ticker.upper() for ticker in config["data"]["tickers"]],
        data=config["data"],
        features=config["features"],
        model=config["model"],
        training=config["training"],
        tracking=config["tracking"],
        service=config["service"],
        cache=config["cache"],
    )
