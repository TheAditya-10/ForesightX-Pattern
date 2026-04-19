from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow

from foresightx_pattern import __version__


LOGGER = logging.getLogger(__name__)


def configure_mlflow(settings: Any) -> None:
    tracking_uri = settings.tracking.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri((settings.mlruns_dir).as_uri())
    mlflow.set_experiment(settings.tracking.get("experiment_name", "foresightx-pattern"))


def log_run_artifacts(
    settings: Any,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
) -> str:
    configure_mlflow(settings)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("service", "foresightx-pattern")
        mlflow.set_tag("package_version", __version__)
        mlflow.log_params(params)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id


def mark_model_production(settings: Any, model_name: str, run_id: str, model_dir: Path) -> str:
    configure_mlflow(settings)
    model_uri = f"runs:/{run_id}/model_artifact"
    try:
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=registered.version,
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception:
            LOGGER.warning("Could not transition model %s version %s to Production", model_name, registered.version)
        metadata_path = model_dir / "registry.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "version": registered.version,
                    "run_id": run_id,
                    "status": "Production",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return registered.version
    except Exception:
        LOGGER.warning("Skipping MLflow model registration")
        return "unregistered"
