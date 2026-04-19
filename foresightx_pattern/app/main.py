from __future__ import annotations

from fastapi import FastAPI

from foresightx_pattern.app.api.routes import build_router
from foresightx_pattern.app.core.settings import get_container
from foresightx_pattern.app.services.confidence_service import ConfidenceService
from foresightx_pattern.app.services.feature_service import FeatureService
from foresightx_pattern.app.services.inference_service import InferenceService
from foresightx_pattern.app.services.model_loader import ModelLoader


def create_app(settings=None, data_provider=None) -> FastAPI:
    container = get_container(settings=settings, data_provider=data_provider)
    model_loader = ModelLoader(container.settings)
    feature_service = FeatureService(container.settings, container.data_provider)
    confidence_service = ConfidenceService(samples=container.settings.service.get("mc_dropout_samples", 20))
    inference_service = InferenceService(
        settings=container.settings,
        model_loader=model_loader,
        feature_service=feature_service,
        confidence_service=confidence_service,
    )
    app = FastAPI(title="ForesightX Pattern", version="1.0.0")
    app.state.inference_service = inference_service
    app.include_router(build_router(inference_service))
    return app


app = create_app()
