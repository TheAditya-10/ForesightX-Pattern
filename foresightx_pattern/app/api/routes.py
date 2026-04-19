from __future__ import annotations

from fastapi import APIRouter, HTTPException

from foresightx_pattern.app.schemas.prediction import PredictionRequest, PredictionResponse
from foresightx_pattern.app.services.inference_service import InferenceService


def build_router(service: InferenceService) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            return service.predict(request.ticker, request.timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return router
