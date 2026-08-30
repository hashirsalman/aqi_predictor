"""FastAPI app for Karachi AQI predictions."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query

from aqi_predictor.inference.predictor import PredictionService

app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "FastAPI backend for Karachi US AQI Day+1/Day+2/Day+3 forecasts. "
        "Uses Hopsworks Feature Store and Hopsworks Model Registry."
    ),
    version="0.1.0",
)

_prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Dependency override hook for tests and deployment wiring."""

    return _prediction_service


@app.get("/health")
def health(service: Annotated[PredictionService, Depends(get_prediction_service)]) -> dict[str, object]:
    """Lightweight API health check."""

    return service.health()


@app.get("/predict")
def predict(
    service: Annotated[PredictionService, Depends(get_prediction_service)],
    force_model_refresh: Annotated[
        bool,
        Query(description="Reload models from Hopsworks Model Registry before predicting."),
    ] = False,
) -> dict[str, object]:
    """Return current AQI context and three daily-average AQI forecasts."""

    try:
        return service.predict(force_model_refresh=force_model_refresh)
    except Exception as exc:  # noqa: BLE001 - API must convert cloud/model failures into a clear response.
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is unavailable. "
                "Check Hopsworks Feature Store, Model Registry, and workflow freshness. "
                f"Cause: {exc}"
            ),
        ) from exc

