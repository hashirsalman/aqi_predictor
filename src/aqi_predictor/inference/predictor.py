"""Prediction orchestration for the FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS
from aqi_predictor.inference.health_alerts import build_alert
from aqi_predictor.inference.live_features import LatestFeatureRow, load_latest_live_features
from aqi_predictor.inference.model_loader import HORIZONS, LoadedRegistryModel, load_latest_models


@dataclass
class PredictionService:
    """Small service object that keeps Hopsworks access behind one interface."""

    feature_store: Any | None = None
    model_registry: Any | None = None
    _cached_models: dict[str, LoadedRegistryModel] | None = None

    def predict(self, force_model_refresh: bool = False) -> dict[str, Any]:
        """Return current AQI plus Day+1/Day+2/Day+3 predictions."""

        latest_features = load_latest_live_features(self.feature_store)
        models = self._models(force_refresh=force_model_refresh)
        predictions = [
            _predict_one_horizon(horizon, models[horizon], latest_features)
            for horizon in HORIZONS
        ]
        current_alert = build_alert(latest_features.current_aqi)
        return {
            "city": latest_features.city,
            "aqi_standard": "US AQI",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_observation": {
                "event_time_utc": latest_features.event_time_utc,
                "event_time_local": latest_features.event_time_local,
                "current_aqi": latest_features.current_aqi,
                "feature_schema_version": latest_features.feature_schema_version,
                "alert": current_alert,
            },
            "current_conditions": _current_conditions(latest_features),
            "predictions": predictions,
            "input_policy": "Observed/current and historical features only. No future weather, pollutant, or AQI forecasts.",
        }

    def health(self) -> dict[str, Any]:
        """Return lightweight API health without touching Hopsworks."""

        return {
            "status": "ok",
            "service": "pearls-aqi-fastapi",
            "required_horizons": list(HORIZONS),
        }

    def _models(self, force_refresh: bool = False) -> dict[str, LoadedRegistryModel]:
        if force_refresh or self._cached_models is None:
            self._cached_models = load_latest_models(self.model_registry)
        return self._cached_models


def _predict_one_horizon(
    horizon: str,
    model: LoadedRegistryModel,
    latest_features: LatestFeatureRow,
) -> dict[str, Any]:
    """Run one model and build a serializable prediction block."""

    expected_columns = list(CANONICAL_FEATURE_COLUMNS)
    if list(model.feature_columns) != expected_columns:
        raise ValueError(
            f"Registry model {model.registry_name} v{model.registry_version} feature contract mismatch."
        )

    prediction = float(model.estimator.predict(latest_features.features[expected_columns])[0])
    return {
        "horizon": horizon,
        "target": f"Day +{horizon[-1]} average AQI",
        "predicted_aqi": prediction,
        "rounded_aqi": round(prediction),
        "alert": build_alert(prediction),
        "model": {
            "registry_name": model.registry_name,
            "registry_version": model.registry_version,
            "model_family": model.model_family,
            "validation_rmse": model.metrics.get("validation_rmse"),
            "test_rmse": model.metrics.get("test_rmse"),
            "validation_mae": model.metrics.get("validation_mae"),
            "validation_r2": model.metrics.get("validation_r2"),
            "beats_persistence_validation": model.metrics.get("beats_persistence_validation"),
        },
    }


def _current_conditions(latest_features: LatestFeatureRow) -> dict[str, Any]:
    """Build user-facing latest pollutant and weather context."""

    observed = latest_features.observed_values
    return {
        "pollutants": {
            "pm2_5": {"label": "PM2.5", "value": observed.get("pm2_5"), "unit": "ug/m3"},
            "pm10": {"label": "PM10", "value": observed.get("pm10"), "unit": "ug/m3"},
            "nitrogen_dioxide": {"label": "NO2", "value": observed.get("nitrogen_dioxide"), "unit": "ug/m3"},
            "carbon_monoxide": {"label": "CO", "value": observed.get("carbon_monoxide"), "unit": "ug/m3"},
            "ozone": {"label": "O3", "value": observed.get("ozone"), "unit": "ug/m3"},
            "sulphur_dioxide": {"label": "SO2", "value": observed.get("sulphur_dioxide"), "unit": "ug/m3"},
        },
        "weather": {
            "temperature_2m": {"label": "Temperature", "value": observed.get("temperature_2m"), "unit": "C"},
            "relative_humidity_2m": {"label": "Humidity", "value": observed.get("relative_humidity_2m"), "unit": "%"},
            "wind_speed_10m": {"label": "Wind speed", "value": observed.get("wind_speed_10m"), "unit": "km/h"},
            "surface_pressure": {"label": "Pressure", "value": observed.get("surface_pressure"), "unit": "hPa"},
            "cloud_cover": {"label": "Cloud cover", "value": observed.get("cloud_cover"), "unit": "%"},
            "precipitation": {"label": "Precipitation", "value": observed.get("precipitation"), "unit": "mm"},
        },
    }
