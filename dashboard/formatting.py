"""Formatting helpers for the Streamlit dashboard."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def prediction_table(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the API prediction payload into a dashboard table."""

    rows = []
    for prediction in payload.get("predictions", []):
        model = prediction.get("model", {})
        alert = prediction.get("alert", {})
        rows.append(
            {
                "Horizon": prediction.get("horizon"),
                "Target": prediction.get("target"),
                "Predicted AQI": prediction.get("rounded_aqi"),
                "Category": alert.get("category"),
                "Alert Level": alert.get("alert_level"),
                "Model": model.get("model_family"),
                "Registry Version": model.get("registry_version"),
                "Validation RMSE": model.get("validation_rmse"),
                "Test RMSE": model.get("test_rmse"),
            }
        )
    return pd.DataFrame(rows)


def model_metric_table(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a compact model freshness/metric table."""

    rows = []
    for prediction in payload.get("predictions", []):
        model = prediction.get("model", {})
        rows.append(
            {
                "Horizon": prediction.get("horizon"),
                "Registry Name": model.get("registry_name"),
                "Version": model.get("registry_version"),
                "Family": model.get("model_family"),
                "Validation RMSE": model.get("validation_rmse"),
                "Validation MAE": model.get("validation_mae"),
                "Validation R²": model.get("validation_r2"),
                "Test RMSE": model.get("test_rmse"),
            }
        )
    return pd.DataFrame(rows)


def condition_table(payload: dict[str, Any], group: str) -> pd.DataFrame:
    """Convert current pollutant/weather context into display rows."""

    conditions = payload.get("current_conditions", {}).get(group, {})
    rows = []
    for key, item in conditions.items():
        value = item.get("value")
        if value is None:
            continue
        rows.append(
            {
                "Metric": item.get("label", key),
                "Value": round(float(value), 2),
                "Unit": item.get("unit", ""),
            }
        )
    return pd.DataFrame(rows)


def forecast_summary(payload: dict[str, Any]) -> dict[str, float | int | None]:
    """Return simple summary statistics for the visible forecast section."""

    values = [
        float(prediction["rounded_aqi"])
        for prediction in payload.get("predictions", [])
        if prediction.get("rounded_aqi") is not None
    ]
    if not values:
        return {"average": None, "maximum": None, "minimum": None}
    return {
        "average": round(sum(values) / len(values), 1),
        "maximum": int(max(values)),
        "minimum": int(min(values)),
    }


def observation_age_hours(payload: dict[str, Any], now: datetime | None = None) -> float | None:
    """Calculate source-observation age in hours from the API payload."""

    event_time = payload.get("source_observation", {}).get("event_time_utc")
    if not event_time:
        return None
    parsed = datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    resolved_now = now or datetime.now(UTC)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=UTC)
    return round((resolved_now - parsed).total_seconds() / 3600, 2)


def strongest_alert(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the most severe alert from current/predicted AQI values."""

    severity = {"none": 0, "notice": 1, "caution": 2, "alert": 3, "hazardous": 4}
    alerts = [payload.get("source_observation", {}).get("alert", {})]
    alerts.extend(prediction.get("alert", {}) for prediction in payload.get("predictions", []))
    alerts = [alert for alert in alerts if alert]
    if not alerts:
        return None
    return max(alerts, key=lambda alert: severity.get(str(alert.get("alert_level")), -1))
