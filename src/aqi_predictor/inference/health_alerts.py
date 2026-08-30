"""US AQI category and health-alert helpers for inference responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AqiCategory:
    """EPA-style AQI category used by the dashboard and API."""

    name: str
    lower_bound: int
    upper_bound: int | None
    color: str
    alert_level: str
    message: str


AQI_CATEGORIES: tuple[AqiCategory, ...] = (
    AqiCategory(
        name="Good",
        lower_bound=0,
        upper_bound=50,
        color="green",
        alert_level="none",
        message="Air quality is satisfactory for most people.",
    ),
    AqiCategory(
        name="Moderate",
        lower_bound=51,
        upper_bound=100,
        color="yellow",
        alert_level="notice",
        message="Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    ),
    AqiCategory(
        name="Unhealthy for Sensitive Groups",
        lower_bound=101,
        upper_bound=150,
        color="orange",
        alert_level="caution",
        message="Sensitive groups should reduce prolonged or heavy outdoor exertion.",
    ),
    AqiCategory(
        name="Unhealthy",
        lower_bound=151,
        upper_bound=200,
        color="red",
        alert_level="alert",
        message="Everyone may begin to experience health effects; sensitive groups should avoid prolonged outdoor exertion.",
    ),
    AqiCategory(
        name="Very Unhealthy",
        lower_bound=201,
        upper_bound=300,
        color="purple",
        alert_level="alert",
        message="Health alert: everyone should reduce outdoor activity.",
    ),
    AqiCategory(
        name="Hazardous",
        lower_bound=301,
        upper_bound=None,
        color="maroon",
        alert_level="hazardous",
        message="Emergency conditions: everyone should avoid outdoor exertion.",
    ),
)


def classify_aqi(aqi: float) -> AqiCategory:
    """Return the AQI category for a numeric US AQI value."""

    if aqi < 0:
        raise ValueError(f"AQI must be non-negative, got {aqi}.")

    rounded = round(float(aqi))
    for category in AQI_CATEGORIES:
        if rounded >= category.lower_bound and (
            category.upper_bound is None or rounded <= category.upper_bound
        ):
            return category
    raise ValueError(f"Unable to classify AQI value {aqi}.")


def build_alert(aqi: float) -> dict[str, object]:
    """Build a serializable health alert block for an AQI value."""

    category = classify_aqi(aqi)
    return {
        "category": category.name,
        "color": category.color,
        "alert_level": category.alert_level,
        "is_health_alert": category.alert_level in {"alert", "hazardous"},
        "message": category.message,
    }

