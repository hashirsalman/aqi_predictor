"""Canonical feature lists and train/serve eligibility reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.data.schemas import ALL_OPEN_METEO_VARIABLES


BASE_OBSERVED_FEATURES = (
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
)

TIME_FEATURES = (
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
)

WIND_DIRECTION_FEATURES = ("wind_dir_sin", "wind_dir_cos")

AQI_DERIVED_FEATURES = (
    "aqi_change_1h",
    "aqi_change_3h",
    "aqi_change_6h",
    "aqi_pct_change_1h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_12h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_6h",
    "aqi_roll_std_12h",
    "aqi_roll_std_24h",
    "aqi_roll_min_24h",
    "aqi_roll_max_24h",
)

AQI_LAG_FEATURES = tuple(f"us_aqi_lag_{lag}h" for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168))
POLLUTANT_LAG_FEATURES = tuple(
    f"{column}_lag_{lag}h"
    for column in ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone")
    for lag in (1, 3, 6, 12, 24)
)
WEATHER_LAG_FEATURES = tuple(
    f"{column}_lag_{lag}h"
    for column in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure")
    for lag in (1, 3, 6, 12, 24)
)

ROLLING_FEATURES = (
    "pm2_5_roll_mean_6h",
    "pm2_5_roll_mean_12h",
    "pm2_5_roll_mean_24h",
    "pm10_roll_mean_6h",
    "pm10_roll_mean_12h",
    "pm10_roll_mean_24h",
    "wind_speed_10m_roll_mean_6h",
    "wind_speed_10m_roll_mean_24h",
    "relative_humidity_2m_roll_mean_6h",
    "relative_humidity_2m_roll_mean_24h",
    "temperature_2m_roll_mean_6h",
    "temperature_2m_roll_mean_24h",
    "precipitation_roll_sum_6h",
    "precipitation_roll_sum_24h",
)

TARGET_COLUMNS = ("target_aqi_day1", "target_aqi_day2", "target_aqi_day3")

CANONICAL_FEATURE_COLUMNS = (
    *BASE_OBSERVED_FEATURES,
    *TIME_FEATURES,
    *WIND_DIRECTION_FEATURES,
    *AQI_DERIVED_FEATURES,
    *AQI_LAG_FEATURES,
    *POLLUTANT_LAG_FEATURES,
    *WEATHER_LAG_FEATURES,
    *ROLLING_FEATURES,
)

FEATURE_SCHEMA_VERSION = 1


def build_feature_eligibility_table(
    validation_path: Path = PROJECT_ROOT / "reports" / "metrics" / "open_meteo_contract_validation.json",
) -> list[dict[str, Any]]:
    """Build the mandatory train/serve feature eligibility table."""

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    availability = {
        row["feature"]: row for row in validation.get("feature_availability_contract", [])
    }
    missing_percent = validation["historical_summary"]["missing_percent"]

    table: list[dict[str, Any]] = []
    for variable in ALL_OPEN_METEO_VARIABLES:
        row = availability.get(variable.name, {})
        missingness = float(missing_percent.get(variable.name, 100.0))
        historical_available = bool(row.get("historical_available", False))
        live_available = bool(row.get("live_current_available", False))
        same_units = bool(row.get("same_units_semantics", False))
        missingness_ok = missingness <= 5.0
        keep = historical_available and live_available and same_units and missingness_ok
        table.append(
            {
                "feature": variable.name,
                "historical_available": historical_available,
                "live_current_available": live_available,
                "same_units_semantics": same_units,
                "missingness_percent": missingness,
                "missingness_acceptable": missingness_ok,
                "keep": keep,
                "reason": "Kept: available consistently with acceptable missingness."
                if keep
                else "Excluded: does not pass train/serve availability, semantics, or missingness gate.",
            }
        )
    return table
