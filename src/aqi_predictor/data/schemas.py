"""Shared data contracts for external source ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenMeteoVariable:
    """Metadata for an Open-Meteo variable used by the project."""

    name: str
    source_group: str
    expected_unit: str
    valid_time: str
    provenance: str


WEATHER_VARIABLES: tuple[OpenMeteoVariable, ...] = (
    OpenMeteoVariable("temperature_2m", "weather", "°C", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("relative_humidity_2m", "weather", "%", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("precipitation", "weather", "mm", "preceding hour sum", "ERA5/forecast model"),
    OpenMeteoVariable("rain", "weather", "mm", "preceding hour sum", "ERA5/forecast model"),
    OpenMeteoVariable("surface_pressure", "weather", "hPa", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("cloud_cover", "weather", "%", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("wind_speed_10m", "weather", "km/h", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("wind_direction_10m", "weather", "°", "instant", "ERA5/forecast model"),
    OpenMeteoVariable("wind_gusts_10m", "weather", "km/h", "instant", "ERA5/forecast model"),
)

AIR_QUALITY_VARIABLES: tuple[OpenMeteoVariable, ...] = (
    OpenMeteoVariable("pm10", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("pm2_5", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("carbon_monoxide", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("nitrogen_dioxide", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("sulphur_dioxide", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("ozone", "air_quality", "μg/m³", "instant", "CAMS/Open-Meteo modeled air quality"),
    OpenMeteoVariable("us_aqi", "air_quality", "US AQI", "instant", "calculated by Open-Meteo from pollutant indices"),
)

WEATHER_VARIABLE_NAMES = tuple(variable.name for variable in WEATHER_VARIABLES)
AIR_QUALITY_VARIABLE_NAMES = tuple(variable.name for variable in AIR_QUALITY_VARIABLES)
ALL_OPEN_METEO_VARIABLES = WEATHER_VARIABLES + AIR_QUALITY_VARIABLES
