"""Locked constants for the Pearls AQI Predictor project."""

from dataclasses import dataclass


CITY = "Karachi"
COUNTRY = "Pakistan"
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"
AQI_STANDARD = "us_aqi"
DATA_SOURCE = "open_meteo"
FEATURE_STORE_PROVIDER = "hopsworks"
MODEL_REGISTRY_PROVIDER = "hopsworks"
RANDOM_SEED = 42


@dataclass(frozen=True)
class TargetHorizon:
    """A direct daily-average AQI forecast target."""

    name: str
    start_hour: int
    end_hour: int


TARGET_HORIZONS: tuple[TargetHorizon, ...] = (
    TargetHorizon(name="day1", start_hour=1, end_hour=24),
    TargetHorizon(name="day2", start_hour=25, end_hour=48),
    TargetHorizon(name="day3", start_hour=49, end_hour=72),
)

TARGET_HORIZON_BY_NAME = {horizon.name: horizon for horizon in TARGET_HORIZONS}
METRICS = ("rmse", "mae", "r2")
