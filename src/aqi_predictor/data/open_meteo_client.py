"""Open-Meteo client for Karachi weather and air-quality observations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from aqi_predictor.config import AppConfig, load_config
from aqi_predictor.data.schemas import (
    AIR_QUALITY_VARIABLE_NAMES,
    WEATHER_VARIABLE_NAMES,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenMeteoDataset:
    """A parsed Open-Meteo dataset plus its source units."""

    frame: pd.DataFrame
    units: dict[str, str]


class OpenMeteoClient:
    """Small robust client for Open-Meteo weather and air-quality endpoints."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        data_config = self.config.raw["data"]["open_meteo"]
        project_config = self.config.raw["project"]

        self.latitude = float(project_config["latitude"])
        self.longitude = float(project_config["longitude"])
        self.timezone_name = str(project_config["timezone"])
        self.local_zone = ZoneInfo(self.timezone_name)
        self.weather_archive_url = str(data_config["weather_archive_url"])
        self.weather_recent_url = str(data_config["weather_recent_url"])
        self.air_quality_url = str(data_config["air_quality_url"])
        self.timeout_seconds = float(data_config["request_timeout_seconds"])
        self.max_retries = int(data_config["max_retries"])
        self.retry_backoff_seconds = float(data_config["retry_backoff_seconds"])
        self.weather_hourly = tuple(data_config["weather_hourly"])
        self.air_quality_hourly = tuple(data_config["air_quality_hourly"])

    def fetch_historical_weather(self, start_date: date, end_date: date) -> OpenMeteoDataset:
        """Fetch historical hourly weather observations from Open-Meteo archive."""

        payload = self._get_json(
            self.weather_archive_url,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(self.weather_hourly),
                "timezone": self.timezone_name,
            },
        )
        return self._parse_hourly_payload(payload, self.weather_hourly)

    def fetch_historical_air_quality(
        self, start_date: date, end_date: date
    ) -> OpenMeteoDataset:
        """Fetch historical hourly air-quality observations from Open-Meteo."""

        payload = self._get_json(
            self.air_quality_url,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(self.air_quality_hourly),
                "timezone": self.timezone_name,
            },
        )
        return self._parse_hourly_payload(payload, self.air_quality_hourly)

    def fetch_recent_weather(self, past_days: int = 2) -> OpenMeteoDataset:
        """Fetch recent observed weather rows and drop any future timestamps."""

        payload = self._get_json(
            self.weather_recent_url,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(self.weather_hourly),
                "past_days": past_days,
                "forecast_days": 0,
                "timezone": self.timezone_name,
            },
        )
        dataset = self._parse_hourly_payload(payload, self.weather_hourly)
        return self._without_future_rows(dataset)

    def fetch_recent_air_quality(self, past_days: int = 2) -> OpenMeteoDataset:
        """Fetch recent observed air-quality rows and drop any future timestamps."""

        payload = self._get_json(
            self.air_quality_url,
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(self.air_quality_hourly),
                "past_days": past_days,
                "forecast_days": 0,
                "timezone": self.timezone_name,
            },
        )
        dataset = self._parse_hourly_payload(payload, self.air_quality_hourly)
        return self._without_future_rows(dataset)

    def fetch_merged_historical_sample(
        self, start_date: date, end_date: date
    ) -> OpenMeteoDataset:
        """Fetch and merge historical weather and air quality by UTC event time."""

        weather = self.fetch_historical_weather(start_date, end_date)
        air_quality = self.fetch_historical_air_quality(start_date, end_date)
        return self._merge(weather, air_quality)

    def fetch_merged_recent_sample(self, past_days: int = 2) -> OpenMeteoDataset:
        """Fetch and merge recent weather and air quality by UTC event time."""

        weather = self.fetch_recent_weather(past_days=past_days)
        air_quality = self.fetch_recent_air_quality(past_days=past_days)
        return self._merge(weather, air_quality)

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.info("Fetching Open-Meteo data from %s, attempt %s", url, attempt)
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Open-Meteo response was not a JSON object")
                if "error" in payload:
                    raise ValueError(f"Open-Meteo error response: {payload}")
                return payload
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning("Open-Meteo request failed: %s; retrying in %.1fs", exc, sleep_seconds)
                time.sleep(sleep_seconds)
        raise RuntimeError(f"Open-Meteo request failed after {self.max_retries} attempts") from last_error

    def _parse_hourly_payload(
        self, payload: dict[str, Any], expected_variables: tuple[str, ...]
    ) -> OpenMeteoDataset:
        self._validate_response_metadata(payload)

        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("Open-Meteo response missing hourly data")

        missing_variables = [name for name in ("time", *expected_variables) if name not in hourly]
        if missing_variables:
            raise ValueError(f"Open-Meteo response missing hourly fields: {missing_variables}")

        times = hourly["time"]
        if not isinstance(times, list):
            raise ValueError("Open-Meteo hourly time field must be a list")

        row_count = len(times)
        for variable in expected_variables:
            values = hourly[variable]
            if not isinstance(values, list):
                raise ValueError(f"Open-Meteo field {variable} must be a list")
            if len(values) != row_count:
                raise ValueError(
                    f"Open-Meteo field {variable} has {len(values)} rows, expected {row_count}"
                )

        frame = pd.DataFrame({name: hourly[name] for name in expected_variables})
        local_times = pd.to_datetime(times)
        if local_times.isna().any():
            raise ValueError("Open-Meteo response contains unparseable timestamps")

        frame.insert(0, "event_time_local", local_times.tz_localize(self.local_zone))
        frame.insert(0, "event_time_utc", frame["event_time_local"].dt.tz_convert("UTC"))

        duplicate_count = int(frame["event_time_utc"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"Open-Meteo response contains duplicate timestamps: {duplicate_count}")

        units = payload.get("hourly_units", {})
        if not isinstance(units, dict):
            units = {}
        clean_units = {name: str(units.get(name, "")) for name in expected_variables}

        return OpenMeteoDataset(
            frame=frame.sort_values("event_time_utc").reset_index(drop=True),
            units=clean_units,
        )

    def _validate_response_metadata(self, payload: dict[str, Any]) -> None:
        latitude = payload.get("latitude")
        longitude = payload.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("Open-Meteo response missing latitude/longitude metadata")

        if abs(float(latitude) - self.latitude) > 1.0:
            raise ValueError(f"Open-Meteo latitude is unexpectedly far from Karachi: {latitude}")
        if abs(float(longitude) - self.longitude) > 1.0:
            raise ValueError(f"Open-Meteo longitude is unexpectedly far from Karachi: {longitude}")

    def _without_future_rows(self, dataset: OpenMeteoDataset) -> OpenMeteoDataset:
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        frame = dataset.frame.loc[dataset.frame["event_time_utc"] <= now_utc].copy()
        return OpenMeteoDataset(frame=frame.reset_index(drop=True), units=dataset.units)

    @staticmethod
    def _merge(weather: OpenMeteoDataset, air_quality: OpenMeteoDataset) -> OpenMeteoDataset:
        frame = weather.frame.merge(
            air_quality.frame,
            on=["event_time_utc", "event_time_local"],
            how="inner",
            validate="one_to_one",
        )
        units = {**weather.units, **air_quality.units}
        expected = (*WEATHER_VARIABLE_NAMES, *AIR_QUALITY_VARIABLE_NAMES)
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise ValueError(f"Merged Open-Meteo data is missing fields: {missing}")
        return OpenMeteoDataset(frame=frame.sort_values("event_time_utc").reset_index(drop=True), units=units)


def default_sample_window(days: int = 10, archive_lag_days: int = 7) -> tuple[date, date]:
    """Return a recent historical window that avoids archive freshness lag."""

    end_date = datetime.now(ZoneInfo("Asia/Karachi")).date() - timedelta(days=archive_lag_days)
    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date
