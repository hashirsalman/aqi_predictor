"""Tests for Open-Meteo parsing and validation logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.data.open_meteo_client import OpenMeteoClient, OpenMeteoDataset
from aqi_predictor.data.validation import validate_hourly_observations


class OpenMeteoClientParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenMeteoClient()

    def test_parse_hourly_payload_localizes_karachi_time_and_converts_to_utc(self) -> None:
        payload = {
            "latitude": 24.86,
            "longitude": 67.0,
            "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
            "hourly": {
                "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                "temperature_2m": [18.0, 19.0],
            },
        }

        dataset = self.client._parse_hourly_payload(payload, ("temperature_2m",))

        self.assertEqual(len(dataset.frame), 2)
        self.assertEqual(str(dataset.frame["event_time_local"].dt.tz), "Asia/Karachi")
        self.assertEqual(str(dataset.frame["event_time_utc"].dt.tz), "UTC")
        self.assertEqual(dataset.units["temperature_2m"], "°C")

    def test_parse_hourly_payload_rejects_missing_field(self) -> None:
        payload = {
            "latitude": 24.86,
            "longitude": 67.0,
            "hourly": {"time": ["2026-01-01T00:00"]},
        }

        with self.assertRaises(ValueError):
            self.client._parse_hourly_payload(payload, ("temperature_2m",))

    def test_validation_rejects_missing_hour(self) -> None:
        frame = pd.DataFrame(
            {
                "event_time_utc": pd.to_datetime(
                    ["2026-01-01T00:00Z", "2026-01-01T02:00Z"]
                ),
                "event_time_local": pd.to_datetime(
                    ["2026-01-01T05:00", "2026-01-01T07:00"]
                ).tz_localize("Asia/Karachi"),
                "us_aqi": [100, 120],
            }
        )

        with self.assertRaises(ValueError):
            validate_hourly_observations(frame, {"us_aqi": "US AQI"}, ("us_aqi",))

    def test_merge_requires_one_to_one_timestamps(self) -> None:
        weather = OpenMeteoDataset(
            frame=pd.DataFrame(
                {
                    "event_time_utc": pd.to_datetime(["2026-01-01T00:00Z"]),
                    "event_time_local": pd.to_datetime(["2026-01-01T05:00"]).tz_localize(
                        "Asia/Karachi"
                    ),
                    "temperature_2m": [18.0],
                    "relative_humidity_2m": [70],
                    "precipitation": [0],
                    "rain": [0],
                    "surface_pressure": [1010],
                    "cloud_cover": [20],
                    "wind_speed_10m": [4],
                    "wind_direction_10m": [180],
                    "wind_gusts_10m": [8],
                }
            ),
            units={},
        )
        air_quality = OpenMeteoDataset(
            frame=pd.DataFrame(
                {
                    "event_time_utc": pd.to_datetime(["2026-01-01T00:00Z"]),
                    "event_time_local": pd.to_datetime(["2026-01-01T05:00"]).tz_localize(
                        "Asia/Karachi"
                    ),
                    "pm10": [200],
                    "pm2_5": [100],
                    "carbon_monoxide": [300],
                    "nitrogen_dioxide": [20],
                    "sulphur_dioxide": [5],
                    "ozone": [80],
                    "us_aqi": [170],
                }
            ),
            units={},
        )

        merged = OpenMeteoClient._merge(weather, air_quality)

        self.assertEqual(len(merged.frame), 1)
        self.assertEqual(merged.frame["us_aqi"].iloc[0], 170)


if __name__ == "__main__":
    unittest.main()
