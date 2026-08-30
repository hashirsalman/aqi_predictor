"""Tests for Hopsworks Feature Store integration helpers."""

from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.feature_store.feature_group import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_PRIMARY_KEY,
    FEATURE_GROUP_TIME_TRAVEL_FORMAT,
    FEATURE_GROUP_VERSION,
    LIVE_FEATURE_GROUP_NAME,
    LIVE_FEATURE_GROUP_VERSION,
    get_or_create_hourly_feature_group,
    get_or_create_live_feature_group,
    prepare_feature_group_frame,
    validate_feature_store_readback,
)
from aqi_predictor.feature_store.hopsworks_client import (
    _clear_known_blackhole_proxy,
    load_hopsworks_settings,
)
from aqi_predictor.features.engineering import build_features
from aqi_predictor.features.target_generation import add_aqi_targets


def fixture_observations(rows: int = 260) -> pd.DataFrame:
    event_time_utc = pd.date_range("2026-01-01T00:00Z", periods=rows, freq="h")
    event_time_local = event_time_utc.tz_convert("Asia/Karachi")
    return pd.DataFrame(
        {
            "event_time_utc": event_time_utc,
            "event_time_local": event_time_local,
            "temperature_2m": np.linspace(20, 30, rows),
            "relative_humidity_2m": np.linspace(60, 80, rows),
            "precipitation": [0.0] * rows,
            "rain": [0.0] * rows,
            "surface_pressure": [1008.0] * rows,
            "cloud_cover": [20.0] * rows,
            "wind_speed_10m": [8.0] * rows,
            "wind_direction_10m": [180.0] * rows,
            "wind_gusts_10m": [12.0] * rows,
            "pm10": np.linspace(40, 60, rows),
            "pm2_5": np.linspace(20, 40, rows),
            "carbon_monoxide": np.linspace(250, 350, rows),
            "nitrogen_dioxide": np.linspace(10, 20, rows),
            "sulphur_dioxide": np.linspace(5, 10, rows),
            "ozone": np.linspace(50, 80, rows),
            "us_aqi": np.arange(rows, dtype=float),
        }
    )


class FakeFeatureStore:
    def __init__(self) -> None:
        self.kwargs = None

    def get_or_create_feature_group(self, **kwargs):
        self.kwargs = kwargs
        return object()


class FeatureStoreIntegrationTest(unittest.TestCase):
    def test_hopsworks_settings_default_cert_folder_is_workspace_local(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HOPSWORKS_PROJECT": "test_project",
                "HOPSWORKS_API_KEY": "test_key",
                "HOPSWORKS_HOST": "eu-west.cloud.hopsworks.ai",
            },
            clear=True,
        ):
            settings = load_hopsworks_settings(ROOT / "does-not-exist.env")

        self.assertEqual(settings.host, "eu-west.cloud.hopsworks.ai")
        self.assertEqual(settings.cert_folder, ROOT / ".hopsworks-certs")

    def test_clear_known_blackhole_proxy_only_removes_blocked_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "HTTP_PROXY": "http://proxy.example:8080",
            },
            clear=True,
        ):
            _clear_known_blackhole_proxy()

            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy.example:8080")

    def test_feature_group_metadata_uses_locked_name_keys_and_event_time(self) -> None:
        feature_store = FakeFeatureStore()

        get_or_create_hourly_feature_group(feature_store)

        self.assertEqual(feature_store.kwargs["name"], FEATURE_GROUP_NAME)
        self.assertEqual(feature_store.kwargs["version"], FEATURE_GROUP_VERSION)
        self.assertEqual(feature_store.kwargs["primary_key"], FEATURE_GROUP_PRIMARY_KEY)
        self.assertEqual(feature_store.kwargs["primary_key"], ["city", "event_time_utc"])
        self.assertEqual(feature_store.kwargs["event_time"], "event_time_utc")
        self.assertEqual(feature_store.kwargs["time_travel_format"], FEATURE_GROUP_TIME_TRAVEL_FORMAT)
        self.assertEqual(feature_store.kwargs["hudi_precombine_key"], "event_time_utc")
        self.assertFalse(feature_store.kwargs["online_enabled"])

    def test_live_feature_group_metadata_is_feature_only_group(self) -> None:
        feature_store = FakeFeatureStore()

        get_or_create_live_feature_group(feature_store)

        self.assertEqual(feature_store.kwargs["name"], LIVE_FEATURE_GROUP_NAME)
        self.assertEqual(feature_store.kwargs["version"], LIVE_FEATURE_GROUP_VERSION)
        self.assertEqual(feature_store.kwargs["primary_key"], ["city", "event_time_utc"])
        self.assertEqual(feature_store.kwargs["time_travel_format"], FEATURE_GROUP_TIME_TRAVEL_FORMAT)
        self.assertFalse(feature_store.kwargs["online_enabled"])

    def test_prepare_feature_group_frame_adds_city_and_naive_timestamps(self) -> None:
        observations = fixture_observations(rows=260)
        features = build_features(observations)
        targets = add_aqi_targets(observations)
        dataset = features.merge(targets, on=["event_time_utc", "event_time_local"]).dropna()

        prepared = prepare_feature_group_frame(dataset)

        self.assertEqual(prepared["city"].iloc[0], "Karachi")
        self.assertIsNone(prepared["event_time_utc"].dt.tz)
        self.assertIn("target_aqi_day1", prepared.columns)

    def test_readback_validation_detects_missing_targets(self) -> None:
        observed = pd.DataFrame(
            {
                "city": ["Karachi"],
                "event_time_utc": [pd.Timestamp("2026-01-01")],
                "feature_schema_version": [1],
                "target_aqi_day1": [None],
                "target_aqi_day2": [80.0],
                "target_aqi_day3": [90.0],
                "us_aqi": [70.0],
            }
        )
        result = validate_feature_store_readback(observed, observed)

        self.assertFalse(result.passed)
        self.assertEqual(result.missing_target_counts["target_aqi_day1"], 1)


if __name__ == "__main__":
    unittest.main()
