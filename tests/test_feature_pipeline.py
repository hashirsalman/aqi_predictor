"""Tests for the live hourly feature pipeline helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.feature_store.feature_group import prepare_live_feature_group_frame
from aqi_predictor.features.engineering import build_features


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


class LiveFeaturePipelineTest(unittest.TestCase):
    def test_prepare_live_frame_is_feature_only_for_unknown_future_outcomes(self) -> None:
        features = build_features(fixture_observations()).dropna().tail(1)

        prepared = prepare_live_feature_group_frame(features)

        self.assertEqual(prepared["city"].iloc[0], "Karachi")
        self.assertIsNone(prepared["event_time_utc"].dt.tz)
        self.assertNotIn("target_aqi_day1", prepared.columns)
        self.assertNotIn("target_aqi_day2", prepared.columns)
        self.assertNotIn("target_aqi_day3", prepared.columns)


if __name__ == "__main__":
    unittest.main()
