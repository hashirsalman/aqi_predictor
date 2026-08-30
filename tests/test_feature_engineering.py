"""Tests for canonical feature engineering."""

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

from aqi_predictor.features.engineering import build_features


def fixture_observations(rows: int = 200) -> pd.DataFrame:
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
            "wind_direction_10m": [359.0] * rows,
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


class FeatureEngineeringTest(unittest.TestCase):
    def test_lag_and_change_features_use_past_values(self) -> None:
        features = build_features(fixture_observations())

        self.assertEqual(features.loc[24, "us_aqi_lag_24h"], 0.0)
        self.assertEqual(features.loc[24, "aqi_change_1h"], 1.0)
        self.assertEqual(features.loc[24, "aqi_roll_mean_24h"], 12.5)

    def test_wind_direction_is_encoded_circularly(self) -> None:
        features = build_features(fixture_observations())

        self.assertAlmostEqual(features.loc[0, "wind_dir_sin"], np.sin(np.deg2rad(359.0)))
        self.assertAlmostEqual(features.loc[0, "wind_dir_cos"], np.cos(np.deg2rad(359.0)))

    def test_missing_hour_is_rejected(self) -> None:
        observations = fixture_observations().drop(index=3).reset_index(drop=True)

        with self.assertRaises(ValueError):
            build_features(observations)


if __name__ == "__main__":
    unittest.main()
