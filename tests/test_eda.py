"""Tests for Phase 3 EDA summary logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.analysis.eda import compute_eda_summary


class EdaSummaryTest(unittest.TestCase):
    def test_compute_eda_summary_reports_expected_keys(self) -> None:
        event_time_utc = pd.date_range("2026-01-01T00:00Z", periods=48, freq="h")
        event_time_local = event_time_utc.tz_convert("Asia/Karachi")
        frame = pd.DataFrame(
            {
                "event_time_utc": event_time_utc,
                "event_time_local": event_time_local,
                "temperature_2m": range(48),
                "relative_humidity_2m": range(48),
                "precipitation": [0.0] * 48,
                "rain": [0.0] * 48,
                "surface_pressure": [1008.0] * 48,
                "cloud_cover": [10.0] * 48,
                "wind_speed_10m": [5.0] * 48,
                "wind_direction_10m": [180.0] * 48,
                "wind_gusts_10m": [8.0] * 48,
                "pm10": range(48),
                "pm2_5": range(48),
                "carbon_monoxide": range(48),
                "nitrogen_dioxide": range(48),
                "sulphur_dioxide": range(48),
                "ozone": range(48),
                "us_aqi": [50 + value for value in range(48)],
            }
        )

        summary = compute_eda_summary(frame)

        self.assertEqual(summary["rows"], 48)
        self.assertIn("aqi_distribution", summary)
        self.assertIn("monthly_aqi_mean", summary)
        self.assertIn("pearson_correlation_with_us_aqi", summary)
        self.assertIn("lagged_pollutant_correlation_with_us_aqi", summary)


if __name__ == "__main__":
    unittest.main()
