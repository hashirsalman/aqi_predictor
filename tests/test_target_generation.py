"""Tests for direct three-horizon AQI target generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.features.target_generation import add_aqi_targets


class TargetGenerationTest(unittest.TestCase):
    def test_targets_are_exact_future_hour_windows(self) -> None:
        event_time_utc = pd.date_range("2026-01-01T00:00Z", periods=80, freq="h")
        frame = pd.DataFrame(
            {
                "event_time_utc": event_time_utc,
                "event_time_local": event_time_utc.tz_convert("Asia/Karachi"),
                "us_aqi": range(80),
            }
        )

        targets = add_aqi_targets(frame)

        self.assertEqual(targets.loc[0, "target_aqi_day1"], sum(range(1, 25)) / 24)
        self.assertEqual(targets.loc[0, "target_aqi_day2"], sum(range(25, 49)) / 24)
        self.assertEqual(targets.loc[0, "target_aqi_day3"], sum(range(49, 73)) / 24)
        self.assertTrue(targets.loc[0, "target_aqi_day3_valid"])
        self.assertFalse(targets.loc[79, "target_aqi_day1_valid"])


if __name__ == "__main__":
    unittest.main()
