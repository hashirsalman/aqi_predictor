"""Tests for leakage-prevention guardrails."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS
from aqi_predictor.features.leakage_checks import (
    assert_canonical_features_are_safe,
    assert_no_target_columns_in_features,
)


class LeakageChecksTest(unittest.TestCase):
    def test_canonical_features_do_not_include_targets(self) -> None:
        assert_canonical_features_are_safe()
        self.assertNotIn("target_aqi_day1", CANONICAL_FEATURE_COLUMNS)

    def test_forbidden_future_columns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_target_columns_in_features(["us_aqi", "future_us_aqi"])

        with self.assertRaises(ValueError):
            assert_no_target_columns_in_features(["us_aqi", "target_aqi_day1"])


if __name__ == "__main__":
    unittest.main()
