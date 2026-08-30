"""Tests for chronological training/evaluation helpers."""

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

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS
from aqi_predictor.models.baselines import persistence_predict
from aqi_predictor.models.candidates import build_candidate_models
from aqi_predictor.models.evaluation import chronological_split, regression_metrics
from aqi_predictor.pipelines.training_pipeline import _prepare_training_frame


class TrainingPipelineTest(unittest.TestCase):
    def test_chronological_split_has_no_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "event_time_utc": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
                "value": range(100),
            }
        )
        splits = chronological_split(frame)

        self.assertLess(splits.train["event_time_utc"].max(), splits.validation["event_time_utc"].min())
        self.assertLess(splits.validation["event_time_utc"].max(), splits.test["event_time_utc"].min())
        self.assertEqual(len(splits.train), 70)
        self.assertEqual(len(splits.validation), 15)
        self.assertEqual(len(splits.test), 15)

    def test_persistence_uses_rolling_day_aqi_when_available(self) -> None:
        frame = pd.DataFrame({"aqi_roll_mean_24h": [80.0, 90.0], "us_aqi": [1.0, 2.0]})

        predictions = persistence_predict(frame)

        np.testing.assert_array_equal(predictions, np.array([80.0, 90.0]))

    def test_regression_metrics_contains_required_metrics(self) -> None:
        metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))

        self.assertEqual(set(metrics), {"rmse", "mae", "r2"})
        self.assertGreater(metrics["rmse"], 0)

    def test_prepare_training_frame_rejects_missing_required_columns(self) -> None:
        with self.assertRaises(ValueError):
            _prepare_training_frame(pd.DataFrame({"event_time_utc": []}))

    def test_prepare_training_frame_accepts_complete_contract(self) -> None:
        row = {"event_time_utc": pd.Timestamp("2026-01-01T00:00Z")}
        row.update({feature: 1.0 for feature in CANONICAL_FEATURE_COLUMNS})
        row.update({target: 2.0 for target in TARGET_COLUMNS})
        frame = pd.DataFrame([row])

        prepared = _prepare_training_frame(frame)

        self.assertEqual(len(prepared), 1)

    def test_candidate_set_includes_real_pytorch_model(self) -> None:
        candidates = build_candidate_models()

        self.assertIn("pytorch_mlp", candidates)
        self.assertIn("TorchMLPRegressor", repr(candidates["pytorch_mlp"]))


if __name__ == "__main__":
    unittest.main()
