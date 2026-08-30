"""Tests for Model Registry packaging helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS
from aqi_predictor.models.registry import _prepare_registry_package


class ModelRegistryPackagingTest(unittest.TestCase):
    def test_prepare_registry_package_contains_model_features_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_model = tmp_path / "ridge_day1.joblib"
            source_model.write_bytes(b"fake-model")
            package_dir = tmp_path / "package"
            selected = {
                "horizon": "day1",
                "model": "ridge",
                "target": "target_aqi_day1",
                "rmse": 1.23,
                "mae": 1.0,
                "r2": 0.5,
                "beats_persistence_validation": True,
            }
            metrics = pd.DataFrame(
                [
                    {
                        "horizon": "day1",
                        "model": "ridge",
                        "split": "validation",
                        "rmse": 1.23,
                    }
                ]
            )

            _prepare_registry_package(package_dir, source_model, selected, metrics)

            self.assertTrue((package_dir / "model.joblib").exists())
            feature_columns = json.loads((package_dir / "feature_columns.json").read_text())
            self.assertEqual(feature_columns, list(CANONICAL_FEATURE_COLUMNS))
            metadata = json.loads((package_dir / "metrics.json").read_text())
            self.assertEqual(metadata["target"], "target_aqi_day1")


if __name__ == "__main__":
    unittest.main()
