"""Tests for SHAP explanation output helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.models.explainability import _mean_abs_importance, _plot_importance


class ExplainabilityTest(unittest.TestCase):
    def test_mean_abs_importance_orders_features_descending(self) -> None:
        importance = _mean_abs_importance(
            shap_values=__import__("numpy").array([[1.0, 0.1], [3.0, 0.2]]),
            feature_columns=["large", "small"],
        )

        self.assertEqual(importance[0]["feature"], "large")
        self.assertGreater(importance[0]["mean_abs_shap"], importance[1]["mean_abs_shap"])

    def test_plot_importance_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "importance.png"
            _plot_importance(
                [{"feature": "us_aqi", "mean_abs_shap": 1.0}],
                title="test",
                path=path,
            )

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
