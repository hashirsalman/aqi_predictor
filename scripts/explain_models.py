"""Generate Phase 9 SHAP explanations for selected AQI models."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.logging_utils import configure_logging
from aqi_predictor.models.explainability import run_shap_explanations


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    configure_logging()
    results = run_shap_explanations()
    print(
        json.dumps(
            [
                {
                    "horizon": result.horizon,
                    "model_family": result.model_family,
                    "sample_rows": result.sample_rows,
                    "top_5_features": result.top_features[:5],
                }
                for result in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
