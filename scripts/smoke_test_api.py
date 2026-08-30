"""Run a read-only smoke test of the FastAPI prediction service internals."""

from __future__ import annotations

import json

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.inference.predictor import PredictionService


def main() -> None:
    """Load live features + latest registry models and write a smoke-test report."""

    result = PredictionService().predict(force_model_refresh=True)
    summary = {
        "city": result["city"],
        "aqi_standard": result["aqi_standard"],
        "generated_at_utc": result["generated_at_utc"],
        "source_observation": result["source_observation"],
        "predictions": [
            {
                "horizon": prediction["horizon"],
                "predicted_aqi": prediction["predicted_aqi"],
                "rounded_aqi": prediction["rounded_aqi"],
                "alert": prediction["alert"],
                "model": prediction["model"],
            }
            for prediction in result["predictions"]
        ],
        "input_policy": result["input_policy"],
    }
    report_path = PROJECT_ROOT / "reports" / "metrics" / "api_smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"FastAPI inference smoke test passed. Report written to {report_path}")
    print(
        "Predictions:",
        [
            (
                row["horizon"],
                row["rounded_aqi"],
                row["model"]["registry_name"],
                row["model"]["registry_version"],
                row["model"]["validation_rmse"],
            )
            for row in summary["predictions"]
        ],
    )


if __name__ == "__main__":
    main()

