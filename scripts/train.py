"""Run Phase 7 model training and evaluation."""

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
from aqi_predictor.pipelines.training_pipeline import run_training_pipeline


def main() -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    configure_logging()
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
        if os.environ.get(proxy_name) == "http://127.0.0.1:9":
            os.environ.pop(proxy_name, None)

    result = run_training_pipeline()
    print(
        json.dumps(
            {
                "metrics_path": str(result.metrics_path),
                "summary_path": str(result.summary_path),
                "best_models_path": str(result.best_models_path),
                "artifact_dir": str(result.artifact_dir),
                "best_models": result.best_models[
                    ["horizon", "model", "rmse", "mae", "r2", "beats_persistence_validation"]
                ].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
