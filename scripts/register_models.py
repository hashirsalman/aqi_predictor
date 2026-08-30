"""Register selected Phase 7 models in Hopsworks Model Registry."""

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
from aqi_predictor.models.registry import register_selected_models


def main() -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    configure_logging()
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
        if os.environ.get(proxy_name) == "http://127.0.0.1:9":
            os.environ.pop(proxy_name, None)

    registered = register_selected_models()
    print(
        json.dumps(
            [
                {
                    "horizon": row.horizon,
                    "model_family": row.model_family,
                    "registry_name": row.registry_name,
                    "registry_version": row.registry_version,
                    "validation_rmse": row.validation_rmse,
                    "test_rmse": row.test_rmse,
                }
                for row in registered
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
