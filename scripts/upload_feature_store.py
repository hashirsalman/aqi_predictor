"""Upload Phase 4 features/targets to Hopsworks Feature Store and validate read-back."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.feature_store.feature_group import upload_features_to_hopsworks
from aqi_predictor.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    result = upload_features_to_hopsworks()
    print(json.dumps({"passed": result.passed, "validation": result.__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
