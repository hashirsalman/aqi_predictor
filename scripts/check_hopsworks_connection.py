"""Check Hopsworks credentials and Feature Store connectivity without printing secrets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.feature_store.hopsworks_client import get_feature_store, load_hopsworks_settings
from aqi_predictor.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    settings = load_hopsworks_settings()
    feature_store = get_feature_store()
    print(
        json.dumps(
            {
                "project_configured": bool(settings.project),
                "api_key_configured": bool(settings.api_key),
                "host_configured": bool(settings.host),
                "feature_store_connected": feature_store is not None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
