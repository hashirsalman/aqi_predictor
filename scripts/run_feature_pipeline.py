"""Run the live hourly feature ingestion pipeline."""

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
from aqi_predictor.pipelines.feature_pipeline import run_live_feature_pipeline


def main() -> int:
    configure_logging()
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
        if os.environ.get(proxy_name) == "http://127.0.0.1:9":
            os.environ.pop(proxy_name, None)

    result = run_live_feature_pipeline()
    print(
        json.dumps(
            {
                "fetched_rows": result.fetched_rows,
                "candidate_feature_rows": result.candidate_feature_rows,
                "uploaded_rows": result.uploaded_rows,
                "event_time_utc": result.event_time_utc,
                "event_time_local": result.event_time_local,
                "staleness_hours": result.staleness_hours,
                "report_path": str(result.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
