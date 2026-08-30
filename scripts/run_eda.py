"""Run Phase 3 exploratory data analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.analysis.eda import run_eda
from aqi_predictor.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    summary = run_eda()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
