"""Run the Streamlit AQI dashboard locally."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from aqi_predictor.config import PROJECT_ROOT


if __name__ == "__main__":
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
            ]
        )
    )

