"""Phase 0 bootstrap checks."""

from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class Phase0BootstrapTest(unittest.TestCase):
    def test_package_imports_and_config_loads(self) -> None:
        from aqi_predictor import load_config

        config = load_config()

        self.assertEqual(config.city, "Karachi")
        self.assertEqual(config.timezone, "Asia/Karachi")
        self.assertEqual(config.aqi_standard, "us_aqi")
        self.assertEqual(config.target_names, ("day1", "day2", "day3"))

    def test_constants_match_locked_architecture(self) -> None:
        from aqi_predictor import constants

        self.assertEqual(constants.DATA_SOURCE, "open_meteo")
        self.assertEqual(constants.FEATURE_STORE_PROVIDER, "hopsworks")
        self.assertEqual(constants.MODEL_REGISTRY_PROVIDER, "hopsworks")
        self.assertEqual(
            [(h.name, h.start_hour, h.end_hour) for h in constants.TARGET_HORIZONS],
            [("day1", 1, 24), ("day2", 25, 48), ("day3", 49, 72)],
        )

    def test_env_file_is_gitignored_if_present(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)
        result = subprocess.run(
            ["git", "check-ignore", ".env"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
