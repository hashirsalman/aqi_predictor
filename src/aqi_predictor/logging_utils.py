"""Logging setup helpers."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from aqi_predictor.config import PROJECT_ROOT


DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.yaml"


def configure_logging(path: str | Path = DEFAULT_LOGGING_PATH) -> None:
    """Configure Python logging from the YAML logging file."""

    logging_path = Path(path)
    if not logging_path.exists():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return

    with logging_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    logging.config.dictConfig(config)
