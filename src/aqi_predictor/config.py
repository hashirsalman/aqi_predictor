"""Configuration loading and validation for Pearls AQI Predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aqi_predictor import constants


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    raw: dict[str, Any]

    @property
    def city(self) -> str:
        return str(self.raw["project"]["city"])

    @property
    def timezone(self) -> str:
        return str(self.raw["project"]["timezone"])

    @property
    def aqi_standard(self) -> str:
        return str(self.raw["project"]["aqi_standard"])

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.raw["targets"].keys())


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate the project YAML configuration."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    config = AppConfig(raw=raw)
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Fail loudly if configuration drifts from the locked architecture."""

    project = config.raw.get("project", {})
    data = config.raw.get("data", {})
    feature_store = config.raw.get("feature_store", {})
    model_registry = config.raw.get("model_registry", {})
    automation = config.raw.get("automation", {})
    modeling = config.raw.get("modeling", {})

    expected_values = {
        "project.city": (project.get("city"), constants.CITY),
        "project.country": (project.get("country"), constants.COUNTRY),
        "project.timezone": (project.get("timezone"), constants.TIMEZONE),
        "project.aqi_standard": (project.get("aqi_standard"), constants.AQI_STANDARD),
        "data.source": (data.get("source"), constants.DATA_SOURCE),
        "data.forecast_input_allowed": (data.get("forecast_input_allowed"), False),
        "feature_store.provider": (
            feature_store.get("provider"),
            constants.FEATURE_STORE_PROVIDER,
        ),
        "model_registry.provider": (
            model_registry.get("provider"),
            constants.MODEL_REGISTRY_PROVIDER,
        ),
        "automation.feature_pipeline_cron": (
            automation.get("feature_pipeline_cron"),
            "hourly",
        ),
        "automation.training_pipeline_cron": (
            automation.get("training_pipeline_cron"),
            "daily",
        ),
        "modeling.primary_metric": (modeling.get("primary_metric"), "rmse"),
    }

    errors = [
        f"{name} expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    ]

    expected_targets = {
        horizon.name: {
            "start_hour": horizon.start_hour,
            "end_hour": horizon.end_hour,
        }
        for horizon in constants.TARGET_HORIZONS
    }
    if config.raw.get("targets") != expected_targets:
        errors.append(
            "targets must define day1/day2/day3 as 1-24, 25-48, and 49-72 hours"
        )

    if tuple(modeling.get("metrics", [])) != constants.METRICS:
        errors.append("modeling.metrics must be rmse, mae, r2 in that order")

    if errors:
        raise ValueError("Invalid locked configuration: " + "; ".join(errors))
