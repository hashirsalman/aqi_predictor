"""Load latest selected AQI models from Hopsworks Model Registry."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from aqi_predictor.feature_store.hopsworks_client import get_model_registry
from aqi_predictor.models.registry import MODEL_NAME_PREFIX

HORIZONS = ("day1", "day2", "day3")


@dataclass(frozen=True)
class LoadedRegistryModel:
    """Loaded model object with registry metadata for API responses."""

    horizon: str
    registry_name: str
    registry_version: int | None
    estimator: Any
    feature_columns: list[str]
    metrics: dict[str, float | int | str | bool]
    model_family: str | None


def load_latest_models(model_registry: Any | None = None) -> dict[str, LoadedRegistryModel]:
    """Load the latest registered model for each required forecast horizon."""

    registry = model_registry or get_model_registry()
    return {horizon: load_latest_model_for_horizon(horizon, registry) for horizon in HORIZONS}


def load_latest_model_for_horizon(horizon: str, model_registry: Any) -> LoadedRegistryModel:
    """Load one horizon model package from Hopsworks Model Registry."""

    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon {horizon!r}; expected one of {HORIZONS}.")

    registry_name = f"{MODEL_NAME_PREFIX}_{horizon}"
    registry_model = _get_latest_registry_model(model_registry, registry_name)
    with tempfile.TemporaryDirectory(prefix=f"aqi_{horizon}_model_") as tmpdir:
        local_dir = Path(_download_registry_model(registry_model, Path(tmpdir)))
        package_dir = _resolve_package_dir(local_dir)
        estimator = joblib.load(package_dir / "model.joblib")
        feature_columns = json.loads((package_dir / "feature_columns.json").read_text(encoding="utf-8"))
        metadata = json.loads((package_dir / "metrics.json").read_text(encoding="utf-8"))

    selected = metadata.get("selected_model", {})
    metrics = _extract_registry_metrics(registry_model, selected)
    return LoadedRegistryModel(
        horizon=horizon,
        registry_name=registry_name,
        registry_version=getattr(registry_model, "version", None),
        estimator=estimator,
        feature_columns=list(feature_columns),
        metrics=metrics,
        model_family=selected.get("model"),
    )


def _get_latest_registry_model(model_registry: Any, registry_name: str) -> Any:
    """Fetch the latest model version, supporting common Hopsworks SDK shapes."""

    try:
        return model_registry.get_model(name=registry_name, version=None)
    except TypeError:
        return model_registry.get_model(registry_name)


def _download_registry_model(registry_model: Any, target_dir: Path) -> str:
    """Download a registry model and return the local directory path."""

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = registry_model.download(str(target_dir))
    except TypeError:
        downloaded = registry_model.download()
    return str(downloaded)


def _resolve_package_dir(downloaded_path: Path) -> Path:
    """Find the directory containing the packaged model files."""

    if (downloaded_path / "model.joblib").exists():
        return downloaded_path

    candidates = [path.parent for path in downloaded_path.rglob("model.joblib")]
    if not candidates:
        raise FileNotFoundError(f"Downloaded model package does not contain model.joblib: {downloaded_path}")
    return candidates[0]


def _extract_registry_metrics(registry_model: Any, selected_model: dict[str, Any]) -> dict[str, float | int | str | bool]:
    """Normalize metrics from the registry object and model package metadata."""

    raw_metrics = getattr(registry_model, "metrics", None) or {}
    normalized: dict[str, float | int | str | bool] = {}
    for key, value in raw_metrics.items():
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            normalized[key] = value

    for source_key, target_key in {
        "rmse": "validation_rmse",
        "mae": "validation_mae",
        "r2": "validation_r2",
        "beats_persistence_validation": "beats_persistence_validation",
    }.items():
        if target_key not in normalized and source_key in selected_model:
            value = selected_model[source_key]
            normalized[target_key] = float(value) if isinstance(value, (int, float)) else value

    return normalized

