"""Hopsworks Model Registry integration for selected AQI models."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.feature_store.feature_group import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from aqi_predictor.feature_store.hopsworks_client import get_model_registry
from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS

MODEL_NAME_PREFIX = "karachi_aqi"


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Summary for one registered model artifact."""

    horizon: str
    model_family: str
    registry_name: str
    registry_version: int | None
    local_package_path: Path
    validation_rmse: float
    test_rmse: float


def register_selected_models(
    metrics_path: Path = PROJECT_ROOT / "reports" / "metrics" / "model_metrics.csv",
    best_models_path: Path = PROJECT_ROOT / "reports" / "metrics" / "best_models.json",
    source_artifact_dir: Path = PROJECT_ROOT / "models" / "phase7_local",
    package_dir: Path = PROJECT_ROOT / "models" / "registry_packages",
    report_path: Path = PROJECT_ROOT / "reports" / "metrics" / "model_registry_report.json",
    model_registry: Any | None = None,
) -> list[RegisteredModelInfo]:
    """Package and register the selected model for each forecast horizon."""

    metrics = pd.read_csv(metrics_path)
    best_models = json.loads(best_models_path.read_text(encoding="utf-8"))
    registry = model_registry or get_model_registry()
    registered: list[RegisteredModelInfo] = []

    for selected in best_models:
        horizon = selected["horizon"]
        model_family = selected["model"]
        artifact_path = source_artifact_dir / f"{model_family}_{horizon}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Selected model artifact does not exist: {artifact_path}")

        horizon_package_dir = package_dir / horizon
        _prepare_registry_package(
            package_dir=horizon_package_dir,
            source_model_path=artifact_path,
            selected=selected,
            metrics=metrics,
        )

        test_row = metrics[
            (metrics["horizon"] == horizon)
            & (metrics["model"] == model_family)
            & (metrics["split"] == "test")
        ].iloc[0]
        registry_name = f"{MODEL_NAME_PREFIX}_{horizon}"
        model = registry.sklearn.create_model(
            name=registry_name,
            metrics={
                "validation_rmse": float(selected["rmse"]),
                "validation_mae": float(selected["mae"]),
                "validation_r2": float(selected["r2"]),
                "test_rmse": float(test_row["rmse"]),
                "test_mae": float(test_row["mae"]),
                "test_r2": float(test_row["r2"]),
                "beats_persistence_validation": 1.0
                if bool(selected["beats_persistence_validation"])
                else 0.0,
            },
            description=(
                f"Karachi US AQI {horizon} direct-regression model. "
                f"Selected model family: {model_family}. "
                f"Training source: Hopsworks Feature Group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}. "
                "Inputs use observed/current and historical features only; no future forecasts."
            ),
        )
        saved = model.save(
            str(horizon_package_dir),
            keep_original_files=True,
            upload_configuration={"simultaneous_uploads": 1},
        )
        registered.append(
            RegisteredModelInfo(
                horizon=horizon,
                model_family=model_family,
                registry_name=registry_name,
                registry_version=getattr(saved, "version", None),
                local_package_path=horizon_package_dir,
                validation_rmse=float(selected["rmse"]),
                test_rmse=float(test_row["rmse"]),
            )
        )

    _write_registry_report(registered, report_path)
    return registered


def _prepare_registry_package(
    package_dir: Path,
    source_model_path: Path,
    selected: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_path, package_dir / "model.joblib")
    (package_dir / "feature_columns.json").write_text(
        json.dumps(list(CANONICAL_FEATURE_COLUMNS), indent=2),
        encoding="utf-8",
    )
    horizon_metrics = metrics[metrics["horizon"] == selected["horizon"]].to_dict(orient="records")
    (package_dir / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected,
                "all_horizon_metrics": horizon_metrics,
                "feature_group_name": FEATURE_GROUP_NAME,
                "feature_group_version": FEATURE_GROUP_VERSION,
                "target": selected["target"],
                "input_policy": "Observed/current and historical features only. No future weather, pollutant, or AQI forecasts.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Karachi AQI {selected['horizon']} Model Package",
                "",
                f"Selected model family: `{selected['model']}`.",
                f"Validation RMSE: `{selected['rmse']:.4f}`.",
                "",
                "This package is registered to Hopsworks Model Registry.",
                "The model predicts direct daily-average US AQI using only observed/current and historical features.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_registry_report(registered: list[RegisteredModelInfo], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "registered_models": [
                    {
                        "horizon": row.horizon,
                        "model_family": row.model_family,
                        "registry_name": row.registry_name,
                        "registry_version": row.registry_version,
                        "local_package_path": str(row.local_package_path),
                        "validation_rmse": row.validation_rmse,
                        "test_rmse": row.test_rmse,
                    }
                    for row in registered
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
