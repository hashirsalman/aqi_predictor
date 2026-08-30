"""Training pipeline for three-horizon Karachi US AQI forecasting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.constants import TARGET_HORIZONS
from aqi_predictor.feature_store.training_dataset import fetch_training_dataset
from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS
from aqi_predictor.features.leakage_checks import assert_no_target_columns_in_features
from aqi_predictor.models.baselines import persistence_predict
from aqi_predictor.models.candidates import build_candidate_models
from aqi_predictor.models.evaluation import chronological_split, regression_metrics
from aqi_predictor.models.selection import select_best_models

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Artifacts produced by one training run."""

    metrics_path: Path
    summary_path: Path
    best_models_path: Path
    artifact_dir: Path
    metrics: pd.DataFrame
    best_models: pd.DataFrame


def run_training_pipeline(
    source_frame: pd.DataFrame | None = None,
    artifact_dir: Path = PROJECT_ROOT / "models" / "phase7_local",
    metrics_path: Path = PROJECT_ROOT / "reports" / "metrics" / "model_metrics.csv",
    summary_path: Path = PROJECT_ROOT / "reports" / "MODEL_EXPERIMENT_SUMMARY.md",
    best_models_path: Path = PROJECT_ROOT / "reports" / "metrics" / "best_models.json",
) -> TrainingPipelineResult:
    """Train/evaluate required model families using chronological validation."""

    LOGGER.info("Loading training data from Hopsworks Feature Store")
    frame = source_frame if source_frame is not None else fetch_training_dataset()
    clean = _prepare_training_frame(frame)
    splits = chronological_split(clean)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []

    for horizon in TARGET_HORIZONS:
        target = f"target_aqi_{horizon.name}"
        LOGGER.info("Training horizon %s using target %s", horizon.name, target)

        for split_name, split_frame in {"validation": splits.validation, "test": splits.test}.items():
            predictions = persistence_predict(split_frame)
            row = {
                "model": "persistence",
                "horizon": horizon.name,
                "target": target,
                "split": split_name,
                **regression_metrics(split_frame[target], predictions),
                "beats_persistence_validation": False,
                "notes": "Baseline: predicts future daily-average AQI as latest known 24h rolling AQI average.",
            }
            metrics_rows.append(row)

        candidates = build_candidate_models()
        train_x = splits.train[list(CANONICAL_FEATURE_COLUMNS)]
        train_y = splits.train[target]
        for model_name, estimator in candidates.items():
            estimator.fit(train_x, train_y)
            joblib.dump(estimator, artifact_dir / f"{model_name}_{horizon.name}.joblib")

            for split_name, split_frame in {"validation": splits.validation, "test": splits.test}.items():
                split_x = split_frame[list(CANONICAL_FEATURE_COLUMNS)]
                predictions = estimator.predict(split_x)
                metrics_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon.name,
                        "target": target,
                        "split": split_name,
                        **regression_metrics(split_frame[target], predictions),
                        "beats_persistence_validation": False,
                        "notes": _model_notes(model_name),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    metrics = _mark_persistence_comparison(metrics)
    best_models = select_best_models(metrics, split="validation")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    _write_best_models(best_models, best_models_path)
    _write_summary(
        summary_path=summary_path,
        metrics=metrics,
        best_models=best_models,
        splits={
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        },
    )

    return TrainingPipelineResult(
        metrics_path=metrics_path,
        summary_path=summary_path,
        best_models_path=best_models_path,
        artifact_dir=artifact_dir,
        metrics=metrics,
        best_models=best_models,
    )


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time_utc", *CANONICAL_FEATURE_COLUMNS, *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Training frame is missing required columns: {missing}")

    assert_no_target_columns_in_features(CANONICAL_FEATURE_COLUMNS)
    clean = frame.sort_values("event_time_utc").copy()
    clean["event_time_utc"] = pd.to_datetime(clean["event_time_utc"], utc=True)
    clean = clean.dropna(subset=[*CANONICAL_FEATURE_COLUMNS, *TARGET_COLUMNS]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("Training frame is empty after dropping missing features/targets.")
    return clean


def _mark_persistence_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    updated = metrics.copy()
    validation = updated[updated["split"] == "validation"]
    baseline_rmse = {
        row["horizon"]: row["rmse"]
        for _, row in validation[validation["model"] == "persistence"].iterrows()
    }
    updated["beats_persistence_validation"] = updated.apply(
        lambda row: bool(row["model"] != "persistence" and row["rmse"] < baseline_rmse[row["horizon"]])
        if row["split"] == "validation"
        else False,
        axis=1,
    )
    return updated


def _model_notes(model_name: str) -> str:
    notes = {
        "ridge": "Linear statistical baseline with feature scaling.",
        "random_forest": "Tree ensemble candidate.",
        "gradient_boosting": "Boosting candidate using scikit-learn gradient boosting.",
        "neural_mlp": "Small CPU-friendly neural-network/MLP candidate; TensorFlow/PyTorch can be added later as heavier optional experiment.",
    }
    return notes[model_name]


def _write_best_models(best_models: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = best_models.to_dict(orient="records")
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _write_summary(
    summary_path: Path,
    metrics: pd.DataFrame,
    best_models: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Model Experiment Summary",
        "",
        "This report is generated by `scripts/train.py`.",
        "",
        "Training data source: Hopsworks Feature Group `karachi_aqi_hourly_features`, version `1`.",
        "",
        "Input policy: observed/current and historical features only. No future weather, pollutant, or AQI forecast inputs are used.",
        "",
        "## Chronological Split",
        "",
    ]
    for split_name, split_frame in splits.items():
        lines.append(
            f"- {split_name}: {len(split_frame):,} rows, "
            f"{split_frame['event_time_utc'].min().isoformat()} to {split_frame['event_time_utc'].max().isoformat()}"
        )

    lines.extend(
        [
            "",
            "## Validation Metrics",
            "",
            _markdown_table(metrics[metrics["split"] == "validation"][
                ["model", "horizon", "rmse", "mae", "r2", "beats_persistence_validation"]
            ].sort_values(["horizon", "rmse"])),
            "",
            "## Test Metrics",
            "",
            _markdown_table(metrics[metrics["split"] == "test"][
                ["model", "horizon", "rmse", "mae", "r2"]
            ].sort_values(["horizon", "model"])),
            "",
            "## Selected Models By Validation RMSE",
            "",
            _markdown_table(best_models[
                ["horizon", "model", "rmse", "mae", "r2", "beats_persistence_validation"]
            ]),
            "",
            "These local artifacts are development outputs only. Production promotion happens later through the Hopsworks Model Registry phase.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional pandas tabulate dependency."""

    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)
