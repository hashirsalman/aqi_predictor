"""SHAP explainability for selected AQI horizon models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS
from aqi_predictor.models.evaluation import chronological_split


@dataclass(frozen=True)
class ExplanationResult:
    """SHAP explanation output for one horizon model."""

    horizon: str
    model_family: str
    sample_rows: int
    top_features: list[dict[str, float | str]]
    csv_path: Path
    figure_path: Path


def run_shap_explanations(
    data_path: Path = PROJECT_ROOT / "data" / "processed" / "karachi_features_targets.csv",
    best_models_path: Path = PROJECT_ROOT / "reports" / "metrics" / "best_models.json",
    artifact_dir: Path = PROJECT_ROOT / "models" / "phase7_local",
    output_dir: Path = PROJECT_ROOT / "reports" / "explainability",
    max_background_rows: int = 500,
    max_explain_rows: int = 500,
) -> list[ExplanationResult]:
    """Generate mean absolute SHAP feature-importance reports for selected models."""

    frame = pd.read_csv(data_path, parse_dates=["event_time_utc", "event_time_local"])
    splits = chronological_split(frame.dropna(subset=[*CANONICAL_FEATURE_COLUMNS]))
    train_x = splits.train[list(CANONICAL_FEATURE_COLUMNS)]
    test_x = splits.test[list(CANONICAL_FEATURE_COLUMNS)]
    background = train_x.tail(min(max_background_rows, len(train_x)))
    explain_x = test_x.tail(min(max_explain_rows, len(test_x)))

    selected_models = json.loads(best_models_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExplanationResult] = []

    for selected in selected_models:
        horizon = selected["horizon"]
        model_family = selected["model"]
        model_path = artifact_dir / f"{model_family}_{horizon}.joblib"
        model = joblib.load(model_path)
        shap_values = _compute_shap_values(model, model_family, background, explain_x)
        importance = _mean_abs_importance(shap_values, list(CANONICAL_FEATURE_COLUMNS))

        csv_path = output_dir / f"shap_importance_{horizon}.csv"
        figure_path = output_dir / f"shap_importance_{horizon}.png"
        pd.DataFrame(importance).to_csv(csv_path, index=False)
        _plot_importance(importance[:20], title=f"SHAP importance - {horizon}", path=figure_path)

        results.append(
            ExplanationResult(
                horizon=horizon,
                model_family=model_family,
                sample_rows=len(explain_x),
                top_features=importance[:20],
                csv_path=csv_path,
                figure_path=figure_path,
            )
        )

    _write_explainability_summary(results, output_dir / "SHAP_SUMMARY.md")
    _write_explainability_json(results, PROJECT_ROOT / "reports" / "metrics" / "shap_summary.json")
    return results


def _compute_shap_values(
    model: Any,
    model_family: str,
    background: pd.DataFrame,
    explain_x: pd.DataFrame,
) -> np.ndarray:
    import shap

    if model_family == "ridge":
        scaler = model.named_steps["scaler"]
        estimator = model.named_steps["model"]
        transformed_background = scaler.transform(background)
        transformed_explain = scaler.transform(explain_x)
        explainer = shap.LinearExplainer(estimator, transformed_background)
        values = explainer.shap_values(transformed_explain)
        return np.asarray(values)

    if model_family in {"gradient_boosting", "random_forest"}:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(explain_x)
        return np.asarray(values)

    explainer = shap.Explainer(model.predict, background)
    values = explainer(explain_x)
    return np.asarray(values.values)


def _mean_abs_importance(shap_values: np.ndarray, feature_columns: list[str]) -> list[dict[str, float | str]]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {"feature": feature_columns[index], "mean_abs_shap": float(mean_abs[index])}
        for index in order
    ]


def _plot_importance(importance: list[dict[str, float | str]], title: str, path: Path) -> None:
    features = [str(row["feature"]) for row in importance][::-1]
    values = [float(row["mean_abs_shap"]) for row in importance][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(features, values, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_explainability_summary(results: list[ExplanationResult], path: Path) -> None:
    lines = [
        "# SHAP Explainability Summary",
        "",
        "Generated from the selected Phase 7 local model artifacts using the same chronological test split.",
        "The local feature/target file used here was previously uploaded to Hopsworks and cloud read-back validated.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.horizon}",
                "",
                f"- Model family: `{result.model_family}`",
                f"- Explanation sample rows: `{result.sample_rows}`",
                f"- CSV: `{result.csv_path}`",
                f"- Figure: `{result.figure_path}`",
                "",
                "| rank | feature | mean_abs_shap |",
                "| --- | --- | --- |",
            ]
        )
        for rank, row in enumerate(result.top_features[:10], start=1):
            lines.append(f"| {rank} | {row['feature']} | {float(row['mean_abs_shap']):.6f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_explainability_json(results: list[ExplanationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "horizon": result.horizon,
                    "model_family": result.model_family,
                    "sample_rows": result.sample_rows,
                    "top_features": result.top_features[:20],
                    "csv_path": str(result.csv_path),
                    "figure_path": str(result.figure_path),
                }
                for result in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
