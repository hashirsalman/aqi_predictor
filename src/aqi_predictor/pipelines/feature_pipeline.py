"""Live hourly feature ingestion pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.data.open_meteo_client import OpenMeteoClient
from aqi_predictor.data.schemas import AIR_QUALITY_VARIABLE_NAMES, WEATHER_VARIABLE_NAMES
from aqi_predictor.data.validation import validate_hourly_observations
from aqi_predictor.feature_store.feature_group import (
    get_or_create_live_feature_group,
    prepare_live_feature_group_frame,
)
from aqi_predictor.feature_store.hopsworks_client import get_feature_store
from aqi_predictor.features.engineering import supervised_feature_frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveFeaturePipelineResult:
    """Result of one live/current feature ingestion run."""

    fetched_rows: int
    candidate_feature_rows: int
    uploaded_rows: int
    event_time_utc: str
    event_time_local: str
    staleness_hours: float
    report_path: Path


def run_live_feature_pipeline(
    past_days: int = 10,
    max_staleness_hours: float = 24.0,
    feature_store: Any | None = None,
    report_path: Path = PROJECT_ROOT / "reports" / "metrics" / "live_feature_pipeline_report.json",
) -> LiveFeaturePipelineResult:
    """Fetch recent observations, build the latest feature row, and upsert it to Hopsworks."""

    LOGGER.info("Fetching recent Open-Meteo observations for live feature pipeline")
    dataset = OpenMeteoClient().fetch_merged_recent_sample(past_days=past_days)
    required_columns = (*WEATHER_VARIABLE_NAMES, *AIR_QUALITY_VARIABLE_NAMES)
    validation = validate_hourly_observations(dataset.frame, dataset.units, required_columns)

    candidate_features = supervised_feature_frame(dataset.frame)
    if candidate_features.empty:
        raise ValueError(
            f"No complete live feature rows available from past_days={past_days}. "
            "Increase the recent window so one-week lag features can be computed."
        )

    latest_feature = candidate_features.tail(1).copy()
    event_time_utc = pd.to_datetime(latest_feature["event_time_utc"].iloc[0], utc=True)
    event_time_local = pd.to_datetime(latest_feature["event_time_local"].iloc[0])
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    staleness_hours = float((now_utc - event_time_utc).total_seconds() / 3600)
    if staleness_hours < -1e-6:
        raise ValueError(f"Latest feature row is in the future: {event_time_utc.isoformat()}")
    if staleness_hours > max_staleness_hours:
        raise ValueError(
            f"Latest Open-Meteo observation is stale by {staleness_hours:.2f} hours, "
            f"above the allowed {max_staleness_hours:.2f} hours."
        )

    upload_frame = prepare_live_feature_group_frame(latest_feature)
    fs = feature_store or get_feature_store()
    feature_group = get_or_create_live_feature_group(fs)
    LOGGER.info("Upserting latest live feature row to Hopsworks at %s", event_time_utc.isoformat())
    feature_group.insert(
        upload_frame,
        operation="upsert",
        write_options={"wait_for_job": True},
    )

    result = LiveFeaturePipelineResult(
        fetched_rows=validation.rows,
        candidate_feature_rows=len(candidate_features),
        uploaded_rows=len(upload_frame),
        event_time_utc=event_time_utc.isoformat(),
        event_time_local=event_time_local.isoformat(),
        staleness_hours=staleness_hours,
        report_path=report_path,
    )
    _write_live_report(result, validation.to_dict())
    return result


def _write_live_report(result: LiveFeaturePipelineResult, validation: dict[str, Any]) -> None:
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "fetched_rows": result.fetched_rows,
                "candidate_feature_rows": result.candidate_feature_rows,
                "uploaded_rows": result.uploaded_rows,
                "event_time_utc": result.event_time_utc,
                "event_time_local": result.event_time_local,
                "staleness_hours": result.staleness_hours,
                "targets_policy": "Live Feature Group rows intentionally omit target_aqi_day1/day2/day3 because future outcomes are not known at ingestion time.",
                "validation": validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
