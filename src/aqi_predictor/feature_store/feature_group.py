"""Hopsworks Feature Group creation, upload, and read-back validation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.feature_store.hopsworks_client import get_feature_store
from aqi_predictor.features.feature_contract import (
    CANONICAL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMNS,
)

LOGGER = logging.getLogger(__name__)

FEATURE_GROUP_NAME = "karachi_aqi_hourly_features"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_PRIMARY_KEY = ["city", "event_time_utc"]
FEATURE_GROUP_TIME_TRAVEL_FORMAT = "HUDI"
LIVE_FEATURE_GROUP_NAME = "karachi_aqi_live_features"
LIVE_FEATURE_GROUP_VERSION = 1


@dataclass(frozen=True)
class FeatureStoreValidationResult:
    """Read-back validation output for uploaded Hopsworks data."""

    expected_rows: int
    observed_rows: int
    expected_columns: list[str]
    observed_columns: list[str]
    duplicate_event_rows: int
    min_event_time_utc: str | None
    max_event_time_utc: str | None
    all_zero_columns: list[str]
    missing_target_counts: dict[str, int]

    @property
    def passed(self) -> bool:
        return (
            self.expected_rows == self.observed_rows
            and not self.duplicate_event_rows
            and not self.all_zero_columns
            and not any(self.missing_target_counts.values())
            and set(self.expected_columns).issubset(self.observed_columns)
        )


def load_feature_target_staging_data(
    path: Path = PROJECT_ROOT / "data" / "processed" / "karachi_features_targets.csv",
) -> pd.DataFrame:
    """Load the local Phase 4 staging artifact for Feature Store upload."""

    if not path.exists():
        raise FileNotFoundError(
            f"Feature/target staging file not found at {path}. Run `python scripts/build_features.py` first."
        )
    return pd.read_csv(path, parse_dates=["event_time_utc", "event_time_local"])


def prepare_feature_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for Hopsworks upload without leaking secrets or local-only state."""

    required = {"event_time_utc", "event_time_local", *CANONICAL_FEATURE_COLUMNS, *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature Store upload frame is missing columns: {missing}")

    prepared = frame[["event_time_utc", "event_time_local", *CANONICAL_FEATURE_COLUMNS, *TARGET_COLUMNS]].copy()
    prepared.insert(0, "city", "Karachi")
    prepared["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    prepared["event_time_utc"] = pd.to_datetime(prepared["event_time_utc"], utc=True).dt.tz_localize(None)
    prepared["event_time_local"] = pd.to_datetime(prepared["event_time_local"], utc=True).dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
    return prepared


def prepare_live_feature_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare live/current feature-only rows for Hopsworks serving/inference."""

    required = {"event_time_utc", "event_time_local", *CANONICAL_FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Live Feature Store upload frame is missing columns: {missing}")

    prepared = frame[["event_time_utc", "event_time_local", *CANONICAL_FEATURE_COLUMNS]].copy()
    prepared.insert(0, "city", "Karachi")
    prepared["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    for integer_column in ("hour", "day_of_week", "day_of_month", "month", "day_of_year", "is_weekend", "feature_schema_version"):
        prepared[integer_column] = prepared[integer_column].astype("int64")
    prepared["event_time_utc"] = pd.to_datetime(prepared["event_time_utc"], utc=True).dt.tz_localize(None)
    prepared["event_time_local"] = pd.to_datetime(prepared["event_time_local"], utc=True).dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
    return prepared


def get_or_create_hourly_feature_group(feature_store: Any) -> Any:
    """Get or create the Karachi hourly AQI feature group."""

    return feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Karachi hourly Open-Meteo weather, air-quality, engineered features, "
            "and Day+1/Day+2/Day+3 daily-average US AQI targets. "
            "AQI standard: US AQI. Source timestamps stored as UTC event time."
        ),
        primary_key=FEATURE_GROUP_PRIMARY_KEY,
        event_time="event_time_utc",
        time_travel_format=FEATURE_GROUP_TIME_TRAVEL_FORMAT,
        hudi_precombine_key="event_time_utc",
        online_enabled=False,
        statistics_config=True,
    )


def get_or_create_live_feature_group(feature_store: Any) -> Any:
    """Get or create the Karachi live/current feature-only group for inference."""

    return feature_store.get_or_create_feature_group(
        name=LIVE_FEATURE_GROUP_NAME,
        version=LIVE_FEATURE_GROUP_VERSION,
        description=(
            "Karachi latest Open-Meteo live/current feature rows for serving AQI predictions. "
            "This Feature Group intentionally stores features only because future Day+1/Day+2/Day+3 "
            "target outcomes are unknown at hourly ingestion time."
        ),
        primary_key=FEATURE_GROUP_PRIMARY_KEY,
        event_time="event_time_utc",
        time_travel_format=FEATURE_GROUP_TIME_TRAVEL_FORMAT,
        hudi_precombine_key="event_time_utc",
        online_enabled=False,
        statistics_config=True,
    )


def upload_features_to_hopsworks(
    frame: pd.DataFrame | None = None,
    feature_store: Any | None = None,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> FeatureStoreValidationResult:
    """Upload engineered features/targets and validate by reading them back."""

    source_frame = frame if frame is not None else load_feature_target_staging_data()
    upload_frame = prepare_feature_group_frame(source_frame)
    fs = feature_store or get_feature_store()
    feature_group = get_or_create_hourly_feature_group(fs)

    _bounded_retry(
        lambda: feature_group.insert(
            upload_frame,
            operation="upsert",
            write_options={"wait_for_job": True},
        ),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        operation_name="Hopsworks feature group insert",
    )

    read_back = _bounded_retry(
        feature_group.read,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        operation_name="Hopsworks feature group read-back",
    )
    result = validate_feature_store_readback(upload_frame, read_back)
    _write_upload_report(result)
    if not result.passed:
        raise RuntimeError(f"Hopsworks read-back validation failed: {result}")
    return result


def validate_feature_store_readback(
    expected_frame: pd.DataFrame, observed_frame: pd.DataFrame
) -> FeatureStoreValidationResult:
    """Verify that Hopsworks contains the uploaded schema and rows."""

    expected_columns = list(expected_frame.columns)
    observed_columns = list(observed_frame.columns)
    expected_key_count = expected_frame[["city", "event_time_utc"]].drop_duplicates().shape[0]
    observed_key_count = observed_frame[["city", "event_time_utc"]].drop_duplicates().shape[0]
    duplicate_count = int(observed_frame[["city", "event_time_utc"]].duplicated().sum())

    numeric_columns = [
        column
        for column in expected_columns
        if column in observed_frame.columns and pd.api.types.is_numeric_dtype(observed_frame[column])
    ]
    all_zero_columns = [
        column
        for column in numeric_columns
        if column not in TARGET_COLUMNS and bool((observed_frame[column].fillna(0) == 0).all())
    ]
    missing_target_counts = {
        target: int(observed_frame[target].isna().sum()) if target in observed_frame.columns else -1
        for target in TARGET_COLUMNS
    }

    event_times = pd.to_datetime(observed_frame["event_time_utc"], errors="coerce")
    return FeatureStoreValidationResult(
        expected_rows=int(expected_key_count),
        observed_rows=int(observed_key_count),
        expected_columns=expected_columns,
        observed_columns=observed_columns,
        duplicate_event_rows=duplicate_count,
        min_event_time_utc=None if event_times.dropna().empty else event_times.min().isoformat(),
        max_event_time_utc=None if event_times.dropna().empty else event_times.max().isoformat(),
        all_zero_columns=all_zero_columns,
        missing_target_counts=missing_target_counts,
    )


def _bounded_retry(
    func: Any,
    max_retries: int,
    retry_backoff_seconds: float,
    operation_name: str,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            LOGGER.info("%s attempt %s/%s", operation_name, attempt, max_retries)
            return func()
        except Exception as exc:  # noqa: BLE001 - cloud clients raise several transient exception types.
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning("%s failed: %s; retrying in %.1fs", operation_name, exc, sleep_seconds)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{operation_name} failed after {max_retries} attempts") from last_error


def _write_upload_report(result: FeatureStoreValidationResult) -> None:
    report_path = PROJECT_ROOT / "reports" / "metrics" / "hopsworks_feature_store_upload_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "feature_group_name": FEATURE_GROUP_NAME,
                "feature_group_version": FEATURE_GROUP_VERSION,
                "passed": result.passed,
                "validation": result.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
