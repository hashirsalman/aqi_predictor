"""Read latest live feature rows from Hopsworks Feature Store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from aqi_predictor.feature_store.feature_group import get_or_create_live_feature_group
from aqi_predictor.feature_store.hopsworks_client import get_feature_store
from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS


@dataclass(frozen=True)
class LatestFeatureRow:
    """Latest live feature vector plus source metadata."""

    features: pd.DataFrame
    city: str
    event_time_utc: str
    event_time_local: str
    current_aqi: float
    feature_schema_version: int | None


def load_latest_live_features(feature_store: Any | None = None) -> LatestFeatureRow:
    """Load the latest Karachi live feature row from the Hopsworks live Feature Group."""

    fs = feature_store or get_feature_store()
    feature_group = get_or_create_live_feature_group(fs)
    frame = feature_group.read(dataframe_type="pandas")
    return latest_live_features_from_frame(frame)


def latest_live_features_from_frame(frame: pd.DataFrame) -> LatestFeatureRow:
    """Extract and validate the latest live feature row from a DataFrame."""

    if frame.empty:
        raise ValueError("Live Feature Group is empty. Run the hourly feature pipeline first.")

    required_columns = {
        "city",
        "event_time_utc",
        "event_time_local",
        "feature_schema_version",
        *CANONICAL_FEATURE_COLUMNS,
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Live Feature Group is missing required inference columns: {missing}")

    working = frame.copy()
    working["event_time_utc"] = pd.to_datetime(working["event_time_utc"], utc=True, errors="coerce")
    working = working.dropna(subset=["event_time_utc"])
    if working.empty:
        raise ValueError("Live Feature Group has no parseable event_time_utc values.")

    latest = working.sort_values("event_time_utc").iloc[-1]
    latest_event_time_utc = latest["event_time_utc"]
    feature_values = latest.loc[list(CANONICAL_FEATURE_COLUMNS)]
    if feature_values.isna().any():
        missing_features = feature_values[feature_values.isna()].index.tolist()
        raise ValueError(f"Latest live feature row has missing model inputs: {missing_features}")

    features = pd.DataFrame([feature_values.to_dict()], columns=list(CANONICAL_FEATURE_COLUMNS))
    return LatestFeatureRow(
        features=features,
        city=str(latest["city"]),
        event_time_utc=latest_event_time_utc.isoformat(),
        event_time_local=latest_event_time_utc.tz_convert("Asia/Karachi").isoformat(),
        current_aqi=float(latest["us_aqi"]),
        feature_schema_version=int(latest["feature_schema_version"])
        if pd.notna(latest["feature_schema_version"])
        else None,
    )
