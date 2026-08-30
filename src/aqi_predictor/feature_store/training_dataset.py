"""Training dataset retrieval helpers for later modeling phases."""

from __future__ import annotations

from typing import Any

import pandas as pd

from aqi_predictor.feature_store.feature_group import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from aqi_predictor.feature_store.hopsworks_client import get_feature_store


def fetch_training_dataset(feature_store: Any | None = None) -> pd.DataFrame:
    """Fetch the Karachi AQI feature group from Hopsworks for training."""

    fs = feature_store or get_feature_store()
    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )
    frame = feature_group.read()
    return frame.sort_values("event_time_utc").reset_index(drop=True)
