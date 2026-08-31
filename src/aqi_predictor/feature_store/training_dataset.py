"""Training dataset retrieval helpers for later modeling phases."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from aqi_predictor.feature_store.feature_group import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from aqi_predictor.feature_store.hopsworks_client import get_feature_store

LOGGER = logging.getLogger(__name__)

DEFAULT_TRAINING_READ_ATTEMPTS = 3
DEFAULT_TRAINING_READ_RETRY_DELAY_SECONDS = 20.0


def fetch_training_dataset(
    feature_store: Any | None = None,
    *,
    max_attempts: int = DEFAULT_TRAINING_READ_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_TRAINING_READ_RETRY_DELAY_SECONDS,
) -> pd.DataFrame:
    """Fetch the Karachi AQI feature group from Hopsworks for training.

    Hopsworks free-tier Query Service reads can occasionally fail with transient
    Arrow Flight / Query Service errors. The training pipeline therefore uses a
    small bounded retry budget instead of immediately failing the daily workflow.
    This deliberately does not fall back to local CSV files, because Hopsworks
    remains the production Feature Store.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    fs = feature_store or get_feature_store()
    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            LOGGER.info(
                "Reading Hopsworks training feature group %s v%s with pandas backend (attempt %s/%s)",
                FEATURE_GROUP_NAME,
                FEATURE_GROUP_VERSION,
                attempt,
                max_attempts,
            )
            frame = feature_group.read(dataframe_type="pandas")
            return frame.sort_values("event_time_utc").reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - exercised through fake objects in tests.
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = retry_delay_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Hopsworks training data read failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                attempt,
                max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        "Could not read training data from Hopsworks Feature Store after "
        f"{max_attempts} attempt(s). This can happen when the Hopsworks Query "
        "Service/free tier is temporarily unavailable. Existing registered "
        "models are preserved because training stops before registration."
    ) from last_error
