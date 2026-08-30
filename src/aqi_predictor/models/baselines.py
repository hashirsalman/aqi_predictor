"""Baseline AQI forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict(frame: pd.DataFrame) -> np.ndarray:
    """Predict future daily-average AQI as the latest known 24h rolling AQI average.

    This implements a conservative "tomorrow behaves like the recent day" baseline.
    If the rolling feature is unavailable, it falls back to the current observed US AQI.
    """

    source_column = "aqi_roll_mean_24h" if "aqi_roll_mean_24h" in frame.columns else "us_aqi"
    return frame[source_column].astype(float).to_numpy()
