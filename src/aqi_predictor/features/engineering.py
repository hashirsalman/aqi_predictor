"""Canonical backward-looking feature engineering."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS


def build_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Build model input features using only current and historical observations."""

    required = {"event_time_utc", "event_time_local", "us_aqi", "wind_direction_10m"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Feature input is missing required columns: {missing}")

    frame = observations.sort_values("event_time_utc").reset_index(drop=True).copy()
    _assert_hourly_continuity(frame)

    local_time = frame["event_time_local"]
    generated: dict[str, pd.Series | np.ndarray] = {}
    generated["hour"] = local_time.dt.hour
    generated["day_of_week"] = local_time.dt.dayofweek
    generated["day_of_month"] = local_time.dt.day
    generated["month"] = local_time.dt.month
    generated["day_of_year"] = local_time.dt.dayofyear
    generated["is_weekend"] = generated["day_of_week"].isin([5, 6]).astype(int)

    generated["hour_sin"] = np.sin(2 * math.pi * generated["hour"] / 24)
    generated["hour_cos"] = np.cos(2 * math.pi * generated["hour"] / 24)
    generated["dow_sin"] = np.sin(2 * math.pi * generated["day_of_week"] / 7)
    generated["dow_cos"] = np.cos(2 * math.pi * generated["day_of_week"] / 7)
    generated["month_sin"] = np.sin(2 * math.pi * generated["month"] / 12)
    generated["month_cos"] = np.cos(2 * math.pi * generated["month"] / 12)

    radians = np.deg2rad(frame["wind_direction_10m"] % 360)
    generated["wind_dir_sin"] = np.sin(radians)
    generated["wind_dir_cos"] = np.cos(radians)

    for lag in (1, 3, 6):
        generated[f"aqi_change_{lag}h"] = frame["us_aqi"] - frame["us_aqi"].shift(lag)
    previous_aqi = frame["us_aqi"].shift(1)
    generated["aqi_pct_change_1h"] = np.where(
        previous_aqi.abs() > 1e-9,
        (frame["us_aqi"] - previous_aqi) / previous_aqi,
        np.nan,
    )

    for window in (3, 6, 12, 24):
        generated[f"aqi_roll_mean_{window}h"] = frame["us_aqi"].rolling(window, min_periods=window).mean()
    for window in (6, 12, 24):
        generated[f"aqi_roll_std_{window}h"] = frame["us_aqi"].rolling(window, min_periods=window).std()
    generated["aqi_roll_min_24h"] = frame["us_aqi"].rolling(24, min_periods=24).min()
    generated["aqi_roll_max_24h"] = frame["us_aqi"].rolling(24, min_periods=24).max()

    for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168):
        generated[f"us_aqi_lag_{lag}h"] = frame["us_aqi"].shift(lag)

    for column in ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"):
        for lag in (1, 3, 6, 12, 24):
            generated[f"{column}_lag_{lag}h"] = frame[column].shift(lag)

    for column in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"):
        for lag in (1, 3, 6, 12, 24):
            generated[f"{column}_lag_{lag}h"] = frame[column].shift(lag)

    for column in ("pm2_5", "pm10"):
        for window in (6, 12, 24):
            generated[f"{column}_roll_mean_{window}h"] = frame[column].rolling(window, min_periods=window).mean()

    for column in ("wind_speed_10m", "relative_humidity_2m", "temperature_2m"):
        for window in (6, 24):
            generated[f"{column}_roll_mean_{window}h"] = frame[column].rolling(window, min_periods=window).mean()

    for window in (6, 24):
        generated[f"precipitation_roll_sum_{window}h"] = frame["precipitation"].rolling(window, min_periods=window).sum()

    frame = pd.concat([frame, pd.DataFrame(generated, index=frame.index)], axis=1)

    missing_features = [column for column in CANONICAL_FEATURE_COLUMNS if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Feature engineering did not create expected columns: {missing_features}")

    return frame[["event_time_utc", "event_time_local", *CANONICAL_FEATURE_COLUMNS]].copy()


def supervised_feature_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Build features and drop rows without enough historical context."""

    features = build_features(observations)
    return features.dropna(subset=list(CANONICAL_FEATURE_COLUMNS)).reset_index(drop=True)


def _assert_hourly_continuity(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Cannot build features from an empty frame")
    expected = pd.date_range(
        start=frame["event_time_utc"].iloc[0],
        end=frame["event_time_utc"].iloc[-1],
        freq="h",
        tz="UTC",
    )
    actual = pd.DatetimeIndex(frame["event_time_utc"])
    missing_hours = len(expected.difference(actual))
    if missing_hours:
        raise ValueError(f"Feature input has {missing_hours} missing hourly timestamps")
