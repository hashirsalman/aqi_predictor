"""Canonical future daily-average AQI target construction."""

from __future__ import annotations

import pandas as pd

from aqi_predictor.features.feature_contract import TARGET_COLUMNS


TARGET_WINDOWS = {
    "target_aqi_day1": (1, 24),
    "target_aqi_day2": (25, 48),
    "target_aqi_day3": (49, 72),
}


def add_aqi_targets(observations: pd.DataFrame) -> pd.DataFrame:
    """Add Day +1, Day +2, and Day +3 future average US AQI targets."""

    required = {"event_time_utc", "event_time_local", "us_aqi"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Target input is missing required columns: {missing}")

    frame = observations.sort_values("event_time_utc").reset_index(drop=True).copy()
    _assert_hourly_continuity(frame)

    for target_name, (start_hour, end_hour) in TARGET_WINDOWS.items():
        shifted_values = [
            frame["us_aqi"].shift(-offset) for offset in range(start_hour, end_hour + 1)
        ]
        target_matrix = pd.concat(shifted_values, axis=1)
        valid_count = target_matrix.notna().sum(axis=1)
        frame[target_name] = target_matrix.mean(axis=1)
        frame[f"{target_name}_valid"] = valid_count == (end_hour - start_hour + 1)
        frame.loc[~frame[f"{target_name}_valid"], target_name] = pd.NA

    return frame[["event_time_utc", "event_time_local", *TARGET_COLUMNS, *(f"{name}_valid" for name in TARGET_COLUMNS)]]


def supervised_targets_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Build targets and keep only rows where all horizons are valid."""

    targets = add_aqi_targets(observations)
    valid_columns = [f"{name}_valid" for name in TARGET_COLUMNS]
    return targets.loc[targets[valid_columns].all(axis=1)].reset_index(drop=True)


def _assert_hourly_continuity(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Cannot build targets from an empty frame")
    expected = pd.date_range(
        start=frame["event_time_utc"].iloc[0],
        end=frame["event_time_utc"].iloc[-1],
        freq="h",
        tz="UTC",
    )
    actual = pd.DatetimeIndex(frame["event_time_utc"])
    missing_hours = len(expected.difference(actual))
    if missing_hours:
        raise ValueError(f"Target input has {missing_hours} missing hourly timestamps")
