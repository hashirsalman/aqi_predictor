"""Validation helpers for Open-Meteo observations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataValidationSummary:
    """A compact validation report for an hourly observation frame."""

    rows: int
    start_utc: str | None
    end_utc: str | None
    start_local: str | None
    end_local: str | None
    expected_hourly_rows: int
    missing_hour_count: int
    duplicate_event_time_count: int
    variables: list[str]
    units: dict[str, str]
    missing_percent: dict[str, float]
    us_aqi_min: float | None
    us_aqi_median: float | None
    us_aqi_max: float | None
    timezone_policy: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation summary."""

        return asdict(self)


def validate_hourly_observations(
    frame: pd.DataFrame,
    units: dict[str, str],
    required_columns: tuple[str, ...],
) -> DataValidationSummary:
    """Validate merged hourly observations and summarize data quality."""

    missing_columns = [
        column
        for column in ("event_time_utc", "event_time_local", *required_columns)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ordered = frame.sort_values("event_time_utc").reset_index(drop=True)
    duplicate_count = int(ordered["event_time_utc"].duplicated().sum())

    if ordered.empty:
        start_utc = end_utc = start_local = end_local = None
        expected_rows = missing_hours = 0
    else:
        start = ordered["event_time_utc"].iloc[0]
        end = ordered["event_time_utc"].iloc[-1]
        expected_index = pd.date_range(start=start, end=end, freq="h", tz="UTC")
        actual_index = pd.DatetimeIndex(ordered["event_time_utc"])
        missing_hours = int(len(expected_index.difference(actual_index)))
        expected_rows = int(len(expected_index))
        start_utc = start.isoformat()
        end_utc = end.isoformat()
        start_local = ordered["event_time_local"].iloc[0].isoformat()
        end_local = ordered["event_time_local"].iloc[-1].isoformat()

    missing_percent = {
        column: round(float(ordered[column].isna().mean() * 100), 3)
        for column in required_columns
    }

    aqi = pd.to_numeric(ordered["us_aqi"], errors="coerce")
    summary = DataValidationSummary(
        rows=int(len(ordered)),
        start_utc=start_utc,
        end_utc=end_utc,
        start_local=start_local,
        end_local=end_local,
        expected_hourly_rows=expected_rows,
        missing_hour_count=missing_hours,
        duplicate_event_time_count=duplicate_count,
        variables=list(required_columns),
        units=units,
        missing_percent=missing_percent,
        us_aqi_min=None if aqi.dropna().empty else float(aqi.min()),
        us_aqi_median=None if aqi.dropna().empty else float(aqi.median()),
        us_aqi_max=None if aqi.dropna().empty else float(aqi.max()),
        timezone_policy="Open-Meteo local timestamps are localized to Asia/Karachi and converted to UTC for storage/joins.",
    )

    if duplicate_count:
        raise ValueError(f"Duplicate event_time_utc rows found: {duplicate_count}")

    if missing_hours:
        raise ValueError(f"Missing hourly timestamps found: {missing_hours}")

    return summary
