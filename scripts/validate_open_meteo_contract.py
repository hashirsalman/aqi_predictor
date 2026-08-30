"""Validate the Phase 1 Open-Meteo data contract for Karachi."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aqi_predictor.config import load_config
from aqi_predictor.data.open_meteo_client import OpenMeteoClient, default_sample_window
from aqi_predictor.data.schemas import (
    AIR_QUALITY_VARIABLES,
    AIR_QUALITY_VARIABLE_NAMES,
    WEATHER_VARIABLES,
    WEATHER_VARIABLE_NAMES,
)
from aqi_predictor.data.validation import validate_hourly_observations
from aqi_predictor.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "metrics" / "open_meteo_contract_validation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    config = load_config()

    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    elif args.start_date or args.end_date:
        raise ValueError("--start-date and --end-date must be provided together")
    else:
        start_date, end_date = default_sample_window(days=args.days)

    client = OpenMeteoClient(config)
    historical = client.fetch_merged_historical_sample(start_date, end_date)
    recent = client.fetch_merged_recent_sample(past_days=2)

    required_columns = (*WEATHER_VARIABLE_NAMES, *AIR_QUALITY_VARIABLE_NAMES)
    historical_summary = validate_hourly_observations(
        historical.frame,
        historical.units,
        required_columns=required_columns,
    )

    recent_columns = set(recent.frame.columns)
    feature_contract = []
    for variable in (*WEATHER_VARIABLES, *AIR_QUALITY_VARIABLES):
        historical_available = variable.name in historical.frame.columns
        recent_available = variable.name in recent_columns
        feature_contract.append(
            {
                "feature": variable.name,
                "source_group": variable.source_group,
                "historical_available": historical_available,
                "live_current_available": recent_available,
                "same_units_semantics": historical_available and recent_available,
                "historical_unit": historical.units.get(variable.name, ""),
                "recent_unit": recent.units.get(variable.name, ""),
                "expected_unit": variable.expected_unit,
                "valid_time": variable.valid_time,
                "provenance": variable.provenance,
                "keep_candidate": historical_available and recent_available,
                "reason": "Available in both historical and recent/current Open-Meteo paths."
                if historical_available and recent_available
                else "Excluded until historical and live/current availability match.",
            }
        )

    result = {
        "city": config.city,
        "timezone": config.timezone,
        "aqi_standard": "US AQI via Open-Meteo us_aqi",
        "sample_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "historical_summary": historical_summary.to_dict(),
        "recent_rows_after_future_filter": int(len(recent.frame)),
        "recent_start_utc": None
        if recent.frame.empty
        else recent.frame["event_time_utc"].iloc[0].isoformat(),
        "recent_end_utc": None
        if recent.frame.empty
        else recent.frame["event_time_utc"].iloc[-1].isoformat(),
        "feature_availability_contract": feature_contract,
        "future_input_policy": "Future forecast timestamps are filtered out of recent/live fetches and are not model inputs.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
