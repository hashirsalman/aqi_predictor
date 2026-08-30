"""Historical Open-Meteo backfill utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from aqi_predictor.config import AppConfig, PROJECT_ROOT, load_config
from aqi_predictor.data.open_meteo_client import OpenMeteoClient, OpenMeteoDataset
from aqi_predictor.data.schemas import AIR_QUALITY_VARIABLE_NAMES, WEATHER_VARIABLE_NAMES
from aqi_predictor.data.validation import validate_hourly_observations

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillPaths:
    """Filesystem locations used for local development backfill artifacts."""

    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "open_meteo"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    metrics_dir: Path = PROJECT_ROOT / "reports" / "metrics"


@dataclass(frozen=True)
class DateChunk:
    """Inclusive date range for one API backfill chunk."""

    start_date: date
    end_date: date

    @property
    def cache_name(self) -> str:
        return f"open_meteo_karachi_{self.start_date.isoformat()}_{self.end_date.isoformat()}.csv"


def default_backfill_window(config: AppConfig | None = None, archive_lag_days: int = 7) -> tuple[date, date]:
    """Return the default approximate historical window ending before archive lag."""

    resolved_config = config or load_config()
    years = int(resolved_config.raw["data"]["historical_years"])
    timezone_name = str(resolved_config.raw["project"]["timezone"])
    end_date = datetime.now(ZoneInfo(timezone_name)).date() - timedelta(days=archive_lag_days)
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=years) + pd.DateOffset(days=1)).date()
    return start_date, end_date


def build_date_chunks(start_date: date, end_date: date, chunk_days: int = 31) -> list[DateChunk]:
    """Split an inclusive date range into inclusive chunks."""

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")

    chunks: list[DateChunk] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        chunks.append(DateChunk(start_date=current, end_date=chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def run_historical_backfill(
    start_date: date | None = None,
    end_date: date | None = None,
    chunk_days: int = 31,
    refresh_cache: bool = False,
    paths: BackfillPaths | None = None,
    client: OpenMeteoClient | None = None,
) -> dict[str, Any]:
    """Fetch, cache, combine, and validate historical Karachi observations."""

    config = load_config()
    resolved_start, resolved_end = (
        (start_date, end_date)
        if start_date is not None and end_date is not None
        else default_backfill_window(config)
    )
    if resolved_start is None or resolved_end is None:
        raise ValueError("start_date and end_date must either both be provided or both omitted")

    resolved_paths = paths or BackfillPaths()
    resolved_paths.raw_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths.processed_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    resolved_client = client or OpenMeteoClient(config)
    chunks = build_date_chunks(resolved_start, resolved_end, chunk_days=chunk_days)
    chunk_results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    failed_chunks: list[dict[str, str]] = []

    for chunk in chunks:
        cache_path = resolved_paths.raw_dir / chunk.cache_name
        try:
            if cache_path.exists() and not refresh_cache:
                LOGGER.info("Loading cached backfill chunk %s", cache_path)
                frame = pd.read_csv(
                    cache_path,
                    parse_dates=["event_time_utc", "event_time_local"],
                )
                source = "cache"
            else:
                LOGGER.info("Fetching backfill chunk %s to %s", chunk.start_date, chunk.end_date)
                dataset = resolved_client.fetch_merged_historical_sample(
                    chunk.start_date, chunk.end_date
                )
                frame = dataset.frame
                frame.to_csv(cache_path, index=False)
                source = "api"

            frames.append(frame)
            chunk_results.append(
                {
                    "start_date": chunk.start_date.isoformat(),
                    "end_date": chunk.end_date.isoformat(),
                    "rows": int(len(frame)),
                    "source": source,
                    "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - we need to log failed chunks and continue audit.
            LOGGER.exception("Backfill chunk failed for %s to %s", chunk.start_date, chunk.end_date)
            failed_chunks.append(
                {
                    "start_date": chunk.start_date.isoformat(),
                    "end_date": chunk.end_date.isoformat(),
                    "error": str(exc),
                }
            )

    if failed_chunks:
        failure_report = {"failed_chunks": failed_chunks, "completed_chunks": chunk_results}
        failure_path = resolved_paths.metrics_dir / "backfill_failed_chunks.json"
        failure_path.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        raise RuntimeError(f"Historical backfill failed for {len(failed_chunks)} chunks")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("event_time_utc")
        .drop_duplicates(subset=["event_time_utc"], keep="last")
        .reset_index(drop=True)
    )

    output_path = resolved_paths.processed_dir / "karachi_open_meteo_hourly_backfill.csv"
    combined.to_csv(output_path, index=False)

    required_columns = (*WEATHER_VARIABLE_NAMES, *AIR_QUALITY_VARIABLE_NAMES)
    units = _collect_known_units()
    validation_summary = validate_hourly_observations(combined, units, required_columns)
    quality_report = build_backfill_quality_report(
        combined,
        validation_summary.to_dict(),
        chunk_results=chunk_results,
        output_path=output_path,
        start_date=resolved_start,
        end_date=resolved_end,
    )

    report_path = resolved_paths.metrics_dir / "historical_backfill_quality_report.json"
    report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    return quality_report


def build_backfill_quality_report(
    frame: pd.DataFrame,
    validation_summary: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    output_path: Path,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Create the Phase 2 data-quality report."""

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"event_time_utc", "event_time_local"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    distribution = {
        column: {
            "min": _finite_or_none(frame[column].min()),
            "p25": _finite_or_none(frame[column].quantile(0.25)),
            "median": _finite_or_none(frame[column].median()),
            "p75": _finite_or_none(frame[column].quantile(0.75)),
            "max": _finite_or_none(frame[column].max()),
            "mean": _finite_or_none(frame[column].mean()),
        }
        for column in numeric_columns
    }

    return {
        "city": "Karachi",
        "country": "Pakistan",
        "source": "Open-Meteo weather archive and air-quality API",
        "aqi_standard": "US AQI via us_aqi",
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "local_artifact": str(output_path.relative_to(PROJECT_ROOT)),
        "local_artifact_role": "development/staging only; not the production Feature Store",
        "chunk_count": len(chunk_results),
        "chunks": chunk_results,
        "validation_summary": validation_summary,
        "duplicate_timestamps_after_deduplication": int(frame["event_time_utc"].duplicated().sum()),
        "missing_percentage_per_feature": validation_summary["missing_percent"],
        "numeric_distribution": distribution,
        "imputation_policy": "No blanket fillna(0) or interpolation applied in Phase 2. Missingness is reported; feature/target phases will decide defensible handling.",
        "date_coverage": {
            "start_utc": validation_summary["start_utc"],
            "end_utc": validation_summary["end_utc"],
            "start_local": validation_summary["start_local"],
            "end_local": validation_summary["end_local"],
        },
    }


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _collect_known_units() -> dict[str, str]:
    return {
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "rain": "mm",
        "surface_pressure": "hPa",
        "cloud_cover": "%",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "°",
        "wind_gusts_10m": "km/h",
        "pm10": "μg/m³",
        "pm2_5": "μg/m³",
        "carbon_monoxide": "μg/m³",
        "nitrogen_dioxide": "μg/m³",
        "sulphur_dioxide": "μg/m³",
        "ozone": "μg/m³",
        "us_aqi": "US AQI",
    }
