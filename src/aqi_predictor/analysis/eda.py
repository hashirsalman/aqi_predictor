"""Reproducible exploratory data analysis for the Karachi AQI backfill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.data.schemas import AIR_QUALITY_VARIABLE_NAMES, WEATHER_VARIABLE_NAMES


DEFAULT_BACKFILL_PATH = PROJECT_ROOT / "data" / "processed" / "karachi_open_meteo_hourly_backfill.csv"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
DEFAULT_EDA_REPORT_PATH = PROJECT_ROOT / "reports" / "EDA_SUMMARY.md"

POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
]


def load_backfill_dataset(path: Path = DEFAULT_BACKFILL_PATH) -> pd.DataFrame:
    """Load the local Phase 2 staging dataset for EDA."""

    if not path.exists():
        raise FileNotFoundError(
            f"Backfill dataset not found at {path}. Run `python scripts/backfill.py` first."
        )

    frame = pd.read_csv(path, parse_dates=["event_time_utc", "event_time_local"])
    return frame.sort_values("event_time_utc").reset_index(drop=True)


def compute_eda_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute EDA tables and statistics without mutating model features."""

    required = {"event_time_utc", "event_time_local", "us_aqi"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EDA input is missing required columns: {missing}")

    local_time = frame["event_time_local"]
    enriched = frame.copy()
    enriched["hour"] = local_time.dt.hour
    enriched["day_of_week"] = local_time.dt.dayofweek
    enriched["month"] = local_time.dt.month
    enriched["date"] = local_time.dt.date
    enriched["aqi_roll_24h_mean"] = enriched["us_aqi"].rolling(24, min_periods=24).mean()
    enriched["aqi_roll_24h_std"] = enriched["us_aqi"].rolling(24, min_periods=24).std()

    numeric_columns = [
        column
        for column in (*WEATHER_VARIABLE_NAMES, *AIR_QUALITY_VARIABLE_NAMES)
        if column in enriched.columns
    ]
    pearson = enriched[numeric_columns].corr(method="pearson")["us_aqi"].drop("us_aqi")
    spearman = enriched[numeric_columns].corr(method="spearman")["us_aqi"].drop("us_aqi")

    lagged_correlations: dict[str, dict[str, float | None]] = {}
    for column in POLLUTANT_COLUMNS:
        if column not in enriched.columns:
            continue
        lagged_correlations[column] = {}
        for lag in (1, 6, 24):
            corr = enriched[column].shift(lag).corr(enriched["us_aqi"])
            lagged_correlations[column][f"lag_{lag}h"] = None if pd.isna(corr) else round(float(corr), 4)

    daily = enriched.groupby("date")["us_aqi"].agg(["mean", "max"])
    high_aqi_thresholds = {
        "unhealthy_for_sensitive_groups_or_worse": 101,
        "unhealthy_or_worse": 151,
        "very_unhealthy_or_worse": 201,
        "hazardous": 301,
    }
    high_aqi_events = {
        name: {
            "hour_count": int((enriched["us_aqi"] >= threshold).sum()),
            "day_count": int((daily["max"] >= threshold).sum()),
        }
        for name, threshold in high_aqi_thresholds.items()
    }

    return {
        "rows": int(len(enriched)),
        "start_local": enriched["event_time_local"].iloc[0].isoformat(),
        "end_local": enriched["event_time_local"].iloc[-1].isoformat(),
        "aqi_distribution": _describe(enriched["us_aqi"]),
        "hourly_aqi_mean": _series_to_float_dict(enriched.groupby("hour")["us_aqi"].mean()),
        "weekly_aqi_mean": _series_to_float_dict(enriched.groupby("day_of_week")["us_aqi"].mean()),
        "monthly_aqi_mean": _series_to_float_dict(enriched.groupby("month")["us_aqi"].mean()),
        "high_aqi_events": high_aqi_events,
        "missing_percent": {
            column: round(float(enriched[column].isna().mean() * 100), 3)
            for column in numeric_columns
        },
        "pearson_correlation_with_us_aqi": _series_to_float_dict(pearson.sort_values(ascending=False)),
        "spearman_correlation_with_us_aqi": _series_to_float_dict(spearman.sort_values(ascending=False)),
        "lagged_pollutant_correlation_with_us_aqi": lagged_correlations,
        "notes": [
            "Correlations are exploratory and not causal claims.",
            "Rolling EDA statistics are backward-looking and are not target features by themselves.",
            "No future forecast inputs are created by this EDA step.",
        ],
    }


def run_eda(
    input_path: Path = DEFAULT_BACKFILL_PATH,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    report_path: Path = DEFAULT_EDA_REPORT_PATH,
) -> dict[str, Any]:
    """Run EDA and write figures plus report artifacts."""

    frame = load_backfill_dataset(input_path)
    summary = compute_eda_summary(frame)

    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _plot_aqi_distribution(frame, figure_dir / "eda_aqi_distribution.png")
    _plot_aqi_timeseries(frame, figure_dir / "eda_aqi_timeseries_rolling.png")
    _plot_seasonality(frame, figure_dir / "eda_aqi_seasonality.png")
    _plot_correlation_heatmap(frame, figure_dir / "eda_correlation_heatmap.png")
    _plot_pollutant_relationships(frame, figure_dir / "eda_pollutant_relationships.png")

    metrics_path = metrics_dir / "eda_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(_render_markdown_report(summary), encoding="utf-8")
    return summary


def _plot_aqi_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(frame["us_aqi"], bins=40, color="#3b82f6", edgecolor="white")
    plt.title("Karachi US AQI Distribution")
    plt.xlabel("US AQI")
    plt.ylabel("Hourly count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_aqi_timeseries(frame: pd.DataFrame, output_path: Path) -> None:
    daily = frame.set_index("event_time_local")["us_aqi"].resample("D").mean()
    rolling_mean = daily.rolling(14, min_periods=7).mean()
    rolling_std = daily.rolling(14, min_periods=7).std()

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily.values, label="Daily mean US AQI", linewidth=1)
    plt.plot(rolling_mean.index, rolling_mean.values, label="14-day rolling mean", linewidth=2)
    plt.fill_between(
        rolling_std.index,
        (rolling_mean - rolling_std).values,
        (rolling_mean + rolling_std).values,
        alpha=0.18,
        label="Rolling mean +/- 1 std",
    )
    plt.title("Karachi Daily US AQI With Rolling Trend")
    plt.xlabel("Date")
    plt.ylabel("US AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_seasonality(frame: pd.DataFrame, output_path: Path) -> None:
    local_time = frame["event_time_local"]
    plot_frame = frame.assign(
        hour=local_time.dt.hour,
        day_of_week=local_time.dt.dayofweek,
        month=local_time.dt.month,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_frame.groupby("hour")["us_aqi"].mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("Mean AQI By Hour")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("US AQI")

    plot_frame.groupby("day_of_week")["us_aqi"].mean().plot(ax=axes[1], marker="o", color="#10b981")
    axes[1].set_title("Mean AQI By Day Of Week")
    axes[1].set_xlabel("0=Mon, 6=Sun")

    plot_frame.groupby("month")["us_aqi"].mean().plot(ax=axes[2], marker="o", color="#f59e0b")
    axes[2].set_title("Mean AQI By Month")
    axes[2].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_correlation_heatmap(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [
        column
        for column in ("us_aqi", *POLLUTANT_COLUMNS, *WEATHER_COLUMNS)
        if column in frame.columns
    ]
    corr = frame[columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_pollutant_relationships(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [column for column in POLLUTANT_COLUMNS if column in frame.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, columns, strict=False):
        sample = frame[[column, "us_aqi"]].dropna()
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)
        ax.scatter(sample[column], sample["us_aqi"], s=8, alpha=0.25)
        ax.set_title(f"{column} vs US AQI")
        ax.set_xlabel(column)
        ax.set_ylabel("US AQI")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _render_markdown_report(summary: dict[str, Any]) -> str:
    pearson_top = list(summary["pearson_correlation_with_us_aqi"].items())[:8]
    monthly = summary["monthly_aqi_mean"]
    highest_month = max(monthly, key=monthly.get)
    lowest_month = min(monthly, key=monthly.get)

    lines = [
        "# EDA Summary - Karachi US AQI",
        "",
        "This report is generated by `python scripts/run_eda.py` from the Phase 2 local backfill staging dataset.",
        "",
        "## Dataset",
        "",
        f"- Rows: `{summary['rows']}` hourly observations.",
        f"- Local coverage: `{summary['start_local']}` through `{summary['end_local']}`.",
        f"- US AQI median: `{summary['aqi_distribution']['median']}`.",
        f"- US AQI min/max: `{summary['aqi_distribution']['min']}` / `{summary['aqi_distribution']['max']}`.",
        "",
        "## High-AQI Events",
        "",
    ]
    for name, values in summary["high_aqi_events"].items():
        lines.append(
            f"- `{name}`: `{values['hour_count']}` hourly records, `{values['day_count']}` days with daily max crossing the threshold."
        )

    lines.extend(
        [
            "",
            "## Seasonality Notes",
            "",
            f"- Highest average AQI month in this backfill: `{highest_month}`.",
            f"- Lowest average AQI month in this backfill: `{lowest_month}`.",
            "- Hourly, weekly, and monthly seasonality tables are stored in `reports/metrics/eda_summary.json`.",
            "",
            "## Strongest Pearson Correlations With US AQI",
            "",
        ]
    )
    for feature, value in pearson_top:
        lines.append(f"- `{feature}`: `{value}`")

    lines.extend(
        [
            "",
            "## Important Caution",
            "",
            "These findings are exploratory. Correlation is not causation, and no future forecast inputs or target-window values are created in this EDA phase.",
            "",
            "## Generated Figures",
            "",
            "- `reports/figures/eda_aqi_distribution.png`",
            "- `reports/figures/eda_aqi_timeseries_rolling.png`",
            "- `reports/figures/eda_aqi_seasonality.png`",
            "- `reports/figures/eda_correlation_heatmap.png`",
            "- `reports/figures/eda_pollutant_relationships.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def _describe(series: pd.Series) -> dict[str, float]:
    return {
        "min": round(float(series.min()), 4),
        "p25": round(float(series.quantile(0.25)), 4),
        "median": round(float(series.median()), 4),
        "p75": round(float(series.quantile(0.75)), 4),
        "max": round(float(series.max()), 4),
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
    }


def _series_to_float_dict(series: pd.Series) -> dict[str, float]:
    return {str(key): round(float(value), 4) for key, value in series.dropna().items()}
