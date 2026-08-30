from datetime import UTC, datetime

from dashboard.api_client import DEFAULT_FASTAPI_BASE_URL, normalize_base_url
from dashboard.formatting import model_metric_table, observation_age_hours, prediction_table, strongest_alert


def _payload():
    return {
        "source_observation": {
            "event_time_utc": "2026-08-30T19:00:00+00:00",
            "current_aqi": 64,
            "alert": {"category": "Moderate", "alert_level": "notice", "is_health_alert": False},
        },
        "current_conditions": {
            "pollutants": {
                "pm2_5": {"label": "PM2.5", "value": 10.123, "unit": "ug/m3"},
                "pm10": {"label": "PM10", "value": None, "unit": "ug/m3"},
            },
            "weather": {
                "temperature_2m": {"label": "Temperature", "value": 29.9, "unit": "C"},
            },
        },
        "predictions": [
            {
                "horizon": "day1",
                "target": "Day +1 average AQI",
                "rounded_aqi": 63,
                "alert": {"category": "Moderate", "alert_level": "notice", "is_health_alert": False},
                "model": {
                    "registry_name": "karachi_aqi_day1",
                    "registry_version": 1,
                    "model_family": "ridge",
                    "validation_rmse": 6.48,
                    "validation_mae": 4.9,
                    "validation_r2": 0.84,
                    "test_rmse": 3.6,
                },
            },
            {
                "horizon": "day2",
                "target": "Day +2 average AQI",
                "rounded_aqi": 155,
                "alert": {"category": "Unhealthy", "alert_level": "alert", "is_health_alert": True},
                "model": {
                    "registry_name": "karachi_aqi_day2",
                    "registry_version": 1,
                    "model_family": "gradient_boosting",
                    "validation_rmse": 15.16,
                    "validation_mae": 11.5,
                    "validation_r2": 0.16,
                    "test_rmse": 8.1,
                },
            },
        ],
    }


def test_normalize_base_url_uses_default_for_blank_value():
    assert normalize_base_url("   ") == DEFAULT_FASTAPI_BASE_URL


def test_normalize_base_url_removes_trailing_slash():
    assert normalize_base_url("http://localhost:8000/") == "http://localhost:8000"


def test_prediction_table_includes_model_version_and_rmse():
    table = prediction_table(_payload())

    assert table.loc[0, "Registry Version"] == 1
    assert table.loc[0, "Validation RMSE"] == 6.48


def test_model_metric_table_includes_registry_names():
    table = model_metric_table(_payload())

    assert table.loc[1, "Registry Name"] == "karachi_aqi_day2"


def test_observation_age_hours_from_payload():
    age = observation_age_hours(
        _payload(),
        now=datetime(2026, 8, 30, 21, 30, tzinfo=UTC),
    )

    assert age == 2.5


def test_strongest_alert_finds_unhealthy_prediction():
    alert = strongest_alert(_payload())

    assert alert["category"] == "Unhealthy"
    assert alert["is_health_alert"] is True


def test_condition_table_skips_missing_values():
    from dashboard.formatting import condition_table

    table = condition_table(_payload(), "pollutants")

    assert table.to_dict("records") == [{"Metric": "PM2.5", "Value": 10.12, "Unit": "ug/m3"}]


def test_forecast_summary_uses_rounded_predictions():
    from dashboard.formatting import forecast_summary

    summary = forecast_summary(_payload())

    assert summary == {"average": 109.0, "maximum": 155, "minimum": 63}
