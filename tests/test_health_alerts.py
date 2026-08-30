import pytest

from aqi_predictor.inference.health_alerts import build_alert, classify_aqi


@pytest.mark.parametrize(
    ("aqi", "category", "level"),
    [
        (25, "Good", "none"),
        (75, "Moderate", "notice"),
        (125, "Unhealthy for Sensitive Groups", "caution"),
        (175, "Unhealthy", "alert"),
        (250, "Very Unhealthy", "alert"),
        (350, "Hazardous", "hazardous"),
    ],
)
def test_classify_aqi_categories(aqi, category, level):
    result = classify_aqi(aqi)

    assert result.name == category
    assert result.alert_level == level


def test_build_alert_marks_hazardous_health_alert():
    alert = build_alert(350)

    assert alert["category"] == "Hazardous"
    assert alert["is_health_alert"] is True


def test_negative_aqi_is_rejected():
    with pytest.raises(ValueError):
        classify_aqi(-1)

