from dataclasses import dataclass

import pandas as pd

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS
from aqi_predictor.inference.live_features import latest_live_features_from_frame
from aqi_predictor.inference.model_loader import LoadedRegistryModel
from aqi_predictor.inference.predictor import _predict_one_horizon


@dataclass
class ConstantEstimator:
    value: float

    def predict(self, frame):
        assert list(frame.columns) == list(CANONICAL_FEATURE_COLUMNS)
        return [self.value]


def _live_frame() -> pd.DataFrame:
    row = {column: 1.0 for column in CANONICAL_FEATURE_COLUMNS}
    row.update(
        {
            "city": "Karachi",
            "event_time_utc": "2026-08-29T18:00:00Z",
            "event_time_local": "2026-08-29T23:00:00+05:00",
            "feature_schema_version": 1,
            "us_aqi": 82.0,
        }
    )
    return pd.DataFrame([row])


def test_latest_live_features_selects_newest_row_and_feature_contract():
    older = _live_frame()
    newer = _live_frame()
    older["event_time_utc"] = "2026-08-29T17:00:00Z"
    frame = pd.concat([newer, older], ignore_index=True)

    latest = latest_live_features_from_frame(frame)

    assert latest.city == "Karachi"
    assert latest.current_aqi == 82.0
    assert list(latest.features.columns) == list(CANONICAL_FEATURE_COLUMNS)
    assert latest.event_time_utc == "2026-08-29T18:00:00+00:00"
    assert latest.event_time_local == "2026-08-29T23:00:00+05:00"


def test_predict_one_horizon_returns_model_version_rmse_and_alert():
    latest = latest_live_features_from_frame(_live_frame())
    model = LoadedRegistryModel(
        horizon="day1",
        registry_name="karachi_aqi_day1",
        registry_version=3,
        estimator=ConstantEstimator(155.2),
        feature_columns=list(CANONICAL_FEATURE_COLUMNS),
        metrics={"validation_rmse": 6.5, "test_rmse": 3.6},
        model_family="ridge",
    )

    prediction = _predict_one_horizon("day1", model, latest)

    assert prediction["rounded_aqi"] == 155
    assert prediction["alert"]["category"] == "Unhealthy"
    assert prediction["model"]["registry_version"] == 3
    assert prediction["model"]["validation_rmse"] == 6.5
