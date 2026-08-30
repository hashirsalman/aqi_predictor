from fastapi.testclient import TestClient

from aqi_predictor.api.main import app, get_prediction_service


class FakePredictionService:
    def health(self):
        return {"status": "ok", "service": "fake"}

    def predict(self, force_model_refresh: bool = False):
        return {
            "city": "Karachi",
            "force_model_refresh": force_model_refresh,
            "predictions": [
                {
                    "horizon": "day1",
                    "predicted_aqi": 80.0,
                    "model": {"registry_version": 1, "validation_rmse": 6.5},
                }
            ],
        }


def test_health_endpoint_uses_prediction_service():
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService()
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_predictions():
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService()
    try:
        response = TestClient(app).get("/predict?force_model_refresh=true")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Karachi"
    assert body["force_model_refresh"] is True
    assert body["predictions"][0]["model"]["validation_rmse"] == 6.5


def test_predict_endpoint_returns_503_on_service_failure():
    class BrokenService:
        def predict(self, force_model_refresh: bool = False):
            raise RuntimeError("registry unavailable")

    app.dependency_overrides[get_prediction_service] = lambda: BrokenService()
    try:
        response = TestClient(app).get("/predict")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Prediction service is unavailable" in response.json()["detail"]

