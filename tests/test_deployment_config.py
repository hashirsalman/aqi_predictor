from pathlib import Path

import yaml


def test_render_fastapi_blueprint_uses_free_plan_and_no_hardcoded_secrets():
    path = Path("deployment/fastapi/render.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    service = config["services"][0]
    assert service["type"] == "web"
    assert service["plan"] == "free"
    assert "uvicorn aqi_predictor.api.main:app" in service["startCommand"]

    env_vars = {row["key"]: row for row in service["envVars"]}
    assert env_vars["HOPSWORKS_API_KEY"]["sync"] is False
    assert env_vars["HOPSWORKS_PROJECT"]["sync"] is False
    assert env_vars["HOPSWORKS_HOST"]["sync"] is False


def test_streamlit_deployment_notes_reference_dashboard_entrypoint_and_fastapi_url():
    text = Path("deployment/streamlit/README.md").read_text(encoding="utf-8")

    assert "dashboard/app.py" in text
    assert "FASTAPI_BASE_URL" in text
    assert "Do not make Streamlit load local models" in text

