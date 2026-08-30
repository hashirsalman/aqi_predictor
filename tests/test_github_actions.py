from pathlib import Path

import yaml


WORKFLOW_DIR = Path(".github/workflows")


def _load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / name
    assert path.exists(), f"Missing workflow file: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_hourly_feature_workflow_has_required_schedule_and_manual_trigger():
    workflow = _load_workflow("feature_pipeline.yml")

    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"][0]["cron"] == "17 * * * *"


def test_hourly_feature_workflow_runs_only_feature_ingestion_script():
    workflow_text = (WORKFLOW_DIR / "feature_pipeline.yml").read_text(encoding="utf-8")

    assert "python scripts/run_feature_pipeline.py" in workflow_text
    assert "python scripts/train.py" not in workflow_text
    assert "python scripts/register_models.py" not in workflow_text


def test_daily_training_workflow_has_required_schedule_and_manual_trigger():
    workflow = _load_workflow("training_pipeline.yml")

    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"][0]["cron"] == "32 1 * * *"


def test_daily_training_workflow_trains_then_registers_models():
    workflow_text = (WORKFLOW_DIR / "training_pipeline.yml").read_text(encoding="utf-8")

    train_index = workflow_text.index("python scripts/train.py")
    register_index = workflow_text.index("python scripts/register_models.py")

    assert train_index < register_index


def test_hopsworks_secrets_are_referenced_but_not_hard_coded():
    for workflow_name in ["feature_pipeline.yml", "training_pipeline.yml"]:
        workflow_text = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")

        assert "${{ secrets.HOPSWORKS_API_KEY }}" in workflow_text
        assert "${{ secrets.HOPSWORKS_PROJECT }}" in workflow_text
        assert "${{ secrets.HOPSWORKS_HOST }}" in workflow_text
        assert "eu-west.cloud.hopsworks.ai" not in workflow_text
        assert "c.app.hopsworks.ai" not in workflow_text


def test_ci_workflow_runs_tests_without_hopsworks_secrets():
    workflow_text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert "python -m pytest" in workflow_text
    assert "HOPSWORKS_API_KEY" not in workflow_text

