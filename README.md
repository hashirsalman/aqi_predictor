# Pearls AQI Predictor

Pearls AQI Predictor is an end-to-end, serverless-oriented AQI forecasting project for Karachi, Pakistan.

The locked architecture is:

Karachi -> Open-Meteo -> Feature Engineering -> Hopsworks Feature Store -> Daily Training -> Hopsworks Model Registry -> FastAPI -> Streamlit

The system will predict three separate daily-average US AQI values:

- Day +1 average US AQI over future hours `t+1` through `t+24`
- Day +2 average US AQI over future hours `t+25` through `t+48`
- Day +3 average US AQI over future hours `t+49` through `t+72`

Important rules:

- Use Open-Meteo for historical and live/current weather and air-quality data.
- Use US AQI, preferably the Open-Meteo `us_aqi` field.
- Use only current and historical observed inputs at inference time.
- Do not use future weather, future pollutant, or future AQI values as model inputs.
- Use Hopsworks Feature Store and Hopsworks Model Registry for the production architecture.
- Use GitHub Actions for hourly feature ingestion and daily training.
- Use FastAPI for inference and Streamlit for the dashboard.
- Keep the budget at `$0`.

## Current Status

The project now includes the core data, feature, training, registry, automation, explainability, FastAPI inference, and Streamlit dashboard pieces.

Implemented so far:

- Open-Meteo Karachi historical/live ingestion
- data validation and EDA outputs
- canonical feature engineering and Day +1/Day +2/Day +3 target generation
- Hopsworks historical Feature Group and live Feature Group
- model training with persistence, Ridge, Random Forest, Gradient Boosting, scikit-learn MLP, and PyTorch MLP candidates
- Hopsworks Model Registry packaging/registration
- SHAP feature-importance reports
- GitHub Actions hourly feature ingestion and daily training workflows
- FastAPI inference backend with AQI health alerts
- Streamlit dashboard that calls FastAPI

Live deployment URLs:

- FastAPI backend: `https://pearls-aqi-fastapi.onrender.com`
- Streamlit dashboard: `https://aqipredictor-x3knr3fztvobzbemejnf3l.streamlit.app/`

Internal planning/handoff files such as `IMPLEMENTATION_STATUS.md` and `README_CODEX_MASTER_AQI_REVISED_2026-08-29.md` are kept locally and intentionally excluded from the public repository.

## Local Checks

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

To validate the Open-Meteo contract with a small live sample:

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_open_meteo_contract.py
```

To run the historical backfill:

```powershell
$env:PYTHONPATH = "src"
python scripts/backfill.py
```

The backfill writes local staging files under `data/raw/` and `data/processed/`. Those files are intentionally gitignored because the production Feature Store will be Hopsworks.

To run EDA from the local backfill:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_eda.py
```

EDA outputs are written to `reports/figures/`, `reports/metrics/eda_summary.json`, and `reports/EDA_SUMMARY.md`.

To build canonical features and targets:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_features.py
```

This writes `data/processed/karachi_features_targets.csv` as a local staging artifact and writes feature/target reports under `reports/metrics/`. The production Feature Store is still Hopsworks.

To check Hopsworks connectivity after installing compatible dependencies:

```powershell
$env:PYTHONPATH = "src"
python scripts/check_hopsworks_connection.py
```

To upload the engineered feature/target staging data to Hopsworks:

```powershell
$env:PYTHONPATH = "src"
python scripts/upload_feature_store.py
```

Use Python 3.11 for Hopsworks work. The local `.env` must contain `HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY`, and the correct non-secret `HOPSWORKS_HOST` for the user’s actual Hopsworks project/instance. Copy only the host name from Hopsworks, without `https://` and without any path. Do not use `c.app.hopsworks.ai`; that value was tested on 2026-08-30 and did not resolve from this environment.

To run the live hourly feature pipeline locally:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_feature_pipeline.py
```

To train models locally from Hopsworks Feature Store:

```powershell
$env:PYTHONPATH = "src"
python scripts/train.py
```

The training data reader uses a bounded retry/backoff policy for Hopsworks Query Service reads. This protects the daily workflow from occasional free-tier/transient Arrow Flight or Query Service failures, while still failing clearly if Hopsworks remains unavailable. It does not use a local CSV as a production fallback.

The PyTorch candidate is a training-only dependency so web deployments stay lightweight. Install it before running the full local training experiment:

```powershell
python -m pip install -r requirements-training.txt
```

The daily GitHub Actions training workflow installs `requirements-training.txt` automatically.

To register selected models in Hopsworks Model Registry:

```powershell
$env:PYTHONPATH = "src"
python scripts/register_models.py
```

To run the read-only FastAPI inference smoke test:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_api.py
```

This writes `reports/metrics/api_smoke_test_report.json`.

To start the FastAPI backend locally:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_api.py
```

Then open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/predict
http://127.0.0.1:8000/docs
```

The `/predict` endpoint reads the latest live feature row from Hopsworks Feature Store, loads the latest registered Day +1, Day +2, and Day +3 models from Hopsworks Model Registry, and returns predictions with model versions, RMSE values, current AQI, data timestamp, and AQI health-alert categories.

To start the Streamlit dashboard locally, first keep the FastAPI backend running in one terminal, then open a second terminal and run:

```powershell
$env:PYTHONPATH = "src;."
python scripts/run_dashboard.py
```

The dashboard calls the FastAPI `/predict` endpoint using `FASTAPI_BASE_URL` from `.env`. If `FASTAPI_BASE_URL` is blank, it defaults to:

```text
http://127.0.0.1:8000
```

The dashboard shows current observed US AQI, pollutant/weather context, source-observation freshness, Day +1/Day +2/Day +3 predicted US AQI, AQI category/alert messages, a forecast chart, and health guidance. It also includes EDA, model-comparison, and SHAP explainability tabs. Model registry names/versions, validation metrics, backend URL, and the raw API response are available in the evaluator-focused technical details tab.

## Deployment Preparation

Deployment has been prepared and manually executed by the repository owner on free-tier services.

Prepared files:

- `deployment/fastapi/render.yaml`
- `deployment/fastapi/README.md`
- `deployment/streamlit/README.md`
- `.streamlit/config.toml`

Recommended free-path candidate:

```text
FastAPI backend: Render Free Web Service
Streamlit dashboard: Streamlit Community Cloud
```

Manual deployment checkpoint:

- Do not deploy until the repository owner signs into the selected provider.
- Do not choose a paid plan.
- Do not add billing unless explicitly approved.
- Store Hopsworks credentials and `FASTAPI_BASE_URL` only as provider secrets/environment variables.
- Never commit `.env` or deployment secrets.

GitHub Actions schedule note:

- The hourly feature workflow is scheduled with cron `17 * * * *`, which means GitHub attempts to run it once per hour at minute `17` UTC.
- The daily training workflow is scheduled with cron `32 1 * * *`, which means GitHub attempts to run it once per day at `01:32` UTC.
- GitHub scheduled workflows are best-effort and may start late or occasionally be skipped by GitHub's scheduler. Manual runs do not reset the next scheduled run.
- Runs labeled `Scheduled` in the Actions UI are automatic runs. Runs labeled `Manually run by <user>` are manual runs.

FastAPI deployment needs these environment variables:

```text
PYTHONPATH=src
HOPSWORKS_API_KEY=<provider secret>
HOPSWORKS_PROJECT=<provider secret>
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_CERT_FOLDER=.hopsworks-certs
```

Streamlit deployment needs this secret after FastAPI is deployed:

```toml
FASTAPI_BASE_URL = "https://<your-fastapi-service-url>"
```

## Detailed Report

The final detailed project report lives at `reports/FINAL_REPORT.md`.

It documents:

- the locked architecture and no-future-input forecasting policy;
- Open-Meteo data ingestion and two-year historical backfill;
- feature engineering and target construction;
- Hopsworks Feature Store and Model Registry usage;
- EDA findings;
- model training/evaluation results;
- SHAP explainability;
- GitHub Actions automation;
- FastAPI and Streamlit deployment;
- hazardous AQI alerts;
- remaining limitations and final submission checklist.

Before final SHINE submission, manually verify the latest GitHub Actions hourly feature workflow, daily training workflow, FastAPI `/predict`, and Streamlit dashboard are all working.
