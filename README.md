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

Phase 4 is the current implementation baseline. The repository has been bootstrapped, includes Open-Meteo ingestion/backfill, generates reproducible EDA outputs, and now builds canonical features plus exact Day +1, Day +2, and Day +3 US AQI targets.

For the detailed engineering handoff, read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

For the authoritative Codex implementation contract, read [README_CODEX_MASTER_AQI_REVISED_2026-08-29.md](README_CODEX_MASTER_AQI_REVISED_2026-08-29.md).

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

## Detailed Report

The final project report will live at `reports/FINAL_REPORT.md`. It has not been written yet because data collection, EDA, modeling, automation, and deployment have not started.
