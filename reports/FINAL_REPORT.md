# Pearls AQI Predictor - Final Project Report

## 1. Executive Summary

Pearls AQI Predictor is an end-to-end AQI forecasting system for Karachi, Pakistan. It predicts the next three daily-average US AQI values using a serverless-oriented MLOps stack:

```text
Open-Meteo -> Feature Engineering -> Hopsworks Feature Store -> Model Training
-> Hopsworks Model Registry -> FastAPI -> Streamlit
```

The project satisfies the core internship requirement: a reproducible pipeline that collects weather and pollutant data, builds model features and targets, stores them in a cloud feature store, trains multiple forecasting models, registers selected models, automates pipeline runs, and exposes real-time predictions through an interactive dashboard.

Live deployment URLs:

- FastAPI backend: `https://pearls-aqi-fastapi.onrender.com`
- Streamlit dashboard: `https://aqipredictor-x3knr3fztvobzbemejnf3l.streamlit.app/`

## 2. Locked Forecast Objective

The project predicts US AQI directly, not pollutant concentration and not AQI category labels.

Three separate regression targets are used:

| Horizon | Target definition |
| --- | --- |
| Day +1 | Mean US AQI over future hours `t+1` through `t+24` |
| Day +2 | Mean US AQI over future hours `t+25` through `t+48` |
| Day +3 | Mean US AQI over future hours `t+49` through `t+72` |

The system uses three separate model artifacts, one per forecast horizon.

Important leakage policy:

- Inference uses only current and historical observed features.
- No future weather forecasts are used as model inputs.
- No future pollutant forecasts are used as model inputs.
- No future AQI values are used as model inputs.

## 3. Data Source

Primary source:

- Open-Meteo weather archive/current APIs
- Open-Meteo air-quality archive/current APIs

Location:

- City: Karachi, Pakistan
- Latitude: `24.8607`
- Longitude: `67.0011`
- Time zone: `Asia/Karachi`

AQI standard:

- US AQI, using Open-Meteo `us_aqi`

Selected pollutant and weather variables:

- AQI/pollutants: `us_aqi`, `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`
- Weather: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`

Open-Meteo/CAMS air-quality data may be modeled or reanalysis-derived rather than direct physical station readings. This is acceptable for this project because the same source family is used consistently for historical training and live inference.

## 4. Historical Backfill

The historical backfill collected two years of hourly Karachi data.

Backfill result:

| Item | Value |
| --- | --- |
| Requested date range | `2024-08-24` through `2026-08-23` |
| UTC coverage | `2024-08-23T19:00:00+00:00` through `2026-08-23T18:00:00+00:00` |
| Local coverage | `2024-08-24T00:00:00+05:00` through `2026-08-23T23:00:00+05:00` |
| Rows collected | `17,520` |
| Expected hourly rows | `17,520` |
| Missing hourly rows | `0` |
| Duplicate timestamps | `0` |
| US AQI min / median / max | `41.0` / `82.0` / `173.0` |

Local backfill CSV files are development/staging artifacts only. Production features are stored in Hopsworks Feature Store.

## 5. Feature Engineering And Targets

The feature contract contains `115` model input features.

Feature groups include:

- current observed pollutant and weather values;
- time-based features such as hour, day of week, day of month, month, and day of year;
- cyclic time encodings such as `hour_sin`, `hour_cos`, `month_sin`, and `month_cos`;
- wind direction sine/cosine features;
- AQI change-rate features such as `aqi_change_1h`, `aqi_change_3h`, `aqi_change_6h`, and `aqi_pct_change_1h`;
- lag features for AQI, pollutants, and weather;
- backward-looking rolling statistics, including rolling AQI mean/std/min/max and rolling pollutant/weather summaries.

Feature/target build result:

| Item | Value |
| --- | --- |
| Input rows | `17,520` |
| Engineered rows before filtering | `17,520` |
| Complete supervised rows | `17,280` |
| Dropped initial rows for lags/rolls | `168` |
| Dropped final rows for target windows | `72` |
| Feature count | `115` |
| Target columns | `target_aqi_day1`, `target_aqi_day2`, `target_aqi_day3` |
| First supervised UTC timestamp | `2024-08-30T19:00:00+00:00` |
| Last supervised UTC timestamp | `2026-08-20T18:00:00+00:00` |

All selected base variables passed the train/serve consistency check.

## 6. Feature Store

Cloud Feature Store:

- Provider: Hopsworks
- Feature Group: `karachi_aqi_hourly_features`
- Version: `1`
- Primary key: `city`, `event_time_utc`
- Time-travel format: HUDI

Cloud validation result:

| Check | Result |
| --- | --- |
| Expected rows | `17,280` |
| Observed rows | `17,280` |
| Duplicate `city + event_time_utc` rows | `0` |
| Missing Day +1 targets | `0` |
| Missing Day +2 targets | `0` |
| Missing Day +3 targets | `0` |
| All-zero feature columns | none |
| Feature Store UTC coverage | `2024-08-30T19:00:00+00:00` through `2026-08-20T18:00:00+00:00` |

The live feature pipeline writes recent/current engineered features to the Hopsworks live feature group used by FastAPI inference.

## 7. EDA Findings

EDA was performed on the two-year hourly backfill dataset.

Key results:

| Finding | Value |
| --- | --- |
| Rows analyzed | `17,520` |
| US AQI median | `82.0` |
| US AQI min / max | `41.0` / `173.0` |
| Highest average AQI month | December |
| Lowest average AQI month | September |
| Hours at US AQI 101 or worse | `3,920` |
| Hours at US AQI 151 or worse | `385` |
| Hours at US AQI 201 or worse | `0` |
| Hazardous hours, AQI 301+ | `0` |

Strongest Pearson correlations with US AQI:

| Feature | Pearson correlation |
| --- | ---: |
| `pm2_5` | `0.7201` |
| `sulphur_dioxide` | `0.5028` |
| `carbon_monoxide` | `0.4492` |
| `surface_pressure` | `0.4360` |
| `nitrogen_dioxide` | `0.3580` |
| `pm10` | `0.2581` |

Generated figures:

- `reports/figures/eda_aqi_distribution.png`
- `reports/figures/eda_aqi_timeseries_rolling.png`
- `reports/figures/eda_aqi_seasonality.png`
- `reports/figures/eda_correlation_heatmap.png`
- `reports/figures/eda_pollutant_relationships.png`

These EDA findings are exploratory and should not be interpreted as causal claims.

## 8. Model Training

Training reads historical features and targets from Hopsworks Feature Store.

The split is chronological:

| Split | Rows | Date range |
| --- | ---: | --- |
| Train | `12,096` | `2024-08-30T19:00:00+00:00` to `2026-01-16T18:00:00+00:00` |
| Validation | `2,592` | `2026-01-16T19:00:00+00:00` to `2026-05-04T18:00:00+00:00` |
| Test | `2,592` | `2026-05-04T19:00:00+00:00` to `2026-08-20T18:00:00+00:00` |

Model families implemented:

- Persistence baseline
- Ridge Regression
- Random Forest
- Gradient Boosting
- scikit-learn MLP neural network
- PyTorch MLP neural network

The local metric artifact currently contains the first five model families. The PyTorch candidate has been added to the training workflow and verified in GitHub Actions after a scikit-learn compatibility fix. If the GitHub workflow artifact is downloaded or local training is rerun with `requirements-training.txt`, this report can be refreshed with the PyTorch metric rows.

## 9. Model Evaluation

Models are evaluated with:

- RMSE
- MAE
- R2
- persistence-baseline comparison

Selected models by validation RMSE:

| Horizon | Selected model | Validation RMSE | Validation MAE | Validation R2 | Test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Day +1 | Ridge | `6.4884` | `4.9698` | `0.8451` | `3.6147` |
| Day +2 | Gradient Boosting | `15.1694` | `11.5838` | `0.1656` | `8.1169` |
| Day +3 | Gradient Boosting | `17.1039` | `13.2103` | `-0.0646` | `10.1054` |

Persistence baseline validation RMSE:

| Horizon | Persistence validation RMSE | Selected model validation RMSE | Selected model beat persistence? |
| --- | ---: | ---: | --- |
| Day +1 | `13.5043` | `6.4884` | yes |
| Day +2 | `17.3132` | `15.1694` | yes |
| Day +3 | `18.7101` | `17.1039` | yes |

The Day +3 validation R2 is slightly negative, but the selected model still improves validation RMSE over persistence. This is documented rather than hidden because AQI forecasting performance should be judged using chronological holdouts and baseline comparison, not only by maximizing R2.

## 10. Model Registry

Selected models were registered in Hopsworks Model Registry.

| Horizon | Registry name | Version | Model family | Validation RMSE | Test RMSE |
| --- | --- | ---: | --- | ---: | ---: |
| Day +1 | `karachi_aqi_day1` | `1` | Ridge | `6.4884` | `3.6147` |
| Day +2 | `karachi_aqi_day2` | `1` | Gradient Boosting | `15.1694` | `8.1169` |
| Day +3 | `karachi_aqi_day3` | `1` | Gradient Boosting | `17.1039` | `10.1054` |

FastAPI loads the registered models from Hopsworks Model Registry instead of serving arbitrary local pickle files.

## 11. Explainability

SHAP explainability was generated for the selected model artifacts.

Top SHAP features:

| Horizon | Model | Most important features |
| --- | --- | --- |
| Day +1 | Ridge | `pm2_5`, `sulphur_dioxide_lag_1h`, `sulphur_dioxide`, `aqi_roll_mean_12h`, `aqi_roll_max_24h` |
| Day +2 | Gradient Boosting | `month_cos`, `pm2_5`, `day_of_year`, `pm2_5_roll_mean_12h`, `pm2_5_roll_mean_6h` |
| Day +3 | Gradient Boosting | `month_cos`, `day_of_year`, `temperature_2m_roll_mean_24h`, `pm2_5`, `wind_speed_10m_roll_mean_24h` |

Generated explainability artifacts:

- `reports/explainability/SHAP_SUMMARY.md`
- `reports/explainability/shap_importance_day1.csv`
- `reports/explainability/shap_importance_day1.png`
- `reports/explainability/shap_importance_day2.csv`
- `reports/explainability/shap_importance_day2.png`
- `reports/explainability/shap_importance_day3.csv`
- `reports/explainability/shap_importance_day3.png`

## 12. Automation

GitHub Actions workflows implement the required automation.

| Workflow | File | Schedule | Purpose |
| --- | --- | --- | --- |
| CI | `.github/workflows/ci.yml` | on code validation events | Runs local validation tests |
| Hourly Feature Pipeline | `.github/workflows/feature_pipeline.yml` | every hour | Fetches latest Open-Meteo observation, computes features, writes live features to Hopsworks |
| Daily Training and Registry Pipeline | `.github/workflows/training_pipeline.yml` | daily | Reads Hopsworks training data, trains/evaluates models, registers selected models |

Manual `workflow_dispatch` is enabled for the operational workflows so they can be triggered before final submission.

The hourly pipeline does not retrain models. The daily pipeline performs retraining and registry updates.

## 13. FastAPI Backend

The FastAPI backend exposes:

- `GET /health`
- `GET /predict`
- `GET /docs`

Public backend URL:

- `https://pearls-aqi-fastapi.onrender.com`

Verified `/health` response:

```json
{"status":"ok","service":"pearls-aqi-fastapi","required_horizons":["day1","day2","day3"]}
```

The `/predict` endpoint:

- reads the latest live features from Hopsworks Feature Store;
- loads the latest registered model for each horizon from Hopsworks Model Registry;
- returns current observed US AQI;
- returns Day +1, Day +2, and Day +3 AQI forecasts;
- returns AQI health-alert categories/messages;
- exposes model registry names, versions, model family, and RMSE metrics;
- states the no-future-input policy in the response.

## 14. Streamlit Dashboard

Public dashboard URL:

- `https://aqipredictor-x3knr3fztvobzbemejnf3l.streamlit.app/`

The dashboard calls FastAPI and displays:

- current observed Karachi US AQI;
- current pollutant and weather context;
- latest observation timestamp and freshness;
- Day +1, Day +2, and Day +3 forecast cards;
- AQI alert/health guidance;
- forecast chart and summary table;
- EDA tab;
- Model Comparison tab;
- Explainability tab;
- evaluator-focused Technical Details tab.

The dashboard does not directly load local model files or bypass FastAPI.

## 15. Hazard Alerts

The system maps current and predicted AQI into health-alert categories, including:

- Good
- Moderate
- Unhealthy for Sensitive Groups
- Unhealthy
- Very Unhealthy
- Hazardous

The dashboard shows user-friendly guidance based on the active AQI category.

## 16. Deployment

The project uses free/serverless-style deployment:

- FastAPI backend on Render Free Web Service
- Streamlit dashboard on Streamlit Community Cloud
- Hopsworks Feature Store and Model Registry
- GitHub Actions for scheduled automation

Important deployment compatibility fix:

- Python is pinned to `3.11.16` because Hopsworks/HSFS dependencies are not safe on Python 3.14.
- `hopsworks==5.0.6` and `hsfs==2.1.8` are pinned for deployment consistency.

No paid API, paid database, paid GPU, or paid cloud service is required.

## 17. Security And Secrets

Secrets are not committed to Git.

Secret values belong in:

- local `.env`;
- GitHub Actions Secrets;
- Render environment variables;
- Streamlit Community Cloud secrets.

Required secret/environment names:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`
- `HOPSWORKS_HOST`
- `FASTAPI_BASE_URL` for Streamlit

The repository intentionally excludes local `.env`, caches, certificates, and model artifacts from Git.

## 18. Requirement Coverage

| Requirement | Status |
| --- | --- |
| External API ingestion | Implemented with Open-Meteo |
| Weather and pollutant data | Implemented |
| Time features | Implemented |
| AQI change-rate feature | Implemented |
| Feature Store | Implemented with Hopsworks |
| Historical backfill | Implemented |
| Training reads from Feature Store | Implemented |
| Ridge Regression | Implemented |
| Random Forest | Implemented |
| Deep-learning model | Implemented with scikit-learn MLP and PyTorch MLP candidate |
| RMSE, MAE, R2 | Implemented |
| Model Registry | Implemented with Hopsworks |
| Hourly feature automation | Implemented with GitHub Actions |
| Daily training automation | Implemented with GitHub Actions |
| FastAPI backend | Implemented and deployed |
| Streamlit dashboard | Implemented and deployed |
| EDA | Implemented |
| SHAP explanations | Implemented |
| Hazard alerts | Implemented |
| Detailed report | Implemented in this file |

## 19. Known Limitations

- The project currently focuses on Karachi only, which satisfies the requirement.
- Open-Meteo air-quality values may be modeled/reanalysis data rather than direct station measurements.
- AQI forecasts are direct statistical/ML forecasts from recent and historical observed signals. The model intentionally does not use Open-Meteo future forecast values as inputs.
- Day +2 and Day +3 are harder forecasting problems than Day +1; their R2 scores are weaker, but selected models still beat persistence on validation RMSE.
- Free-tier Hopsworks, Render, Streamlit, and GitHub Actions availability can change or temporarily throttle usage.
- The local committed metrics file may need refreshing after future GitHub Actions training runs if the evaluator wants the newest PyTorch metric rows displayed from repository artifacts.

## 20. Final Submission Checklist

Before submitting to the SHINE portal:

- [ ] Confirm GitHub repository is public or accessible to evaluators.
- [ ] Confirm `.env` and secrets are not committed.
- [ ] Confirm CI workflow passes.
- [ ] Manually run the hourly feature workflow and confirm it passes.
- [ ] Manually run the daily training workflow and confirm it passes.
- [ ] Open `https://pearls-aqi-fastapi.onrender.com/health` and confirm status is OK.
- [ ] Open `https://pearls-aqi-fastapi.onrender.com/predict` and confirm predictions return.
- [ ] Open `https://aqipredictor-x3knr3fztvobzbemejnf3l.streamlit.app/` and confirm the dashboard loads.
- [ ] Confirm the dashboard shows current AQI, 3-day forecast, alerts, EDA, model comparison, and explainability tabs.
- [ ] Submit the GitHub repository URL through the SHINE portal.

