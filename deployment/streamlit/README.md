# Streamlit Deployment Notes

Recommended free-path candidate: Streamlit Community Cloud.

Why this is prepared:

- Streamlit Community Cloud is designed for deploying Streamlit apps from GitHub.
- The dashboard entrypoint is:

```text
dashboard/app.py
```

Important limitation:

- The dashboard needs a live FastAPI backend URL.
- Deploy FastAPI first, then set `FASTAPI_BASE_URL` in Streamlit secrets.
- Do not make Streamlit load local models or local CSV files as a production fallback.

Streamlit app settings:

```text
Repository: hashirsalman/aqi_predictor
Branch: master
Main file path: dashboard/app.py
Python version: 3.11
```

Streamlit secret to configure:

```toml
FASTAPI_BASE_URL = "https://<your-fastapi-service-url>"
```

After deployment, verify the dashboard shows:

- current observed US AQI;
- latest source observation timestamp;
- current pollutant context;
- weather context;
- Day +1, Day +2, and Day +3 forecast AQI;
- AQI categories and alerts;
- health guidance for the strongest current/forecast AQI alert;
- EDA tab with historical trend/seasonality/correlation figures;
- Model Comparison tab with RMSE/MAE/R² metrics;
- Explainability tab with SHAP feature-importance figures;
- Hopsworks model registry names/versions and validation/test RMSE metrics inside the technical evaluator section.

The main page should be user-facing. Keep backend URL, raw API response, and registry internals inside the technical details tab rather than showing them as primary dashboard content.
