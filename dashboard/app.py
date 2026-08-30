"""Streamlit dashboard for Karachi AQI forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import fetch_dashboard_payload, get_fastapi_base_url
from formatting import (
    condition_table,
    forecast_summary,
    model_metric_table,
    observation_age_hours,
    prediction_table,
    strongest_alert,
)


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render the dashboard."""

    st.title("Pearls AQI Predictor - Karachi")
    st.caption("Real-time air quality monitoring and 3-day US AQI forecast for Karachi.")

    with st.sidebar:
        st.header("Karachi AQI")
        st.write("3-day forecast - live observations - hazard alerts")
        refresh = st.button("Refresh predictions", type="primary")
        with st.expander("Technical settings"):
            base_url = st.text_input("FastAPI base URL", value=get_fastapi_base_url())
            force_refresh = st.checkbox(
                "Force model refresh",
                value=False,
                help="Reload models from the production model registry on this request.",
            )

    if refresh:
        st.cache_data.clear()

    try:
        payload = _cached_payload(base_url, force_refresh)
    except Exception as exc:  # noqa: BLE001 - Streamlit should show friendly errors.
        st.error("The dashboard could not reach the prediction backend.")
        st.info("If running locally, start the backend with `python scripts/run_api.py`, then refresh this page.")
        with st.expander("Technical error details"):
            st.code(str(exc))
        return

    forecast_tab, eda_tab, model_tab, explain_tab, technical_tab = st.tabs(
        ["Dashboard", "EDA", "Model Comparison", "Explainability", "Technical Details"]
    )

    with forecast_tab:
        _render_forecast_dashboard(payload)

    with eda_tab:
        _render_eda_tab()

    with model_tab:
        _render_model_comparison_tab()

    with explain_tab:
        _render_explainability_tab()

    with technical_tab:
        _render_technical_tab(payload, base_url)


def _render_forecast_dashboard(payload: dict) -> None:
    """Render the primary user-facing AQI forecast dashboard."""

    source = payload["source_observation"]
    current_alert = source["alert"]
    age_hours = observation_age_hours(payload)

    st.subheader("Current Air Quality")
    current_cols = st.columns(4)
    current_cols[0].metric("Live US AQI", round(source["current_aqi"]))
    current_cols[1].metric("Category", current_alert["category"])
    current_cols[2].metric("Location", "Karachi")
    current_cols[3].metric("Updated", f"{age_hours}h ago" if age_hours is not None else "unknown")
    st.caption(f"Latest observation: {source['event_time_local']} PKT")

    top_alert = strongest_alert(payload)
    if top_alert and top_alert.get("is_health_alert"):
        st.warning(f"{top_alert['category']}: {top_alert['message']}")
    elif top_alert:
        st.info(f"{top_alert['category']}: {top_alert['message']}")

    pollutants = condition_table(payload, "pollutants")
    if not pollutants.empty:
        st.subheader("Current Pollutants")
        pollutant_cols = st.columns(min(len(pollutants), 6))
        for column, (_, row) in zip(pollutant_cols, pollutants.iterrows(), strict=False):
            column.metric(str(row["Metric"]), f"{row['Value']:g}", str(row["Unit"]))

    st.subheader("3-Day Daily Summary Forecast")
    table = prediction_table(payload)
    forecast_cols = st.columns(len(table))
    for column, (_, row) in zip(forecast_cols, table.iterrows(), strict=True):
        column.metric(
            label=f"{row['Horizon'].upper()} forecast",
            value=int(row["Predicted AQI"]),
            help=f"{row['Category']} - expected daily-average US AQI",
        )

    chart = px.bar(
        table,
        x="Horizon",
        y="Predicted AQI",
        color="Category",
        text="Predicted AQI",
        title="Forecasted daily-average US AQI",
    )
    chart.update_layout(yaxis_title="US AQI", xaxis_title="Forecast horizon")
    st.plotly_chart(chart, use_container_width=True)

    user_table = table[["Horizon", "Target", "Predicted AQI", "Category", "Alert Level"]]
    st.dataframe(user_table, use_container_width=True, hide_index=True)

    summary = forecast_summary(payload)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Forecast average", summary["average"] if summary["average"] is not None else "n/a")
    summary_cols[1].metric("Forecast maximum", summary["maximum"] if summary["maximum"] is not None else "n/a")
    summary_cols[2].metric("Forecast minimum", summary["minimum"] if summary["minimum"] is not None else "n/a")

    weather = condition_table(payload, "weather")
    if not weather.empty:
        st.subheader("Weather Context")
        st.dataframe(weather, use_container_width=True, hide_index=True)

    st.subheader("Health Guidance")
    st.write(top_alert["message"] if top_alert else "No AQI alert is available for the current forecast.")


def _render_eda_tab() -> None:
    """Render EDA highlights and generated trend figures."""

    st.subheader("Exploratory Data Analysis")
    st.caption("Historical Karachi US AQI trends from the reproducible Open-Meteo backfill.")
    summary_path = REPO_ROOT / "reports" / "EDA_SUMMARY.md"
    if summary_path.exists():
        with st.expander("EDA written summary", expanded=False):
            st.markdown(summary_path.read_text(encoding="utf-8"))

    figures = [
        ("AQI distribution", REPO_ROOT / "reports" / "figures" / "eda_aqi_distribution.png"),
        ("AQI trend and rolling statistics", REPO_ROOT / "reports" / "figures" / "eda_aqi_timeseries_rolling.png"),
        ("Hourly, weekly, and monthly seasonality", REPO_ROOT / "reports" / "figures" / "eda_aqi_seasonality.png"),
        ("Feature correlation heatmap", REPO_ROOT / "reports" / "figures" / "eda_correlation_heatmap.png"),
        ("Pollutant relationships", REPO_ROOT / "reports" / "figures" / "eda_pollutant_relationships.png"),
    ]
    for title, path in figures:
        if path.exists():
            st.markdown(f"#### {title}")
            st.image(str(path), use_container_width=True)


def _render_model_comparison_tab() -> None:
    """Render model comparison metrics from the training pipeline."""

    st.subheader("Model Comparison")
    st.caption("Chronological validation/test metrics. Lower RMSE and MAE are better; higher R² is better.")
    metrics_path = REPO_ROOT / "reports" / "metrics" / "model_metrics.csv"
    if not metrics_path.exists():
        st.info("Model metrics are not available yet. Run `python scripts/train.py`.")
        return

    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    validation = metrics[metrics["split"] == "validation"].copy()
    if not validation.empty:
        chart = px.bar(
            validation,
            x="model",
            y="rmse",
            color="horizon",
            barmode="group",
            title="Validation RMSE by model and horizon",
        )
        chart.update_layout(xaxis_title="Model", yaxis_title="Validation RMSE")
        st.plotly_chart(chart, use_container_width=True)

    summary_path = REPO_ROOT / "reports" / "MODEL_EXPERIMENT_SUMMARY.md"
    if summary_path.exists():
        with st.expander("Training report", expanded=False):
            st.markdown(summary_path.read_text(encoding="utf-8"))


def _render_explainability_tab() -> None:
    """Render SHAP feature importance artifacts."""

    st.subheader("Explainability")
    st.caption("SHAP feature-importance summaries for the selected production models.")
    summary_path = REPO_ROOT / "reports" / "explainability" / "SHAP_SUMMARY.md"
    if summary_path.exists():
        with st.expander("SHAP written summary", expanded=False):
            st.markdown(summary_path.read_text(encoding="utf-8"))

    for horizon in ("day1", "day2", "day3"):
        path = REPO_ROOT / "reports" / "explainability" / f"shap_importance_{horizon}.png"
        if path.exists():
            st.markdown(f"#### {horizon.upper()} feature importance")
            st.image(str(path), use_container_width=True)


def _render_technical_tab(payload: dict, base_url: str) -> None:
    """Render evaluator-facing architecture and raw API details."""

    st.subheader("Technical Details for Evaluators")
    st.caption(
        "Production path: Open-Meteo observations -> Hopsworks Feature Store -> "
        "Hopsworks Model Registry -> FastAPI -> Streamlit."
    )
    generated_time = str(payload.get("generated_at_utc", "unknown")).replace("T", " ").split(".")[0]
    st.write(f"Backend API: `{base_url}`")
    st.write(f"API generated response at UTC: `{generated_time}`")
    st.write(f"Input policy: {payload.get('input_policy')}")
    st.write("Forecast model details")
    st.dataframe(model_metric_table(payload), use_container_width=True, hide_index=True)
    with st.expander("Raw API response"):
        st.json(payload)


@st.cache_data(ttl=300, show_spinner="Fetching latest AQI predictions...")
def _cached_payload(base_url: str, force_refresh: bool) -> dict:
    return fetch_dashboard_payload(base_url=base_url, force_model_refresh=force_refresh)


if __name__ == "__main__":
    main()
