"""Streamlit dashboard for Karachi AQI forecasts."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from api_client import fetch_dashboard_payload, get_fastapi_base_url
from formatting import model_metric_table, observation_age_hours, prediction_table, strongest_alert


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)


def main() -> None:
    """Render the dashboard."""

    st.title("Pearls AQI Predictor — Karachi")
    st.caption("US AQI forecasts using Open-Meteo, Hopsworks Feature Store, Hopsworks Model Registry, FastAPI, and Streamlit.")

    with st.sidebar:
        st.header("Backend")
        base_url = st.text_input("FastAPI base URL", value=get_fastapi_base_url())
        force_refresh = st.checkbox(
            "Force model refresh",
            value=False,
            help="Reload models from Hopsworks Model Registry on this request.",
        )
        refresh = st.button("Refresh predictions", type="primary")

    if refresh:
        st.cache_data.clear()

    try:
        payload = _cached_payload(base_url, force_refresh)
    except Exception as exc:  # noqa: BLE001 - Streamlit should show friendly errors.
        st.error("The dashboard could not reach the FastAPI prediction backend.")
        st.info("Start the backend with: `python scripts/run_api.py`, then refresh this page.")
        st.code(str(exc))
        return

    source = payload["source_observation"]
    current_alert = source["alert"]
    age_hours = observation_age_hours(payload)

    st.subheader("Current observed AQI")
    current_cols = st.columns(4)
    current_cols[0].metric("Current US AQI", round(source["current_aqi"]))
    current_cols[1].metric("Current category", current_alert["category"])
    current_cols[2].metric("Latest observation UTC", source["event_time_utc"])
    current_cols[3].metric("Observation age", f"{age_hours}h" if age_hours is not None else "unknown")

    top_alert = strongest_alert(payload)
    if top_alert and top_alert.get("is_health_alert"):
        st.warning(f"{top_alert['category']}: {top_alert['message']}")
    elif top_alert:
        st.info(f"{top_alert['category']}: {top_alert['message']}")

    st.subheader("3-day AQI forecast")
    table = prediction_table(payload)
    forecast_cols = st.columns(len(table))
    for column, (_, row) in zip(forecast_cols, table.iterrows(), strict=True):
        column.metric(
            label=f"{row['Horizon'].upper()} forecast",
            value=int(row["Predicted AQI"]),
            help=f"{row['Category']} | model v{row['Registry Version']} | validation RMSE {row['Validation RMSE']:.2f}",
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

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Model registry freshness and metrics")
    st.caption("These metrics come from the currently loaded Hopsworks Model Registry versions, not hard-coded local artifacts.")
    st.dataframe(model_metric_table(payload), use_container_width=True, hide_index=True)

    with st.expander("Raw API response"):
        st.json(payload)


@st.cache_data(ttl=300, show_spinner="Fetching latest AQI predictions from FastAPI...")
def _cached_payload(base_url: str, force_refresh: bool) -> dict:
    return fetch_dashboard_payload(base_url=base_url, force_model_refresh=force_refresh)


if __name__ == "__main__":
    main()

