"""Streamlit dashboard for Karachi AQI forecasts."""

from __future__ import annotations

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
    page_icon="AQI",
    layout="wide",
)


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

    source = payload["source_observation"]
    current_alert = source["alert"]
    age_hours = observation_age_hours(payload)
    generated_time = str(payload.get("generated_at_utc", "unknown")).replace("T", " ").split(".")[0]

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

    with st.expander("Technical details for evaluators"):
        st.caption(
            "Production path: Open-Meteo observations -> Hopsworks Feature Store -> "
            "Hopsworks Model Registry -> FastAPI -> Streamlit."
        )
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
