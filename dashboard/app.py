"""Streamlit dashboard for Karachi AQI forecasts."""

from __future__ import annotations

import json
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

    _inject_theme()
    st.markdown(
        """
        <section class="hero-card">
          <div>
            <p class="eyebrow">Karachi Air Quality</p>
            <h1>Pearls AQI Predictor</h1>
            <p class="hero-copy">Real-time air-quality context and a 3-day US AQI forecast for Karachi.</p>
          </div>
          <div class="hero-badge">Live ML Forecast</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Karachi AQI")
        st.write("3-day forecast · live observations · hazard alerts")
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
        color_discrete_map={
            "Good": "#22c55e",
            "Moderate": "#eab308",
            "Unhealthy for Sensitive Groups": "#f97316",
            "Unhealthy": "#ef4444",
            "Very Unhealthy": "#8b5cf6",
            "Hazardous": "#7f1d1d",
        },
    )
    chart.update_layout(
        yaxis_title="US AQI",
        xaxis_title="Forecast horizon",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
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
    summary = _load_json(REPO_ROOT / "reports" / "metrics" / "eda_summary.json")
    if summary:
        distribution = summary.get("aqi_distribution", {})
        high_aqi = summary.get("high_aqi_events", {})
        unhealthy = high_aqi.get("unhealthy_or_worse", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Historical hours", f"{int(summary.get('rows', 0)):,}" if summary.get("rows") else "n/a")
        col2.metric("Median US AQI", _format_optional_number(distribution.get("median")))
        col3.metric("Highest AQI", _format_optional_number(distribution.get("max")))
        col4.metric("Unhealthy hours", f"{int(unhealthy.get('hour_count', 0)):,}" if unhealthy else "n/a")

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
        st.info("Model metrics are not available yet.")
        return

    metrics = pd.read_csv(metrics_path)
    display_metrics = _clean_metric_frame(metrics)

    selected = display_metrics[
        (display_metrics["Split"] == "Validation") & (display_metrics["Beat Persistence"] == "Yes")
    ].sort_values(["Horizon", "RMSE"])
    if not selected.empty:
        st.markdown("#### Current champion models")
        champion_rows = selected.groupby("Horizon", as_index=False).first()
        champion_cols = st.columns(len(champion_rows))
        for column, (_, row) in zip(champion_cols, champion_rows.iterrows(), strict=False):
            column.metric(
                row["Horizon"],
                row["Model"],
                delta=f"RMSE {row['RMSE']:.2f}",
                help="Selected from chronological validation performance.",
            )

    st.markdown("#### Validation results")
    validation_display = display_metrics[display_metrics["Split"] == "Validation"]
    st.dataframe(validation_display, use_container_width=True, hide_index=True)

    validation = metrics[metrics["split"] == "validation"].copy()
    if not validation.empty:
        validation["Model"] = validation["model"].map(_display_model_name)
        validation["Horizon"] = validation["horizon"].map(_display_horizon)
        chart = px.bar(
            validation,
            x="Model",
            y="rmse",
            color="Horizon",
            barmode="group",
            title="Validation RMSE by model and horizon",
            color_discrete_sequence=["#0ea5e9", "#22c55e", "#f97316"],
        )
        chart.update_layout(
            xaxis_title="Model",
            yaxis_title="Validation RMSE",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(chart, use_container_width=True)

    with st.expander("What these metrics mean", expanded=False):
        st.write(
            "RMSE and MAE measure forecast error in AQI points, so lower is better. "
            "R² summarizes how much variation the model explains on a chronological holdout. "
            "A model is considered stronger when it improves over the persistence baseline."
        )


def _render_explainability_tab() -> None:
    """Render SHAP feature importance artifacts."""

    st.subheader("Explainability")
    st.caption("Feature-importance summaries for the selected AQI forecasting models.")

    for horizon in ("day1", "day2", "day3"):
        st.markdown(f"#### {_display_horizon(horizon)} feature importance")
        importance_path = REPO_ROOT / "reports" / "explainability" / f"shap_importance_{horizon}.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(8)
            importance = importance.rename(
                columns={
                    "rank": "Rank",
                    "feature": "Feature",
                    "mean_abs_shap": "Importance",
                }
            )
            if "Importance" in importance:
                importance["Importance"] = importance["Importance"].round(3)
            st.dataframe(importance, use_container_width=True, hide_index=True)
        path = REPO_ROOT / "reports" / "explainability" / f"shap_importance_{horizon}.png"
        if path.exists():
            st.image(str(path), use_container_width=True)

    with st.expander("How to read this", expanded=False):
        st.write(
            "Higher importance means the feature had a larger average influence on model output "
            "for the explained sample. These explanations help show which recent pollutant, weather, "
            "and seasonal signals the model relied on most."
        )


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


def _inject_theme() -> None:
    """Apply lightweight dashboard styling."""

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 30rem),
                linear-gradient(180deg, #f8fbff 0%, #eef7f4 45%, #f8fafc 100%);
            color: #0f172a;
        }
        .hero-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1.5rem;
            padding: 1.5rem 1.75rem;
            border-radius: 1.5rem;
            margin-bottom: 1.25rem;
            background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 52%, #22c55e 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 118, 110, 0.25);
        }
        .hero-card h1 {
            margin: 0;
            font-size: 2.6rem;
            letter-spacing: -0.04em;
        }
        .hero-copy {
            margin: 0.35rem 0 0;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.05rem;
        }
        .eyebrow {
            margin: 0 0 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.78);
        }
        .hero-badge {
            white-space: nowrap;
            padding: 0.65rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.35);
            font-weight: 700;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 1rem;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ecfeff 0%, #f0fdf4 100%);
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clean_metric_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return user-friendly model metrics for dashboard display."""

    display = metrics.copy()
    display = display.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon",
            "split": "Split",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R²",
            "beats_persistence_validation": "Beat Persistence",
        }
    )
    display["Model"] = display["Model"].map(_display_model_name)
    display["Horizon"] = display["Horizon"].map(_display_horizon)
    display["Split"] = display["Split"].map(lambda value: str(value).title())
    display["Beat Persistence"] = display["Beat Persistence"].map(lambda value: "Yes" if _as_bool(value) else "No")
    for column in ("RMSE", "MAE", "R²"):
        display[column] = display[column].round(3)
    return display[["Model", "Horizon", "Split", "RMSE", "MAE", "R²", "Beat Persistence"]]


def _display_model_name(value: str) -> str:
    """Convert internal model IDs into dashboard labels."""

    names = {
        "persistence": "Persistence",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "neural_mlp": "Neural MLP",
        "pytorch_mlp": "PyTorch MLP",
    }
    return names.get(str(value), str(value).replace("_", " ").title())


def _display_horizon(value: str) -> str:
    """Convert horizon IDs into dashboard labels."""

    names = {"day1": "Day 1", "day2": "Day 2", "day3": "Day 3"}
    return names.get(str(value), str(value).replace("_", " ").title())


def _load_json(path: Path) -> dict:
    """Load a JSON report if present."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_optional_number(value: object) -> str:
    """Format a metric value for dashboard cards."""

    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def _as_bool(value: object) -> bool:
    """Coerce CSV/string boolean values for display only."""

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


if __name__ == "__main__":
    main()
