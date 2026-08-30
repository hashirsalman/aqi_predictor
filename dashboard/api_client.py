"""FastAPI client helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

DEFAULT_FASTAPI_BASE_URL = "http://127.0.0.1:8000"


def get_fastapi_base_url() -> str:
    """Return the configured FastAPI base URL for the dashboard."""

    load_dotenv()
    return normalize_base_url(
        os.getenv("FASTAPI_BASE_URL")
        or _streamlit_secret("FASTAPI_BASE_URL")
        or DEFAULT_FASTAPI_BASE_URL
    )


def normalize_base_url(base_url: str) -> str:
    """Normalize a user-provided FastAPI base URL."""

    cleaned = base_url.strip()
    if not cleaned:
        return DEFAULT_FASTAPI_BASE_URL
    return cleaned.rstrip("/")


def _streamlit_secret(name: str) -> str | None:
    """Read a Streamlit Community Cloud secret when Streamlit is available."""

    try:
        import streamlit as st
    except Exception:  # noqa: BLE001 - Streamlit may be unavailable in non-dashboard contexts.
        return None

    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - st.secrets can raise when no secrets file exists locally.
        return None
    return str(value).strip() if value else None


def fetch_dashboard_payload(
    base_url: str | None = None,
    force_model_refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Fetch prediction payload from FastAPI."""

    resolved_base_url = normalize_base_url(base_url or get_fastapi_base_url())
    response = requests.get(
        f"{resolved_base_url}/predict",
        params={"force_model_refresh": force_model_refresh},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
