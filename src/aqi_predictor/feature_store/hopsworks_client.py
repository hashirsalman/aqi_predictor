"""Hopsworks connection helpers with safe secret loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from aqi_predictor.config import PROJECT_ROOT


@dataclass(frozen=True)
class HopsworksSettings:
    """Required Hopsworks connection settings."""

    project: str
    api_key: str
    host: str | None = None
    cert_folder: Path = PROJECT_ROOT / ".hopsworks-certs"


def load_hopsworks_settings(env_path: Path = PROJECT_ROOT / ".env") -> HopsworksSettings:
    """Load Hopsworks settings from `.env` or process environment without printing secrets."""

    if env_path.exists():
        load_dotenv(env_path)

    project = os.getenv("HOPSWORKS_PROJECT", "").strip()
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    host = os.getenv("HOPSWORKS_HOST", "").strip() or None
    cert_folder = Path(
        os.getenv("HOPSWORKS_CERT_FOLDER", "").strip() or PROJECT_ROOT / ".hopsworks-certs"
    ).expanduser()

    missing = [
        name
        for name, value in {
            "HOPSWORKS_PROJECT": project,
            "HOPSWORKS_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required Hopsworks environment variables: {missing}")

    return HopsworksSettings(
        project=project,
        api_key=api_key,
        host=host,
        cert_folder=cert_folder,
    )


def connect_hopsworks(settings: HopsworksSettings | None = None) -> Any:
    """Connect to Hopsworks and return the project object."""

    resolved = settings or load_hopsworks_settings()
    try:
        import hopsworks
    except ImportError as exc:
        raise RuntimeError(
            "The `hopsworks` package is not installed in this Python environment. "
            "Use Python 3.11 and install requirements before running cloud integration."
        ) from exc

    _patch_hopsworks_windows_kafka_pem_paths(resolved.cert_folder)

    kwargs: dict[str, Any] = {
        "project": resolved.project,
        "api_key_value": resolved.api_key,
        "cert_folder": str(resolved.cert_folder.resolve()),
    }
    if resolved.host:
        kwargs["host"] = resolved.host
    return hopsworks.login(**kwargs)


def get_feature_store(project: Any | None = None) -> Any:
    """Return the Hopsworks Feature Store handle."""

    resolved_project = project or connect_hopsworks()
    return resolved_project.get_feature_store()


def get_model_registry(project: Any | None = None) -> Any:
    """Return the Hopsworks Model Registry handle."""

    resolved_project = project or connect_hopsworks()
    return resolved_project.get_model_registry()


def _patch_hopsworks_windows_kafka_pem_paths(cert_folder: Path) -> None:
    """Redirect Hopsworks Kafka PEM files away from Unix-style `/tmp` on Windows.

    Hopsworks' Python ingestion path currently writes Kafka PEM files with
    `os.path.join("/tmp", ...)`. On Windows this resolves to `\\tmp\\...`,
    which usually does not exist. We keep the patch local to this process and
    store the generated PEM files under the gitignored Hopsworks certificate
    folder for this project.
    """

    if os.name != "nt":
        return

    from hopsworks_common.client import base as hopsworks_base_client

    if getattr(hopsworks_base_client.Client, "_aqi_windows_tmp_patch", False):
        return

    kafka_pem_folder = (cert_folder / "kafka-pem").resolve()

    def _write_pem_windows_safe(
        self: Any,
        keystore_path: str,
        keystore_pw: str,
        truststore_path: str,
        truststore_pw: str,
        prefix: str,
    ) -> tuple[str, str, str]:
        kafka_pem_folder.mkdir(parents=True, exist_ok=True)
        ks = hopsworks_base_client.jks.KeyStore.load(Path(keystore_path), keystore_pw, try_decrypt_keys=True)
        ts = hopsworks_base_client.jks.KeyStore.load(Path(truststore_path), truststore_pw, try_decrypt_keys=True)

        ca_chain_path = str(kafka_pem_folder / f"{prefix}_ca_chain.pem")
        self._write_ca_chain(ks, ts, ca_chain_path)

        client_cert_path = str(kafka_pem_folder / f"{prefix}_client_cert.pem")
        self._write_client_cert(ks, client_cert_path)

        client_key_path = str(kafka_pem_folder / f"{prefix}_client_key.pem")
        self._write_client_key(ks, client_key_path)

        return ca_chain_path, client_cert_path, client_key_path

    hopsworks_base_client.Client._write_pem = _write_pem_windows_safe
    hopsworks_base_client.Client._aqi_windows_tmp_patch = True
