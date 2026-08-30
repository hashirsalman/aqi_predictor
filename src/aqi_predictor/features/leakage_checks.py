"""Automated leakage checks for feature and target frames."""

from __future__ import annotations

import pandas as pd

from aqi_predictor.features.feature_contract import CANONICAL_FEATURE_COLUMNS, TARGET_COLUMNS


FORBIDDEN_FEATURE_SUBSTRINGS = ("target_", "future_", "forecast_")


def assert_no_target_columns_in_features(feature_columns: tuple[str, ...] | list[str]) -> None:
    """Raise if target or future-looking columns are present in model inputs."""

    forbidden = [
        column
        for column in feature_columns
        if column in TARGET_COLUMNS
        or any(fragment in column for fragment in FORBIDDEN_FEATURE_SUBSTRINGS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden leakage-prone feature columns found: {forbidden}")


def assert_canonical_features_are_safe() -> None:
    """Validate the central feature contract against obvious leakage names."""

    assert_no_target_columns_in_features(CANONICAL_FEATURE_COLUMNS)


def assert_chronological_split(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Ensure train/validation/test frames are ordered without time overlap."""

    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        if not frame["event_time_utc"].is_monotonic_increasing:
            raise ValueError(f"{name} split is not chronological")

    if train["event_time_utc"].max() >= validation["event_time_utc"].min():
        raise ValueError("Train split overlaps validation split")
    if validation["event_time_utc"].max() >= test["event_time_utc"].min():
        raise ValueError("Validation split overlaps test split")
