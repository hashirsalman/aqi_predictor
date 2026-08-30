"""Candidate model factory for the required model families."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aqi_predictor.constants import RANDOM_SEED


def build_candidate_models() -> dict[str, object]:
    """Build deterministic AQI regression candidates.

    The neural candidate is intentionally a small scikit-learn MLP so it can run on
    CPU/free-tier environments. A TensorFlow/PyTorch model can be added later as a
    heavier optional experiment, but this keeps Phase 7 reproducible locally.
    """

    return {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            learning_rate=0.06,
            n_estimators=180,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_SEED,
        ),
        "neural_mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.001,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=300,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }
