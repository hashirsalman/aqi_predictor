"""Small scikit-learn compatible PyTorch regressor for AQI experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from aqi_predictor.constants import RANDOM_SEED


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """CPU-friendly PyTorch MLP with a minimal sklearn-style interface.

    The class imports PyTorch lazily inside ``fit``/``predict`` so normal web
    inference and CI tests can import the project without requiring the large
    training-only PyTorch dependency.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        epochs: int = 80,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 256,
        random_state: int = RANDOM_SEED,
    ) -> None:
        """Initialize hyperparameters in the format expected by scikit-learn."""

        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, x: Any, y: Any) -> "TorchMLPRegressor":
        """Fit the neural network on tabular AQI features."""

        torch = _import_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_array = _to_numpy(x).astype(np.float32)
        y_array = _to_numpy(y).reshape(-1, 1).astype(np.float32)
        self.n_features_in_ = x_array.shape[1]

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_array),
            torch.from_numpy(y_array),
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(self.n_features_in_, self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units // 2, 1),
        )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Predict AQI values for a feature frame/array."""

        torch = _import_torch()
        if not hasattr(self, "model_"):
            raise ValueError("TorchMLPRegressor must be fitted before predict().")

        x_array = _to_numpy(x).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(torch.from_numpy(x_array)).numpy().reshape(-1)
        return predictions


def _to_numpy(values: Any) -> np.ndarray:
    """Convert pandas/numpy/list-like data to a NumPy array."""

    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _import_torch() -> Any:
    """Import PyTorch with a clear install message when unavailable."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package state.
        raise ImportError(
            "PyTorch is required for the pytorch_mlp training candidate. "
            "Install it with: python -m pip install -r requirements-training.txt"
        ) from exc
    return torch
