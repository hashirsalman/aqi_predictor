"""Model-selection helpers."""

from __future__ import annotations

import pandas as pd


def select_best_models(metrics: pd.DataFrame, split: str = "validation") -> pd.DataFrame:
    """Select the lowest-RMSE candidate per horizon for a given split."""

    subset = metrics[metrics["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No metrics available for split={split!r}.")
    subset = subset.sort_values(["horizon", "rmse", "mae"], ascending=[True, True, True])
    return subset.groupby("horizon", as_index=False).first()
