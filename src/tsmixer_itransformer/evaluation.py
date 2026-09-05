"""Deterministic point-forecast evaluation."""

import numpy as np
import pandas as pd

from tsmixer_itransformer.errors import BackendError

MODEL_COLUMNS = ("TSMixer", "iTransformer")


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregate MAE and MSE for each supported model."""
    required = {"y", *MODEL_COLUMNS}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise BackendError("evaluation requires actuals and both model predictions")
    actual = predictions["y"].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    for name in MODEL_COLUMNS:
        forecast = predictions[name].to_numpy(dtype=np.float64)
        error = actual - forecast
        if not np.isfinite(error).all():
            raise BackendError("evaluation contains non-finite values")
        rows.append(
            {
                "model": name,
                "mae": float(np.mean(np.abs(error))),
                "mse": float(np.mean(np.square(error))),
            }
        )
    return pd.DataFrame(rows, columns=["model", "mae", "mse"])
