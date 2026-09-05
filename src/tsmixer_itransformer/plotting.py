"""Optional non-interactive benchmark visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tsmixer_itransformer.errors import ArtifactError, DependencyUnavailableError


def save_data_plot(frame: pd.DataFrame, path: Path) -> Path:
    """Save one panel containing every source series."""
    plt = _pyplot()
    figure: Any = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111)
    for item, group in frame.groupby("unique_id", sort=False):
        axis.plot(group["ds"], group["y"], label=str(item))
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Value")
    axis.set_title("Benchmark data")
    axis.grid(True)
    axis.legend()
    return _save(plt, figure, path)


def save_forecast_plot(predictions: pd.DataFrame, path: Path) -> Path:
    """Save actuals and both model forecasts for every series."""
    plt = _pyplot()
    figure: Any = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111)
    for item, group in predictions.groupby("unique_id", sort=False):
        axis.plot(group["ds"], group["y"], label=f"actual {item}")
        axis.plot(group["ds"], group["TSMixer"], linestyle="--", label=f"TSMixer {item}")
        axis.plot(
            group["ds"],
            group["iTransformer"],
            linestyle=":",
            label=f"iTransformer {item}",
        )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Value")
    axis.set_title("Actuals and forecasts")
    axis.grid(True)
    axis.legend()
    return _save(plt, figure, path)


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyUnavailableError(
            "plotting requires: pip install 'tsmixer-itransformer-forecasting[plots]'"
        ) from exc
    return plt


def _save(plt: Any, figure: Any, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
    except OSError as exc:
        raise ArtifactError(f"cannot save plot {path}: {exc}") from exc
    finally:
        plt.close(figure)
    return path
