"""Immutable values exchanged between benchmark stages."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class BenchmarkData:
    """Validated panel plus metadata required by multivariate models."""

    frame: pd.DataFrame
    series_count: int
    time_points: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Validated predictions and deterministic evaluation metrics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    """Files emitted by one benchmark run."""

    predictions: Path
    metrics: Path
    manifest: Path
    data_plot: Path | None
    forecast_plot: Path | None
