"""End-to-end benchmark orchestration."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from tsmixer_itransformer.artifacts import save_artifacts
from tsmixer_itransformer.backend import NeuralForecastBackend
from tsmixer_itransformer.config import AppConfig, DataConfig, ModelConfig, SplitConfig
from tsmixer_itransformer.data import DatasetReader, load_benchmark, validate_history
from tsmixer_itransformer.evaluation import evaluate_predictions
from tsmixer_itransformer.models import BenchmarkArtifacts
from tsmixer_itransformer.models import BenchmarkResult as Result
from tsmixer_itransformer.plotting import save_data_plot, save_forecast_plot


class ForecastBackend(Protocol):
    def predict(
        self,
        frame: pd.DataFrame,
        data: DataConfig,
        split: SplitConfig,
        model: ModelConfig,
        series_count: int,
    ) -> pd.DataFrame: ...


def run_benchmark(
    config: AppConfig,
    *,
    backend: ForecastBackend | None = None,
    reader: DatasetReader | None = None,
    plot: bool = False,
) -> BenchmarkArtifacts:
    """Load, validate, forecast, evaluate, plot, and persist one run."""
    data = load_benchmark(config.data, reader=reader)
    validate_history(data, config.split, config.model)
    selected_backend = backend or NeuralForecastBackend()
    predictions = selected_backend.predict(
        data.frame,
        config.data,
        config.split,
        config.model,
        data.series_count,
    )
    result = Result(predictions=predictions, metrics=evaluate_predictions(predictions))
    data_plot = None
    forecast_plot = None
    if plot:
        data_plot = save_data_plot(
            data.frame, config.output.directory / config.output.data_plot_file
        )
        forecast_plot = save_forecast_plot(
            predictions, config.output.directory / config.output.forecast_plot_file
        )
    return save_artifacts(
        config,
        data,
        result,
        data_plot=data_plot,
        forecast_plot=forecast_plot,
    )
