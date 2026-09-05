from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsmixer_itransformer.config import AppConfig
from tsmixer_itransformer.service import run_benchmark


class FakeBackend:
    def __init__(self, predictions: pd.DataFrame) -> None:
        self.predictions = predictions
        self.call: tuple[object, ...] | None = None

    def predict(
        self,
        frame: pd.DataFrame,
        data: object,
        split: object,
        model: object,
        series_count: int,
    ) -> pd.DataFrame:
        self.call = (frame, data, split, model, series_count)
        return self.predictions


def test_run_benchmark_end_to_end(
    app_config: AppConfig, panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    backend = FakeBackend(predictions)
    result = run_benchmark(app_config, backend=backend, reader=lambda config: panel)
    assert backend.call is not None
    assert backend.call[-1] == 2
    assert result.predictions.exists()
    metrics = pd.read_csv(result.metrics)
    assert metrics["model"].tolist() == ["TSMixer", "iTransformer"]
    assert result.data_plot is None


def test_run_benchmark_saves_plots(
    app_config: AppConfig,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Path]] = []

    def data_plot(frame: pd.DataFrame, path: Path) -> Path:
        calls.append(("data", path))
        return path

    def forecast_plot(frame: pd.DataFrame, path: Path) -> Path:
        calls.append(("forecast", path))
        return path

    monkeypatch.setattr("tsmixer_itransformer.service.save_data_plot", data_plot)
    monkeypatch.setattr("tsmixer_itransformer.service.save_forecast_plot", forecast_plot)
    result = run_benchmark(
        app_config,
        backend=FakeBackend(predictions),
        reader=lambda config: panel,
        plot=True,
    )
    assert [name for name, _ in calls] == ["data", "forecast"]
    assert result.data_plot == calls[0][1]
    assert result.forecast_plot == calls[1][1]


def test_run_benchmark_uses_default_backend(
    app_config: AppConfig,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(predictions)
    monkeypatch.setattr(
        "tsmixer_itransformer.service.NeuralForecastBackend", lambda: backend
    )
    result = run_benchmark(app_config, reader=lambda config: panel)
    assert result.metrics.exists()
