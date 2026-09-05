from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tsmixer_itransformer.artifacts import save_artifacts
from tsmixer_itransformer.config import AppConfig, OutputConfig
from tsmixer_itransformer.errors import ArtifactError
from tsmixer_itransformer.models import BenchmarkData, BenchmarkResult


def test_save_artifacts(
    app_config: AppConfig, panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    metrics = pd.DataFrame({"model": ["TSMixer"], "mae": [1.0], "mse": [1.0]})
    data = BenchmarkData(panel, series_count=2, time_points=12)
    plot = app_config.output.directory / "plot.png"
    result = save_artifacts(
        app_config,
        data,
        BenchmarkResult(predictions, metrics),
        data_plot=plot,
        forecast_plot=plot,
    )
    assert result.predictions.exists()
    assert result.metrics.exists()
    assert result.manifest.exists()
    assert result.data_plot == plot
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest == {
        "dataset": "ettm2",
        "frequency": "15min",
        "horizon": 2,
        "prediction_rows": 8,
        "series_count": 2,
        "time_points": 12,
    }


def test_save_artifacts_wraps_write_error(
    tmp_path: Path, panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    config = AppConfig(output=OutputConfig(directory=occupied))
    with pytest.raises(ArtifactError, match="cannot write"):
        save_artifacts(
            config,
            BenchmarkData(panel, 2, 12),
            BenchmarkResult(predictions, pd.DataFrame({"model": ["x"]})),
        )
