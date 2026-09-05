"""Filesystem persistence for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

from tsmixer_itransformer.config import AppConfig
from tsmixer_itransformer.errors import ArtifactError
from tsmixer_itransformer.models import BenchmarkArtifacts, BenchmarkData, BenchmarkResult


def save_artifacts(
    config: AppConfig,
    data: BenchmarkData,
    result: BenchmarkResult,
    *,
    data_plot: Path | None = None,
    forecast_plot: Path | None = None,
) -> BenchmarkArtifacts:
    """Persist predictions, metrics, and a compact machine-readable manifest."""
    output = config.output
    predictions_path = output.directory / output.predictions_file
    metrics_path = output.directory / output.metrics_file
    manifest_path = output.directory / output.manifest_file
    manifest = {
        "dataset": config.data.dataset,
        "frequency": config.data.frequency,
        "horizon": config.model.horizon,
        "prediction_rows": len(result.predictions),
        "series_count": data.series_count,
        "time_points": data.time_points,
    }
    try:
        output.directory.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(predictions_path, index=False)
        result.metrics.to_csv(metrics_path, index=False)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot write benchmark artifacts: {exc}") from exc
    return BenchmarkArtifacts(
        predictions=predictions_path,
        metrics=metrics_path,
        manifest=manifest_path,
        data_plot=data_plot,
        forecast_plot=forecast_plot,
    )
