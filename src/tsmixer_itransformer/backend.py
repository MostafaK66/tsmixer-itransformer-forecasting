"""NeuralForecast boundary isolated from the offline application core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pandas as pd

from tsmixer_itransformer.config import DataConfig, ModelConfig, SplitConfig
from tsmixer_itransformer.errors import BackendError, DependencyUnavailableError


class ForecastEngine(Protocol):
    def cross_validation(self, **kwargs: object) -> object: ...


ModelFactory = Callable[..., object]
EngineFactory = Callable[..., ForecastEngine]


class NeuralForecastBackend:
    """Build both models and execute non-overlapping temporal cross-validation."""

    def __init__(
        self,
        *,
        engine_factory: EngineFactory | None = None,
        tsmixer_factory: ModelFactory | None = None,
        itransformer_factory: ModelFactory | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._tsmixer_factory = tsmixer_factory
        self._itransformer_factory = itransformer_factory

    def predict(
        self,
        frame: pd.DataFrame,
        data: DataConfig,
        split: SplitConfig,
        model: ModelConfig,
        series_count: int,
    ) -> pd.DataFrame:
        """Train TSMixer and iTransformer and validate their predictions."""
        engine_factory, tsmixer_factory, itransformer_factory = self._factories()
        common: dict[str, object] = {
            "h": model.horizon,
            "input_size": model.input_size,
            "n_series": series_count,
            "learning_rate": model.learning_rate,
            "batch_size": model.batch_size,
            "max_steps": model.max_steps,
            "early_stop_patience_steps": model.early_stop_patience_steps,
            "scaler_type": model.scaler_type,
            "random_seed": model.random_seed,
            "enable_progress_bar": False,
            "logger": False,
        }
        try:
            tsmixer = tsmixer_factory(
                **common,
                n_block=model.n_block,
                ff_dim=model.ff_dim,
            )
            itransformer = itransformer_factory(
                **common,
                hidden_size=model.hidden_size,
                n_heads=model.n_heads,
                e_layers=model.encoder_layers,
                d_layers=model.decoder_layers,
                d_ff=model.d_ff,
            )
            engine = engine_factory(models=[tsmixer, itransformer], freq=data.frequency)
            raw = engine.cross_validation(
                df=frame,
                val_size=split.validation_size,
                test_size=split.test_size,
                n_windows=None,
                step_size=model.horizon,
            )
        except Exception as exc:
            raise BackendError(f"NeuralForecast benchmark failed: {exc}") from exc
        return _validate_predictions(raw, frame, split.test_size)

    def _factories(self) -> tuple[EngineFactory, ModelFactory, ModelFactory]:
        if all(
            factory is not None
            for factory in (
                self._engine_factory,
                self._tsmixer_factory,
                self._itransformer_factory,
            )
        ):
            return cast(
                tuple[EngineFactory, ModelFactory, ModelFactory],
                (
                    self._engine_factory,
                    self._tsmixer_factory,
                    self._itransformer_factory,
                ),
            )
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TSMixer, iTransformer
        except ImportError as exc:
            raise DependencyUnavailableError(
                "forecasting requires: pip install "
                "'tsmixer-itransformer-forecasting[forecast]'"
            ) from exc
        return (
            self._engine_factory or cast(EngineFactory, NeuralForecast),
            self._tsmixer_factory or cast(ModelFactory, TSMixer),
            self._itransformer_factory or cast(ModelFactory, iTransformer),
        )


def _validate_predictions(
    value: object, source: pd.DataFrame, test_size: int
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise BackendError("NeuralForecast returned empty or invalid predictions")
    result = value.copy()
    required = ["unique_id", "ds", "y", "TSMixer", "iTransformer"]
    if not set(required).issubset(result.columns):
        result = result.reset_index()
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise BackendError(f"predictions are missing columns: {', '.join(missing)}")
    if result[required].isnull().any().any():
        raise BackendError("predictions contain null values")
    for column in ("y", "TSMixer", "iTransformer"):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise BackendError(f"prediction column {column} is not numeric") from exc
    if not np.isfinite(result[["y", "TSMixer", "iTransformer"]].to_numpy()).all():
        raise BackendError("predictions contain non-finite values")
    if result.duplicated(["unique_id", "ds"]).any():
        raise BackendError("predictions contain overlapping item/timestamp rows")
    try:
        result["ds"] = pd.to_datetime(result["ds"], errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise BackendError("prediction timestamps are invalid") from exc
    expected = source.groupby("unique_id", sort=False).tail(test_size)
    expected = expected.sort_values(["unique_id", "ds"], kind="stable").reset_index(
        drop=True
    )
    result = result.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)
    if len(result) != len(expected) or not result[["unique_id", "ds"]].equals(
        expected[["unique_id", "ds"]]
    ):
        raise BackendError("predictions do not exactly cover the configured test region")
    if not np.allclose(result["y"].to_numpy(), expected["y"].to_numpy()):
        raise BackendError("prediction actuals do not match the source test region")
    return result
