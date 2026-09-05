from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tsmixer_itransformer.backend import NeuralForecastBackend
from tsmixer_itransformer.config import DataConfig, ModelConfig, SplitConfig
from tsmixer_itransformer.errors import BackendError, DependencyUnavailableError


class FakeEngine:
    def __init__(self, result: object, *, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.options: dict[str, object] = {}

    def cross_validation(self, **kwargs: object) -> object:
        self.options = kwargs
        if self.error:
            raise self.error
        return self.result


def test_backend_builds_dynamic_models_and_cross_validation(
    panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    model_calls: dict[str, dict[str, object]] = {}
    engine = FakeEngine(predictions)
    engine_calls: list[dict[str, object]] = []

    def model_factory(name: str):
        def build(**kwargs: object) -> object:
            model_calls[name] = kwargs
            return name

        return build

    def engine_factory(**kwargs: object) -> FakeEngine:
        engine_calls.append(kwargs)
        return engine

    model = ModelConfig(
        horizon=2,
        input_size_multiplier=2,
        n_block=3,
        ff_dim=16,
        learning_rate=0.01,
        batch_size=8,
        max_steps=20,
        hidden_size=16,
        n_heads=4,
        encoder_layers=3,
        decoder_layers=2,
        d_ff=32,
        scaler_type="standard",
        random_seed=7,
    )
    result = NeuralForecastBackend(
        engine_factory=engine_factory,
        tsmixer_factory=model_factory("TSMixer"),
        itransformer_factory=model_factory("iTransformer"),
    ).predict(panel, DataConfig(), SplitConfig(2, 4), model, series_count=2)
    assert engine_calls == [{"models": ["TSMixer", "iTransformer"], "freq": "15min"}]
    assert model_calls["TSMixer"]["n_series"] == 2
    assert model_calls["TSMixer"]["n_block"] == 3
    assert model_calls["iTransformer"]["n_heads"] == 4
    assert model_calls["iTransformer"]["e_layers"] == 3
    assert model_calls["TSMixer"]["enable_progress_bar"] is False
    assert engine.options["step_size"] == 2
    assert engine.options["n_windows"] is None
    assert result.equals(predictions)


def configured_backend(result: object) -> NeuralForecastBackend:
    engine = FakeEngine(result)
    return NeuralForecastBackend(
        engine_factory=lambda **kwargs: engine,
        tsmixer_factory=lambda **kwargs: object(),
        itransformer_factory=lambda **kwargs: object(),
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "empty or invalid"),
        (pd.DataFrame(), "empty or invalid"),
        (pd.DataFrame({"y": [1]}), "missing columns"),
    ],
)
def test_backend_rejects_invalid_frames(
    panel: pd.DataFrame, value: object, message: str
) -> None:
    with pytest.raises(BackendError, match=message):
        configured_backend(value).predict(
            panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("y", None, "null"),
        ("TSMixer", "bad", "not numeric"),
        ("iTransformer", np.inf, "non-finite"),
    ],
)
def test_backend_rejects_bad_prediction_values(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    column: str,
    value: object,
    message: str,
) -> None:
    broken = predictions.copy()
    if isinstance(value, str):
        broken[column] = broken[column].astype(object)
    broken.loc[0, column] = value
    with pytest.raises(BackendError, match=message):
        configured_backend(broken).predict(
            panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
        )


def test_backend_accepts_required_columns_in_index(
    panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    indexed = predictions.set_index(["unique_id", "ds"])
    result = configured_backend(indexed).predict(
        panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
    )
    assert {"unique_id", "ds"}.issubset(result.columns)


def test_backend_rejects_overlapping_predictions(
    panel: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(BackendError, match="overlapping"):
        configured_backend(duplicated).predict(
            panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
        )


@pytest.mark.parametrize("change", ["missing", "timestamp", "actual"])
def test_backend_requires_exact_source_alignment(
    panel: pd.DataFrame, predictions: pd.DataFrame, change: str
) -> None:
    broken = predictions.copy()
    if change == "missing":
        broken = broken.iloc[:-1]
    elif change == "timestamp":
        broken["ds"] = broken["ds"].astype(object)
        broken.loc[0, "ds"] = "invalid"
    else:
        broken.loc[0, "y"] += 1
    message = "timestamps" if change == "timestamp" else "test region|actuals"
    with pytest.raises(BackendError, match=message):
        configured_backend(broken).predict(
            panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
        )


def test_backend_wraps_execution_failure(panel: pd.DataFrame) -> None:
    engine = FakeEngine(None, error=RuntimeError("accelerator failed"))
    backend = NeuralForecastBackend(
        engine_factory=lambda **kwargs: engine,
        tsmixer_factory=lambda **kwargs: object(),
        itransformer_factory=lambda **kwargs: object(),
    )
    with pytest.raises(BackendError, match="accelerator failed"):
        backend.predict(panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2)


def test_backend_explains_missing_dependency(
    panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("neuralforecast"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="forecast"):
        NeuralForecastBackend().predict(
            panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
        )


def test_backend_uses_default_neuralforecast_factories(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[str] = []

    def tsmixer(**kwargs: object) -> str:
        model_calls.append("TSMixer")
        return "TSMixer"

    def itransformer(**kwargs: object) -> str:
        model_calls.append("iTransformer")
        return "iTransformer"

    class NeuralForecast:
        def __init__(self, **kwargs: object) -> None:
            pass

        def cross_validation(self, **kwargs: object) -> pd.DataFrame:
            return predictions

    package = types.ModuleType("neuralforecast")
    models = types.ModuleType("neuralforecast.models")
    package.NeuralForecast = NeuralForecast  # type: ignore[attr-defined]
    models.TSMixer = tsmixer  # type: ignore[attr-defined]
    models.iTransformer = itransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setitem(sys.modules, "neuralforecast.models", models)
    result = NeuralForecastBackend().predict(
        panel, DataConfig(), SplitConfig(2, 4), ModelConfig(horizon=2), 2
    )
    assert model_calls == ["TSMixer", "iTransformer"]
    assert len(result) == 8
