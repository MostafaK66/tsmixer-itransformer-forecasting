"""Immutable, strictly typed TOML configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tsmixer_itransformer.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataConfig:
    dataset: str = "ettm2"
    download_dir: Path = Path("datasets")
    frequency: str = "15min"
    series: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        normalized = self.dataset.lower()
        if normalized not in {"ettm1", "ettm2"}:
            raise ConfigurationError("data.dataset must be ettm1 or ettm2")
        object.__setattr__(self, "dataset", normalized)
        if not self.frequency.strip():
            raise ConfigurationError("data.frequency must not be empty")
        if self.series is not None and (
            not self.series or any(not item.strip() for item in self.series)
        ):
            raise ConfigurationError("data.series must contain non-empty names")
        if self.series is not None and len(set(self.series)) != len(self.series):
            raise ConfigurationError("data.series must not contain duplicates")

    @property
    def group(self) -> str:
        return "ETTm1" if self.dataset == "ettm1" else "ETTm2"

    @property
    def selected_series(self) -> tuple[str, ...] | None:
        if self.series is not None:
            return self.series
        return ("OT",) if self.dataset == "ettm1" else None


@dataclass(frozen=True, slots=True)
class SplitConfig:
    validation_size: int = 11_520
    test_size: int = 11_520

    def __post_init__(self) -> None:
        if self.validation_size < 1 or self.test_size < 1:
            raise ConfigurationError("split sizes must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    horizon: int = 96
    input_size_multiplier: int = 3
    n_block: int = 4
    ff_dim: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    max_steps: int = 1_000
    early_stop_patience_steps: int = 3
    hidden_size: int = 512
    n_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 1
    d_ff: int = 2_048
    scaler_type: str = "identity"
    random_seed: int = 42

    def __post_init__(self) -> None:
        positive = (
            self.horizon,
            self.input_size_multiplier,
            self.n_block,
            self.ff_dim,
            self.batch_size,
            self.max_steps,
            self.hidden_size,
            self.n_heads,
            self.encoder_layers,
            self.decoder_layers,
            self.d_ff,
        )
        if any(value < 1 for value in positive):
            raise ConfigurationError("model size and training values must be positive")
        if not 0 < self.learning_rate < 1:
            raise ConfigurationError("model.learning_rate must be between 0 and 1")
        if self.early_stop_patience_steps < -1:
            raise ConfigurationError(
                "model.early_stop_patience_steps must be at least -1"
            )
        if self.hidden_size % self.n_heads != 0:
            raise ConfigurationError("model.hidden_size must be divisible by n_heads")
        if not self.scaler_type.strip() or self.random_seed < 0:
            raise ConfigurationError("model scaler and random seed must be valid")

    @property
    def input_size(self) -> int:
        return self.horizon * self.input_size_multiplier


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: Path = Path("outputs")
    predictions_file: str = "predictions.csv"
    metrics_file: str = "metrics.csv"
    manifest_file: str = "run.json"
    data_plot_file: str = "dataset.png"
    forecast_plot_file: str = "forecast.png"

    def __post_init__(self) -> None:
        for name, value in (
            ("predictions_file", self.predictions_file),
            ("metrics_file", self.metrics_file),
            ("manifest_file", self.manifest_file),
            ("data_plot_file", self.data_plot_file),
            ("forecast_plot_file", self.forecast_plot_file),
        ):
            if not value or Path(value).name != value:
                raise ConfigurationError(f"output.{name} must be a simple filename")


@dataclass(frozen=True, slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
        base = path.resolve().parent
        data = _table(raw, "data")
        split = _table(raw, "split")
        model = _table(raw, "model")
        output = _table(raw, "output")
        return cls(
            data=DataConfig(
                dataset=_string(data.get("dataset", "ettm2"), "data.dataset"),
                download_dir=_resolve(
                    base,
                    Path(
                        _string(data.get("download_dir", "datasets"), "data.download_dir")
                    ),
                ),
                frequency=_string(data.get("frequency", "15min"), "data.frequency"),
                series=_optional_strings(data.get("series"), "data.series"),
            ),
            split=SplitConfig(
                validation_size=_integer(
                    split.get("validation_size", 11_520), "split.validation_size"
                ),
                test_size=_integer(split.get("test_size", 11_520), "split.test_size"),
            ),
            model=ModelConfig(
                horizon=_integer(model.get("horizon", 96), "model.horizon"),
                input_size_multiplier=_integer(
                    model.get("input_size_multiplier", 3), "model.input_size_multiplier"
                ),
                n_block=_integer(model.get("n_block", 4), "model.n_block"),
                ff_dim=_integer(model.get("ff_dim", 128), "model.ff_dim"),
                learning_rate=_number(
                    model.get("learning_rate", 0.001), "model.learning_rate"
                ),
                batch_size=_integer(model.get("batch_size", 32), "model.batch_size"),
                max_steps=_integer(model.get("max_steps", 1_000), "model.max_steps"),
                early_stop_patience_steps=_integer(
                    model.get("early_stop_patience_steps", 3),
                    "model.early_stop_patience_steps",
                ),
                hidden_size=_integer(model.get("hidden_size", 512), "model.hidden_size"),
                n_heads=_integer(model.get("n_heads", 8), "model.n_heads"),
                encoder_layers=_integer(
                    model.get("encoder_layers", 2), "model.encoder_layers"
                ),
                decoder_layers=_integer(
                    model.get("decoder_layers", 1), "model.decoder_layers"
                ),
                d_ff=_integer(model.get("d_ff", 2_048), "model.d_ff"),
                scaler_type=_string(
                    model.get("scaler_type", "identity"), "model.scaler_type"
                ),
                random_seed=_integer(model.get("random_seed", 42), "model.random_seed"),
            ),
            output=OutputConfig(
                directory=_resolve(
                    base,
                    Path(_string(output.get("directory", "outputs"), "output.directory")),
                ),
                predictions_file=_string(
                    output.get("predictions_file", "predictions.csv"),
                    "output.predictions_file",
                ),
                metrics_file=_string(
                    output.get("metrics_file", "metrics.csv"), "output.metrics_file"
                ),
                manifest_file=_string(
                    output.get("manifest_file", "run.json"), "output.manifest_file"
                ),
                data_plot_file=_string(
                    output.get("data_plot_file", "dataset.png"),
                    "output.data_plot_file",
                ),
                forecast_plot_file=_string(
                    output.get("forecast_plot_file", "forecast.png"),
                    "output.forecast_plot_file",
                ),
            ),
        )


def _table(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a TOML table")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be numeric")
    return float(value)


def _optional_strings(value: object, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{name} must be an array of strings")
    return tuple(value)


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path
