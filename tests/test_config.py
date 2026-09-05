from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tsmixer_itransformer.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
)
from tsmixer_itransformer.errors import ConfigurationError


def test_defaults_are_normalized_and_immutable() -> None:
    config = AppConfig(data=DataConfig(dataset="ETTM1"))
    assert config.data.dataset == "ettm1"
    assert config.data.group == "ETTm1"
    assert config.data.selected_series == ("OT",)
    assert config.model.input_size == 288
    with pytest.raises(FrozenInstanceError):
        config.model.batch_size = 1  # type: ignore[misc]


def test_ettm2_selects_all_series_by_default() -> None:
    config = DataConfig(dataset="ettm2")
    assert config.group == "ETTm2"
    assert config.selected_series is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "other"},
        {"frequency": ""},
        {"series": ()},
        {"series": ("",)},
        {"series": ("OT", "OT")},
    ],
)
def test_data_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        DataConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs", [{"validation_size": 0}, {"test_size": 0}, {"test_size": -1}]
)
def test_split_config_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError, match="positive"):
        SplitConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "positive"),
        ({"learning_rate": 0.0}, "between"),
        ({"learning_rate": 1.0}, "between"),
        ({"early_stop_patience_steps": -2}, "at least"),
        ({"hidden_size": 7, "n_heads": 2}, "divisible"),
        ({"scaler_type": ""}, "valid"),
        ({"random_seed": -1}, "valid"),
    ],
)
def test_model_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predictions_file": ""},
        {"metrics_file": "nested/file.csv"},
        {"manifest_file": "../run.json"},
        {"data_plot_file": "plots/data.png"},
        {"forecast_plot_file": "root/forecast.png"},
    ],
)
def test_output_config_rejects_unsafe_names(kwargs: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError, match="simple filename"):
        OutputConfig(**kwargs)


def test_load_complete_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[data]
dataset = "ettm1"
download_dir = "cache"
frequency = "30min"
series = ["OT", "HUFL"]
[split]
validation_size = 10
test_size = 20
[model]
horizon = 4
input_size_multiplier = 2
n_block = 3
ff_dim = 16
learning_rate = 0.01
batch_size = 8
max_steps = 50
early_stop_patience_steps = -1
hidden_size = 32
n_heads = 4
encoder_layers = 3
decoder_layers = 2
d_ff = 64
scaler_type = "standard"
random_seed = 7
[output]
directory = "artifacts"
predictions_file = "p.csv"
metrics_file = "m.csv"
manifest_file = "r.json"
data_plot_file = "d.png"
forecast_plot_file = "f.png"
""",
        encoding="utf-8",
    )
    config = AppConfig.from_toml(path)
    assert config.data.download_dir == tmp_path / "cache"
    assert config.data.selected_series == ("OT", "HUFL")
    assert config.split.test_size == 20
    assert config.model.input_size == 8
    assert config.model.scaler_type == "standard"
    assert config.output.directory == tmp_path / "artifacts"


def test_load_minimal_toml_with_absolute_path(tmp_path: Path) -> None:
    absolute = tmp_path / "cache"
    path = tmp_path / "config.toml"
    path.write_text(f'[data]\ndownload_dir = "{absolute}"\n', encoding="utf-8")
    config = AppConfig.from_toml(path)
    assert config.data.download_dir == absolute
    assert config.data.series is None


@pytest.mark.parametrize(
    "content",
    [
        "invalid = [",
        "data = 1",
        "[data]\ndataset = 1",
        "[data]\nseries = 1",
        "[data]\nseries = [1]",
        "[split]\ntest_size = true",
        '[model]\nhorizon = "2"',
        '[model]\nlearning_rate = "slow"',
        "[output]\ndirectory = 3",
    ],
)
def test_load_rejects_invalid_toml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_load_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        AppConfig.from_toml(tmp_path / "missing.toml")
