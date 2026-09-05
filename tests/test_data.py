from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsmixer_itransformer.config import DataConfig, ModelConfig, SplitConfig
from tsmixer_itransformer.data import load_benchmark, validate_history
from tsmixer_itransformer.errors import DataValidationError, DependencyUnavailableError
from tsmixer_itransformer.models import BenchmarkData


def test_load_validates_and_sorts_panel(panel: pd.DataFrame) -> None:
    shuffled = panel.sample(frac=1, random_state=4).assign(extra=1)
    calls: list[DataConfig] = []

    def reader(config: DataConfig) -> pd.DataFrame:
        calls.append(config)
        return shuffled

    config = DataConfig(dataset="ettm2")
    result = load_benchmark(config, reader=reader)
    assert calls == [config]
    assert result.series_count == 2
    assert result.time_points == 12
    assert result.frame.columns.tolist() == ["unique_id", "ds", "y"]
    assert result.frame["y"].dtype == np.float64
    assert result.frame.equals(result.frame.sort_values(["unique_id", "ds"]))


def test_load_ettm1_selects_ot(panel: pd.DataFrame) -> None:
    source = panel.replace({"a": "OT", "b": "HUFL"})
    result = load_benchmark(DataConfig(dataset="ettm1"), reader=lambda config: source)
    assert result.series_count == 1
    assert result.frame["unique_id"].unique().tolist() == ["OT"]


def test_load_honors_explicit_series(panel: pd.DataFrame) -> None:
    result = load_benchmark(
        DataConfig(dataset="ettm2", series=("b",)), reader=lambda config: panel
    )
    assert result.series_count == 1
    assert set(result.frame["unique_id"]) == {"b"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="y"), "missing columns"),
        (lambda frame: frame.iloc[:0], "no selected rows"),
        (lambda frame: frame.assign(y=np.nan), "must not contain nulls"),
        (lambda frame: frame.assign(ds="invalid"), "cannot parse"),
        (lambda frame: frame.assign(y="invalid"), "cannot parse"),
        (lambda frame: frame.assign(y=np.inf), "non-finite"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "unique"),
    ],
)
def test_load_rejects_invalid_frames(panel: pd.DataFrame, mutate, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_benchmark(DataConfig(), reader=lambda config: mutate(panel.copy()))


def test_load_rejects_non_dataframe() -> None:
    with pytest.raises(DataValidationError, match="DataFrame"):
        load_benchmark(DataConfig(), reader=lambda config: [])  # type: ignore[arg-type,return-value]


def test_load_wraps_reader_error() -> None:
    def broken(config: DataConfig) -> pd.DataFrame:
        raise OSError("offline")

    with pytest.raises(DataValidationError, match="offline"):
        load_benchmark(DataConfig(), reader=broken)


def test_load_reports_missing_selected_series(panel: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="missing selected series: c"):
        load_benchmark(DataConfig(series=("a", "c")), reader=lambda config: panel)


def test_load_rejects_too_few_timestamps(panel: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="at least two"):
        load_benchmark(
            DataConfig(), reader=lambda config: panel.groupby("unique_id").head(1)
        )


def test_load_rejects_irregular_frequency(panel: pd.DataFrame) -> None:
    broken = panel.drop(index=2)
    with pytest.raises(DataValidationError, match="configured frequency"):
        load_benchmark(DataConfig(), reader=lambda config: broken)


def test_load_rejects_unbalanced_grid(panel: pd.DataFrame) -> None:
    broken = panel.drop(index=12)
    with pytest.raises(DataValidationError, match="same timestamp grid"):
        load_benchmark(DataConfig(), reader=lambda config: broken)


def test_load_rejects_invalid_frequency(panel: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="invalid data frequency"):
        load_benchmark(
            DataConfig(frequency="not-a-frequency"), reader=lambda config: panel
        )


def test_validate_history_accepts_boundary(panel: pd.DataFrame) -> None:
    data = load_benchmark(DataConfig(), reader=lambda config: panel)
    validate_history(
        data,
        SplitConfig(validation_size=2, test_size=4),
        ModelConfig(horizon=2, input_size_multiplier=2),
    )


def test_validate_history_rejects_short_panel(panel: pd.DataFrame) -> None:
    data = BenchmarkData(panel, series_count=2, time_points=11)
    with pytest.raises(DataValidationError, match="at least 12.*found 11"):
        validate_history(
            data,
            SplitConfig(validation_size=2, test_size=4),
            ModelConfig(horizon=2, input_size_multiplier=2),
        )


def test_validate_history_requires_complete_nonoverlapping_windows(
    panel: pd.DataFrame,
) -> None:
    data = BenchmarkData(panel, series_count=2, time_points=20)
    with pytest.raises(DataValidationError, match="divisible"):
        validate_history(
            data,
            SplitConfig(validation_size=2, test_size=5),
            ModelConfig(horizon=2, input_size_multiplier=2),
        )


def test_default_reader_explains_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("datasetsforecast"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="forecast"):
        load_benchmark(DataConfig(download_dir=Path("unused")))


@pytest.mark.parametrize("valid", [True, False])
def test_default_reader_uses_datasetsforecast(
    panel: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid: bool,
) -> None:
    calls: list[tuple[str, str]] = []

    class LongHorizon:
        @staticmethod
        def load(*, directory: str, group: str) -> tuple[object, None, None]:
            calls.append((directory, group))
            return (panel if valid else [], None, None)

    package = types.ModuleType("datasetsforecast")
    module = types.ModuleType("datasetsforecast.long_horizon")
    module.LongHorizon = LongHorizon  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasetsforecast", package)
    monkeypatch.setitem(sys.modules, "datasetsforecast.long_horizon", module)
    config = DataConfig(download_dir=tmp_path)
    if valid:
        result = load_benchmark(config)
        assert result.series_count == 2
        assert calls == [(str(tmp_path), "ETTm2")]
    else:
        with pytest.raises(DataValidationError, match="invalid target frame"):
            load_benchmark(config)
