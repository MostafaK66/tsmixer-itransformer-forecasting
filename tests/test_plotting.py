from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from tsmixer_itransformer.errors import ArtifactError, DependencyUnavailableError
from tsmixer_itransformer.plotting import save_data_plot, save_forecast_plot


class FakeAxis:
    def __init__(self) -> None:
        self.lines = 0

    def plot(self, *args: object, **kwargs: object) -> None:
        self.lines += 1

    def set_xlabel(self, value: str) -> None:
        pass

    def set_ylabel(self, value: str) -> None:
        pass

    def set_title(self, value: str) -> None:
        pass

    def grid(self, value: bool) -> None:
        pass

    def legend(self) -> None:
        pass


class FakeFigure:
    def __init__(self, *, error: bool = False) -> None:
        self.axis = FakeAxis()
        self.error = error

    def add_subplot(self, value: int) -> FakeAxis:
        return self.axis

    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path) -> None:
        if self.error:
            raise OSError("disk full")
        path.write_text("plot", encoding="utf-8")


def install_matplotlib(
    monkeypatch: pytest.MonkeyPatch, *, error: bool = False
) -> FakeFigure:
    figure = FakeFigure(error=error)
    package = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda **kwargs: figure  # type: ignore[attr-defined]
    pyplot.close = lambda value: None  # type: ignore[attr-defined]
    package.pyplot = pyplot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return figure


def test_save_data_plot(
    panel: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = install_matplotlib(monkeypatch)
    result = save_data_plot(panel, tmp_path / "nested" / "data.png")
    assert result.exists()
    assert figure.axis.lines == 2


def test_save_forecast_plot(
    predictions: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = install_matplotlib(monkeypatch)
    result = save_forecast_plot(predictions, tmp_path / "forecast.png")
    assert result.exists()
    assert figure.axis.lines == 6


def test_plot_wraps_write_error(
    panel: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_matplotlib(monkeypatch, error=True)
    with pytest.raises(ArtifactError, match="disk full"):
        save_data_plot(panel, tmp_path / "plot.png")


def test_plot_explains_missing_dependency(
    panel: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="plots"):
        save_data_plot(panel, tmp_path / "plot.png")
