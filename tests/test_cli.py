from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from tsmixer_itransformer.cli import build_parser, main
from tsmixer_itransformer.errors import DataValidationError
from tsmixer_itransformer.models import BenchmarkArtifacts


def artifacts(tmp_path: Path, *, plots: bool = False) -> BenchmarkArtifacts:
    return BenchmarkArtifacts(
        predictions=tmp_path / "predictions.csv",
        metrics=tmp_path / "metrics.csv",
        manifest=tmp_path / "run.json",
        data_plot=tmp_path / "data.png" if plots else None,
        forecast_plot=tmp_path / "forecast.png" if plots else None,
    )


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config is None
    assert args.plot is False


def test_main_runs_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, bool]] = []

    def run(config: object, *, plot: bool) -> BenchmarkArtifacts:
        calls.append((config, plot))
        return artifacts(tmp_path)

    times = iter([60.0, 120.0])
    monkeypatch.setattr("tsmixer_itransformer.cli.run_benchmark", run)
    monkeypatch.setattr("tsmixer_itransformer.cli.time.monotonic", lambda: next(times))
    assert main([]) == 0
    assert calls[0][1] is False
    output = capsys.readouterr().out
    assert "predictions:" in output
    assert "elapsed minutes: 1.00" in output
    assert "data plot:" not in output


def test_main_loads_config_and_prints_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[model]\nhorizon = 2\n", encoding="utf-8")

    def run(config: object, *, plot: bool) -> BenchmarkArtifacts:
        assert config.model.horizon == 2  # type: ignore[attr-defined]
        assert plot is True
        return artifacts(tmp_path, plots=True)

    monkeypatch.setattr("tsmixer_itransformer.cli.run_benchmark", run)
    assert main(["--config", str(path), "--plot"]) == 0
    output = capsys.readouterr().out
    assert "data plot:" in output
    assert "forecast plot:" in output


def test_main_translates_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(config: object, *, plot: bool) -> BenchmarkArtifacts:
        raise DataValidationError("bad panel")

    monkeypatch.setattr("tsmixer_itransformer.cli.run_benchmark", broken)
    assert main([]) == 2
    assert "error: bad panel" in capsys.readouterr().err


def test_module_entrypoint_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tsmixer_itransformer.cli.main", lambda: 9)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("tsmixer_itransformer.__main__", run_name="__main__")
    assert raised.value.code == 9
