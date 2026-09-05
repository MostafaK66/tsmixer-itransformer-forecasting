"""Thin command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from tsmixer_itransformer.config import AppConfig
from tsmixer_itransformer.errors import BenchmarkError
from tsmixer_itransformer.service import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-benchmark",
        description="Compare TSMixer and iTransformer with temporal cross-validation",
    )
    parser.add_argument("--config", type=Path, help="TOML configuration path")
    parser.add_argument("--plot", action="store_true", help="save static PNG plots")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = time.monotonic()
    try:
        config = AppConfig.from_toml(args.config) if args.config else AppConfig()
        artifacts = run_benchmark(config, plot=args.plot)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"predictions: {artifacts.predictions}")
    print(f"metrics: {artifacts.metrics}")
    print(f"manifest: {artifacts.manifest}")
    if artifacts.data_plot is not None:
        print(f"data plot: {artifacts.data_plot}")
    if artifacts.forecast_plot is not None:
        print(f"forecast plot: {artifacts.forecast_plot}")
    print(f"elapsed minutes: {(time.monotonic() - started) / 60:.2f}")
    return 0
