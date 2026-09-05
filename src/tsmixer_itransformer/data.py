"""Dataset download boundary and balanced-panel validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from tsmixer_itransformer.config import DataConfig, ModelConfig, SplitConfig
from tsmixer_itransformer.errors import DataValidationError, DependencyUnavailableError
from tsmixer_itransformer.models import BenchmarkData

DatasetReader = Callable[[DataConfig], pd.DataFrame]


def load_benchmark(
    config: DataConfig, *, reader: DatasetReader | None = None
) -> BenchmarkData:
    """Load, select, and validate a long-format benchmark panel."""
    selected_reader = reader or _read_long_horizon
    try:
        raw = selected_reader(config)
    except DependencyUnavailableError:
        raise
    except Exception as exc:
        raise DataValidationError(f"cannot load {config.group}: {exc}") from exc
    if not isinstance(raw, pd.DataFrame):
        raise DataValidationError("dataset loader must return a pandas DataFrame")
    missing = [name for name in ("unique_id", "ds", "y") if name not in raw.columns]
    if missing:
        raise DataValidationError(f"dataset is missing columns: {', '.join(missing)}")
    frame = raw.loc[:, ["unique_id", "ds", "y"]].copy()
    selected = config.selected_series
    if selected is not None:
        frame = frame[frame["unique_id"].isin(selected)].copy()
        found = set(frame["unique_id"].astype(str))
        absent = [item for item in selected if item not in found]
        if absent:
            raise DataValidationError(
                f"dataset is missing selected series: {', '.join(absent)}"
            )
    if frame.empty:
        raise DataValidationError("dataset contains no selected rows")
    if frame[["unique_id", "ds", "y"]].isnull().any().any():
        raise DataValidationError("unique_id, ds, and y must not contain nulls")
    try:
        frame["ds"] = pd.to_datetime(frame["ds"], errors="raise", format="mixed")
        frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot parse ds or y: {exc}") from exc
    if not np.isfinite(frame["y"].to_numpy()).all():
        raise DataValidationError("y contains non-finite values")
    if frame.duplicated(["unique_id", "ds"]).any():
        raise DataValidationError("unique_id and ds pairs must be unique")
    frame = frame.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)
    time_points = _validate_grid(frame, config.frequency)
    return BenchmarkData(
        frame=frame,
        series_count=int(frame["unique_id"].nunique()),
        time_points=time_points,
    )


def validate_history(data: BenchmarkData, split: SplitConfig, model: ModelConfig) -> None:
    """Reject panels that cannot support training, validation, and testing."""
    if split.test_size % model.horizon != 0:
        raise DataValidationError(
            "split.test_size must be divisible by model.horizon for complete "
            "non-overlapping evaluation"
        )
    required = split.validation_size + split.test_size + model.input_size + model.horizon
    if data.time_points < required:
        raise DataValidationError(
            f"each series needs at least {required} time points; found {data.time_points}"
        )


def _validate_grid(frame: pd.DataFrame, frequency: str) -> int:
    groups = list(frame.groupby("unique_id", sort=False))
    first_times = pd.DatetimeIndex(groups[0][1]["ds"])
    if len(first_times) < 2:
        raise DataValidationError("every series must contain at least two timestamps")
    try:
        expected = pd.date_range(first_times[0], periods=len(first_times), freq=frequency)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid data frequency {frequency}: {exc}") from exc
    if not first_times.equals(expected):
        raise DataValidationError(
            "series timestamps do not match the configured frequency"
        )
    for _, group in groups[1:]:
        if not pd.DatetimeIndex(group["ds"]).equals(first_times):
            raise DataValidationError("all series must share the same timestamp grid")
    return len(first_times)


def _read_long_horizon(config: DataConfig) -> pd.DataFrame:
    try:
        from datasetsforecast.long_horizon import LongHorizon
    except ImportError as exc:
        raise DependencyUnavailableError(
            "dataset loading requires: pip install "
            "'tsmixer-itransformer-forecasting[forecast]'"
        ) from exc
    loaded = LongHorizon.load(directory=str(config.download_dir), group=config.group)
    frame = loaded[0]
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("datasetsforecast returned an invalid target frame")
    return frame
