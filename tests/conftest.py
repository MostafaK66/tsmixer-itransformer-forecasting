from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsmixer_itransformer.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2026-01-01", periods=12, freq="15min")
    for item, offset in (("a", 0.0), ("b", 20.0)):
        for index, timestamp in enumerate(timestamps):
            rows.append({"unique_id": item, "ds": timestamp, "y": offset + index})
    return pd.DataFrame(rows)


@pytest.fixture
def predictions(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.groupby("unique_id", sort=False).tail(4).copy()
    frame["TSMixer"] = frame["y"] + 1.0
    frame["iTransformer"] = frame["y"] - 2.0
    return frame.reset_index(drop=True)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(download_dir=tmp_path / "datasets"),
        split=SplitConfig(validation_size=2, test_size=4),
        model=ModelConfig(
            horizon=2,
            input_size_multiplier=2,
            n_block=2,
            ff_dim=8,
            batch_size=4,
            max_steps=10,
            hidden_size=8,
            n_heads=2,
            d_ff=16,
        ),
        output=OutputConfig(directory=tmp_path / "outputs"),
    )
