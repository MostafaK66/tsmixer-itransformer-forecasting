import numpy as np
import pandas as pd
import pytest

from tsmixer_itransformer.errors import BackendError
from tsmixer_itransformer.evaluation import evaluate_predictions


def test_evaluate_predictions(predictions: pd.DataFrame) -> None:
    result = evaluate_predictions(predictions)
    assert result.to_dict("records") == [
        {"model": "TSMixer", "mae": 1.0, "mse": 1.0},
        {"model": "iTransformer", "mae": 2.0, "mse": 4.0},
    ]


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame(), pd.DataFrame({"y": [1], "TSMixer": [1]})],
)
def test_evaluation_requires_complete_nonempty_frame(frame: pd.DataFrame) -> None:
    with pytest.raises(BackendError, match="requires"):
        evaluate_predictions(frame)


def test_evaluation_rejects_nonfinite_values(predictions: pd.DataFrame) -> None:
    predictions.loc[0, "TSMixer"] = np.inf
    with pytest.raises(BackendError, match="non-finite"):
        evaluate_predictions(predictions)
