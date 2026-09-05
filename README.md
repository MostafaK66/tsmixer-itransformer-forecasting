# TSMixer and iTransformer Forecasting

A validated benchmark pipeline that compares TSMixer and iTransformer on the ETTm1
or ETTm2 long-horizon datasets through NeuralForecast. The project supports Python
3.11 and 3.12 while keeping downloads, GPU training, and plotting outside its
deterministic offline test path.

## What it does

- downloads ETTm1 or ETTm2 through `datasetsforecast`, or accepts an injected frame;
- validates schema, timestamps, numeric values, uniqueness, frequency, and panel balance;
- selects `OT` by default for the ETTm1 univariate experiment;
- derives the correct series count dynamically for the ETTm2 multivariate experiment;
- runs explicit non-overlapping temporal cross-validation;
- compares TSMixer and iTransformer with deterministic MAE and MSE metrics;
- writes predictions, metrics, a run manifest, and optional static plots.

The model parameters follow the current official [TSMixer](https://nixtlaverse.nixtla.io/neuralforecast/models.tsmixer.html),
[iTransformer](https://nixtlaverse.nixtla.io/neuralforecast/models.itransformer.html),
and [cross-validation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html)
documentation.

## Installation

Linux and macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cp config.example.toml config.toml
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
Copy-Item config.example.toml config.toml
```

The base package contains the testable data, evaluation, and artifact core. Install
`.[forecast]` for benchmark downloads and NeuralForecast, or `.[plots]` for plots.

## Usage

```bash
forecast-benchmark --config config.toml
forecast-benchmark --config config.toml --plot
```

Without `--config`, the ETTm2 defaults from `config.example.toml` are used. Relative
paths in TOML are resolved from that configuration file. The first real run downloads
the selected benchmark and trains two neural models; runtime depends on hardware and
the configured training steps.

The default 96-step horizon divides the official 11,520-point test segment exactly.
Other configurations must also make `test_size` divisible by `horizon`, preventing
partial or overlapping evaluation windows.

## Data contract

Frames must contain `unique_id`, `ds`, and `y`. Each item/timestamp pair must be
unique, finite, and non-null. All selected series must share the same strictly
regular timestamp grid. Every series must also leave enough history for the model
input window, validation region, test region, and forecast horizon.

The original script removed 10% of timestamps using Isolation Forest. That makes a
regular forecasting panel irregular and changes the official benchmark, so the
production pipeline intentionally validates and preserves the source observations.
Outlier experiments should transform values without deleting timestamps and should
be implemented as a separately tested preprocessing policy.

## Outputs

The output directory contains:

- `predictions.csv`: aligned actuals and both model forecasts;
- `metrics.csv`: MAE and MSE per model;
- `run.json`: dataset, series count, row count, and configuration summary;
- optional dataset and forecast PNG files.

Generated datasets, outputs, model checkpoints, caches, and logs are excluded from Git.

## Architecture

Frozen configuration objects validate all parameters. Dataset access and the complete
NeuralForecast/model factory are injectable, so local tests exercise orchestration
without importing PyTorch or using a GPU. Prediction validation prevents missing,
duplicated, reordered, or non-finite model output from reaching evaluation. The CLI
only parses arguments and translates domain errors.

## Development

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest --cov=tsmixer_itransformer --cov-report=term-missing
python -m mypy
python -m compileall -q src tests
```

CI runs the same gates on Python 3.11 and 3.12. Tests do not download a benchmark,
import NeuralForecast, train a model, contact a network service, or require a GPU.

## License and attribution

See `NOTICE` for project lineage and `LICENSE` for MIT terms. Third-party libraries,
model implementations, and benchmark datasets retain their own licenses.
