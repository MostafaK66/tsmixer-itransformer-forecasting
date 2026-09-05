# Engineering contract

- Support Python 3.11 and 3.12 using the `src/` package layout.
- Keep configuration immutable and validate all external frames before training.
- Keep downloads, NeuralForecast, GPU use, plotting, and writes behind boundaries.
- Unit tests must be deterministic and require no network, GPU, or benchmark download.
- Derive `n_series` from the validated panel; never hard-code it.
- Preserve the balanced, regular timestamp grid required by multivariate models.
- Run Ruff, strict mypy, branch-aware pytest coverage, and compile checks before merge.
- Never commit datasets, predictions, checkpoints, caches, logs, or credentials.
