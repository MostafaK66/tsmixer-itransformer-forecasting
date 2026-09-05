"""Expected application failures."""


class BenchmarkError(Exception):
    """Base class for recoverable benchmark errors."""


class ConfigurationError(BenchmarkError):
    """Configuration is invalid or cannot be loaded."""


class DataValidationError(BenchmarkError):
    """Benchmark data violates the panel contract."""


class DependencyUnavailableError(BenchmarkError):
    """An optional runtime dependency is unavailable."""


class BackendError(BenchmarkError):
    """The forecasting backend failed or returned malformed output."""


class ArtifactError(BenchmarkError):
    """A generated artifact could not be written."""
