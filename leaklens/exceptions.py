class LeakLensError(Exception):
    """Base exception for all LeakLens errors."""


class InvalidTargetError(LeakLensError):
    """Raised when the target column is missing or invalid."""


class SchemaMismatchError(LeakLensError):
    """Raised when train/test schemas are incompatible for comparison."""


class MissingTimestampError(LeakLensError):
    """Raised when temporal checks are requested but no datetime column is found."""


class EmptyDataFrameError(LeakLensError):
    """Raised when an input dataframe is empty."""
