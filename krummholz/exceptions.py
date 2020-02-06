class NotFittedError(ValueError, AttributeError):
    """Exception raised when an estimator is used before fitting."""
