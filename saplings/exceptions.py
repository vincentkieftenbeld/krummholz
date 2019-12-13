class NotFittedError(ValueError, AttributeError):
    """Exception to raise when an estimator is used before fitting."""
