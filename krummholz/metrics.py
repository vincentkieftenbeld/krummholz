import numpy as np


def accuracy_score(y_true, y_pred) -> float:
    """
    Compute the accuracy classification score.

    Parameters
    ----------
    y_true : array_like
        Ground truth labels
    y_pred : array_like
        Predicted labels.

    Returns
    -------
    score : float
        The proportion of correctly classified samples.

    Examples
    --------
    >>> y_true = [1, 0, 1]
    >>> y_pred = [0, 0, 1]
    >>> accuracy_score(y_true, y_pred)
    0.6666666666666666

    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    return np.mean(np.array(y_true) == np.array(y_pred))


def mean_absolute_error(y_true, y_pred) -> float:
    """
    Compute the mean absolute error regression loss.

    Parameters
    ----------
    y_true : array_like
        Ground truth targets.
    y_pred : float, or array_like
        Predicted targets. If a single value, then prediction is constant.

    Returns
    ------
    loss : float
      The mean absolute error regression loss.
    """
    assert (
        isinstance(y_pred, float) or y_true.shape == y_pred.shape
    ), "y_pred must be a float, or y_true and y_pred must have the same shape"
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred) -> float:
    """
    Compute the mean squared error regression loss.

    Parameters
    ----------
    y_true : array_like
        Ground truth targets.
    y_pred : float, or array_like
        Predicted targets. If a single value, then prediction is constant.

    Returns
    ------
    loss : float
      The mean squared error regression loss.
    """
    assert isinstance(y_pred, float) or (
        y_true.shape == y_pred.shape
    ), "y_pred must be a float, or y_true and y_pred must have the same shape"
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
