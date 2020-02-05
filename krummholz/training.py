import numpy as np

from typing import Optional, Tuple


def train_test_split(
    X,
    y,
    train_size: float = 0.8,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple:
    """Split a dataset into a training and test set.

    Parameters
    ----------
    X : array_like
      Features.
    y : array_like
      Targets.
    train_size : float, between 0 and 1
      Proportion of samples in the training set.
    shuffle : bool, default True
      Shuffle the data before splitting.
    random_state : int, optional
      Random state used as a seed.

    Returns
    -------
    X_train, X_test : array_like
      Training and test features.
    y_train, y_test : array_like
      Training and test targets.
    """
    X, y = np.array(X), np.array(y)

    assert 0 < train_size < 1, "train_size should be strictly between 0 and 1"
    assert X.ndim == 2, "X should be 2-dimensional"
    assert y.ndim == 1, "y should be 1-dimensional"
    assert X.shape[0] == y.shape[0]

    n_samples = X.shape[0]
    n_train = int(train_size * n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        # shuffle features and targets using the same permutation
        permutation = np.random.permutation(n_samples)
        X, y = X[permutation], y[permutation]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test
