import numpy as np

from krummholz.training import train_test_split


def test_train_test_split():
    X = np.array(range(100))[:, np.newaxis]
    y = range(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    assert y_train.shape == (80,)
    assert y_test.shape == (20,)

    assert len(X_train), len(X_test) == (80, 20)
    assert len(y_train), len(y_test) == (80, 20)

    assert (X_train == np.arange(80)[:, np.newaxis]).all()
    assert (y_train == np.arange(80)).all()

    assert (X_test == np.arange(80, 100)[:, np.newaxis]).all()
    assert (y_test == np.arange(80, 100)).all()


def test_train_test_split_shuffle():
    X = np.array(range(100))[:, np.newaxis]
    y = range(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    assert y_train.shape == (80,)
    assert y_test.shape == (20,)

    # features and targets should be in the same (random) order
    assert (X_train[:, 0] == y_train).all()
    assert (X_test[:, 0] == y_test).all()
