import numpy as np
import pytest

from saplings.models import BaseClassifier, BaseRegressor


def test_base_classifier():
    X_train = [0, 1, 2]
    X_test = [3, 5]
    y_train = [0, 0, 1, 1, 1]

    clf = BaseClassifier()
    clf.fit(X_train, y_train)

    # base classifier always predicts majority class
    y_pred = clf.predict(X_test)
    assert y_pred.shape == (2,)
    assert (y_pred == 1).all()

    # base classifier always predicts probabilities equal to observed class proportions
    y_proba = clf.predict_proba(X_test)
    assert y_proba.shape == (2, 2)
    assert (y_proba == [0.4, 0.6]).all()


def test_base_regressor():
    X_train = np.arange(5)[:, np.newaxis]
    y_train = np.arange(10, 60, 10)
    X_test = np.random.random(5)[:, np.newaxis]

    clf = BaseRegressor()
    clf.fit(X_train, y_train)

    # base regressor always predicts observed mean
    y_pred = clf.predict(X_test)
    assert (y_pred == 30.0).all()

    # base regressor cannot predict probabilities
    with pytest.raises(NotImplementedError):
        clf.predict_proba(X_test)
