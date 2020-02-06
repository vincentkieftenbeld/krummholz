import numpy as np

from krummholz.criteria import MAE, MSE
from krummholz.gradient_boosting import GradientBoostedTrees

# example from Parr & Howard
X = np.array([[750], [800], [850], [900], [950]])

y = np.array([1160, 1200, 1280, 1450, 2000])


def test_gradient_boosted_trees_mae():
    n_estimators = 4
    criterion = MAE()

    clf = GradientBoostedTrees(n_estimators, criterion=criterion)
    clf.fit(X, y)

    assert len(clf.estimators) == n_estimators

    y_pred = clf.estimators[0].predict(X)
    assert (y_pred == np.median(y)).all()

    y_pred = clf.predict(X)
    expected = np.array([1155, 1185, 1455, 1455, 2000])
    assert (y_pred == expected).all()


def test_gradient_boosted_trees_mse():
    clf = GradientBoostedTrees(n_estimators=1, criterion=MSE()).fit(X, y)
    y_pred = clf.predict(X)
    assert (y_pred == np.mean(y)).all()
