import numpy as np
import pytest

from saplings.criteria import Gini, MSE
from saplings.decision_tree import best_split, DecisionTree


@pytest.mark.parametrize("criterion", [Gini(), MSE()], ids=["Gini", "MSE"])
def test_best_split(criterion):
    X = np.array(
        [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1], [9, 0]]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    feature, threshold, left, right = best_split(X, y, criterion)

    assert feature == 0
    assert threshold == 3.5
    assert np.array_equal(left, range(0, 4))
    assert np.array_equal(right, range(4, 10))


def test_decision_tree_classification():
    X = np.array(
        [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1], [9, 0]]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    clf = DecisionTree(Gini())
    clf.fit(X, y)

    return clf
