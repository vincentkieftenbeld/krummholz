import numpy as np
import pytest

from krummholz.criteria import Gini, MSE
from krummholz.decision_tree import find_best_split, DecisionTree, Split, Leaf


@pytest.mark.parametrize("criterion", [Gini(), MSE()], ids=["Gini", "MSE"])
def test_best_split(criterion):
    X = np.array(
        [[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1], [9, 0]]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    feature, threshold, left, right = find_best_split(X, y, criterion)

    assert feature == 0
    assert threshold == 3.5
    assert np.array_equal(left, range(0, 4))
    assert np.array_equal(right, range(4, 10))


tree_depth_1 = Split(feature=0, threshold=3.5, left=Leaf(value=1), right=Leaf(value=2))

tree_depth_2 = Split(
    feature=0,
    threshold=3.5,
    left=Split(feature=0, threshold=0.5, left=Leaf(value=0), right=Leaf(value=1)),
    right=Leaf(value=2),
)


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [(None, tree_depth_2), (1, tree_depth_1), (2, tree_depth_2), (3, tree_depth_2)],
    ids=["max_depth=None", "max_depth=1", "max_depth=2", "max_depth=3"],
)
def test_decision_tree_classification(max_depth, expected):
    X = np.array(
        [[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1], [9, 0]]
    )
    y = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    clf = DecisionTree(Gini(), max_depth).fit(X, y)
    actual = clf.tree
    assert actual == expected
