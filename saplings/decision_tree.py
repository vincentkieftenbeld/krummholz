from __future__ import annotations

import numpy as np

from collections import namedtuple
from typing import List, Optional, Tuple, Union

from .criteria import Criterion
from .exceptions import NotFittedError
from .models import Model

# types to represent a binary tree
Leaf = namedtuple("Leaf", "value")
Split = namedtuple("Split", ["feature", "threshold", "left", "right"])
Tree = Union[Leaf, Split]


class DecisionTree(Model):
    """A decision tree model for classification or regression.

    The CART algorithm is used to fit a decision tree.

    Parameters
    ----------
    criterion : Criterion
        Criterion to determine the quality of a split.
        For example, Gini impurity is a criterion for classification,
        and mean squared error for regression.

    max_depth : int, optional
        The maximum depth of the tree. When None, the tree is expanded
        until every leaf is pure.

    Attributes
    ----------
    tree : Optional[Tree]
        Decision tree learned from data (None when untrained)

    References
    ----------
    .. [1]

    """

    def __init__(self, criterion: Criterion, max_depth: Optional[int] = None) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree: Optional[Tree] = None

    def __repr__(self) -> str:
        return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})"

    def fit(self, X, y) -> DecisionTree:
        """Train a decision-tree on the training data.

        Arguments
        ---------
        X : array_like
          Features

        y : array_like
          Targets

        Returns
        -------
        self : DecisionTree
            Fitted decision tree model.
        """
        depth = 0
        max_depth = self.max_depth or 1  # not implemented yet

        self.tree = _create_tree(X, y, self.criterion)
        return self

    def _predict(self, x) -> float:
        """Generate a prediction by traversing the tree."""
        if self.tree is None:
            raise NotFittedError("DecisionTree is not fitted yet.")

        node = self.tree

        while not isinstance(node, Leaf):
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value


def _create_tree(X, y, criterion: Criterion) -> Tree:
    """Recursively construct a decision tree."""

    # when we reached a pure node, return a leaf
    if len(np.unique(y)) == 1:
        return Leaf(value=criterion.predict(y))

    # otherwise, find the best split
    feature, threshold, left, right = best_split(X, y, criterion)

    tree = Split(
        feature,
        threshold,
        _create_tree(X[left, :], y[left], criterion),
        _create_tree(X[right, :], y[right], criterion),
    )

    return tree


def best_split(X, y, criterion: Criterion) -> Tuple[int, float, List[int], List[int]]:
    """
    Find the best binary split, based on an impurity criterion.

    This returns the best feature and threshold found, along with
    the indices of the left and right samples. This allows the caller to
    find the features as well as the labels.

    Returns
    -------
      feature : int
        Best feature column to split on.
      threshold : float
        Best threshold to use for the split.
      left, right : list of int
        Indices of left and right samples.

    """
    best_feature = None
    best_threshold = None
    best_gain = float("-inf")

    # consider every possible feature
    for feature in range(X.shape[1]):
        # levels are the sorted unique feature values
        values = X[:, feature]
        levels = np.unique(values)

        # candidate thresholds are the midpoints between levels
        thresholds = (levels[1:] + levels[:-1]) / 2.0

        # consider every possible split
        for threshold in thresholds:
            left = y[values <= threshold]
            right = y[values > threshold]
            gain = criterion.gain(left, right)
            if gain > best_gain:
                best_feature = feature
                best_threshold = threshold
                best_gain = gain

    # find the indices of the left and right samples of the best split
    left = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
    right = np.argwhere(X[:, best_feature] > best_threshold).flatten()

    return best_feature, best_threshold, left, right


