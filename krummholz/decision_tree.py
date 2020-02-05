from __future__ import annotations

import numpy as np


from typing import List, Optional, Set, Tuple

from .criteria import Criterion
from .exceptions import NotFittedError
from .models import Estimator
from .tree import Leaf, Split, Tree


class DecisionTree(Estimator):
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
        Decision tree learned from data, None when untrained

    References
    ----------
    .. [1] Decision tree
       [2] CART

    """

    def __init__(
        self,
        criterion: Criterion,
        max_depth: Optional[int] = None,
        n_features: float = 1.0,
    ) -> None:
        assert 0 < n_features <= 1.0 or int(n_features) == n_features
        assert max_depth is None or max_depth >= 0

        self.criterion = criterion
        self.max_depth = max_depth
        self.n_features = n_features

        self.tree: Optional[Tree] = None

    def __repr__(self) -> str:
        return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})"

    def fit(self, X, y) -> DecisionTree:
        """Train a decision tree on the training data.

        Uses a random selection of `n_features`

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
        max_depth = self.max_depth if self.max_depth is not None else float("inf")
        n_features = (
            int(self.n_features)
            if self.n_features > 1
            else int(self.n_features * X.shape[1])
        )

        self.tree = _create_tree(X, y, self.criterion, max_depth)
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

    def _predict_proba(self, x):
        """Predict probabilities as portion of leaf.

        Only meaningful for a classifier.
        """
        pass


def _create_tree(X, y, criterion: Criterion, max_depth: float = float("inf")) -> Tree:
    """
    Recursively construct a binary decision tree.

    Each node in the decision tree is either a leaf node `Leaf` or a binary split `Split`.
    Every leaf has a value determined by splitting criterion, namely, the value that minimizes the loss.
    Every split has a feature index and a threshold, with a left side consisting of all rows such that
    feature <= threshold, and right side all rows such that feature > threshold.

    Parameters
    ----------
        X : array_like
            Features
        y : array_list
            Targets
        criterion : Criterion
            Criterion used to judge each split.
        max_depth : int
            Maximum depth of tree, possibly infinite `float("inf")`

    Returns
    -------
        tree : Tree
            Constructed binary tree
    """

    # if we have reached a pure node or the maximum depth,
    # return a leaf with a value provided by the criterion
    if len(np.unique(y)) == 1 or max_depth == 0:
        return Leaf(value=criterion.predict(y))

    # otherwise, find the best split and recurse on the left and right samples
    feature, threshold, left, right = find_best_split(X, y, criterion)
    tree = Split(
        feature,
        threshold,
        _create_tree(X[left, :], y[left], criterion, max_depth - 1),
        _create_tree(X[right, :], y[right], criterion, max_depth - 1),
    )

    return tree


def find_best_split(
    X, y, criterion: Criterion
) -> Tuple[int, float, List[int], List[int]]:
    """
    Find the best binary split, based on an impurity criterion.

    This returns the best feature and threshold found, along with
    the indices of the left and right samples. This allows the caller to
    find the features as well as the labels.

    Parameters
    ----------
      X, y : array_like
      criterion : Criterion

    Returns
    -------
      feature : int
        Best feature column to split on.
      threshold : float
        Best threshold to use for the split.
      left, right : list of int
        Indices of left and right samples.

    """
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
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

    assert best_feature is not None
    assert best_threshold is not None

    # find the indices of the left and right samples of the best split
    best_left = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
    best_right = np.argwhere(X[:, best_feature] > best_threshold).flatten()

    return best_feature, best_threshold, best_left, best_right
