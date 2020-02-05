"""
Gradient boosted trees

References
----------

.. Parr, Terence, & Howard, Jeremy. How to explain gradient boosting.
explained.ai

.. XGBoost: A scalable tree boosting system
https://arxiv.org/pdf/1603.02754.pdf

.. Jerome H. Friendman, Greedy Function approximation: A gradient boosting machine
https://statweb.stanford.edu/~jhf/ftp/trebst.pdf

.. Biau, Gerard et al. Accelerated gradient boosting
https://arxiv.org/abs/1803.02042

"""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from krummholz.ensembles import Estimator
from krummholz.criteria import MSE
from krummholz.decision_tree import DecisionTree


class GradientBoostedTrees(Estimator):
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float = 1.0,
        criterion: Criterion = MSE(),
        max_depth: int = 1,
    ):
        """
        A gradient boosted tree ensemble.

        Parameters
        ----------
        n_estimators : int
            Number of trees to fit.

        learning_rate : float
            Learning rate

        criterion : Criterion
            Criterion to minimize; either mean squared error (MSE) or mean absolute error (MAE).

        max_depth : int, optional, defaults to 1
            Maximum depth of each tree.

        Attributes
        ----------
        estimators : List[DecisionTree]
            Boosted decision trees
        losses : List[float]
            Training loss at each stage

        References
        ----------
        .. Aarshay Jain: Complete Guide to Parameter Tuning in XGBoost
        .. Jason Brownlee: Tune Learning Rate for Gradient Boosting with XGBoost in Python.

        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.max_depth = max_depth

        self.estimators: List[DecisionTree] = []
        self.losses: List[float] = []

    def fit(self, X, y) -> GradientBoostedTrees:
        """
        Fit a gradient boosted tree model.

        With the mean squared error loss, this uses Friedman's L2boost algorithm.

        """

        # start with a decision tree of depth 0, which predicts the mean (MSE) or median (MAE)
        clf = DecisionTree(self.criterion, max_depth=0).fit(X, y)
        y_pred = clf.predict(X)
        loss = self.criterion.loss(y, y_pred)
        self.losses.append(loss)
        self.estimators.append(clf)

        # fit subsequent regression tree stumps on the residuals
        for stage in range(1, self.n_estimators):
            residuals = -self.criterion.gradient(y, y_pred)
            clf = DecisionTree(self.criterion, self.max_depth).fit(X, residuals)
            y_pred += self.learning_rate * clf.predict(X)
            loss = self.criterion.loss(y, y_pred)
            self.losses.append(loss)
            self.estimators.append(clf)

        return self

    def predict(self, X) -> np.array:
        predictions = [estimator.predict(X) for estimator in self.estimators[1:]]
        y_pred = self.estimators[0].predict(X) + self.learning_rate * np.sum(
            predictions, axis=0
        )
        return y_pred
