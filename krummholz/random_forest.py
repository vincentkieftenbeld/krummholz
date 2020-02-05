from __future__ import annotations
from typing import Optional

import numpy as np

from .criteria import Criterion
from .decision_tree import DecisionTree
from .models import Estimator


class RandomForest(Estimator):
    """A random forest is an ensemble of decision trees.

    Each decision tree is built using a different subset of the features and/or a different subsample of the data.

    Bagged decision trees use all the features, but different data

    References
    ----------
    .. Breiman, L. (2001). Random Forests, Machine Learning, 45(1), 5-32.
    """

    def __init__(
        self,
        criterion: Criterion,
        n_estimators: float,
        n_features: float = 1.0,
        n_samples: float = 1.0,
        replace: bool = True,
        max_depth: Optional[int] = None,
    ):
        assert 0 < n_samples <= 1.0 or int(n_samples) == n_samples

        self.n_estimators = n_estimators
        self.n_features = n_features
        self.n_samples = n_samples
        self.replace = replace

        self.estimators = [
            DecisionTree(criterion, n_features=self.n_features, max_depth=max_depth)
            for _ in range(n_estimators)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_estimators={self.n_estimators})"

    def fit(self, X, y) -> RandomForest:
        n_samples = (
            int(self.n_samples)
            if self.n_samples > 1
            else int(self.n_samples * X.shape[0])
        )

        for estimator in self.estimators:
            indices = sorted(
                np.random.choice(X.shape[0], n_samples, replace=self.replace)
            )
            X_sample = X[indices, :]
            y_sample = y[indices]
            estimator.fit(X_sample, y_sample)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        y_pred = np.array(
            [
                np.argmax(np.bincount(predictions[:, index]))
                for index in range(predictions.shape[1])
            ]
        )
        return y_pred
