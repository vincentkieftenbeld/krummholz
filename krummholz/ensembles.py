import numpy as np

from typing import List
from .models import Estimator


class Ensemble(Estimator):
    """Base class for an ensemble model consisting of a bag of models."""

    def __init__(self, estimators: List[Estimator] = None) -> None:
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y) -> Estimator:
        """Fit every member of the ensemble to the training data."""
        for estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def _predict(self, x):
        """Predict by averaging predictions of every estimator"""
        y_pred = [estimator.predict(x) for estimator in self.estimators]
        return np.mean(y_pred, axis=0)

    def _predict_proba(self, x):
        """Predict probability by averaging probabilities of every estimator."""
        y_proba = [estimator.predict_proba(x) for estimator in self.estimators]
        return np.mean(y_proba, axis=0)


class BaggingEstimator(Estimator):
    """
    A bagging ensemble.

    References
    ----------
    .. Breiman, L. (1996). Bagging predictors, Machine Learning, 24, 123-140.

    """

    def __init__(
        self,
        estimator: Estimator,
        n_estimators: int,
        max_samples: int,
        replacement: bool = True,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.replacement = True

    def fit(self, X, y):
        # need to draw samples here
        super().fit(X, y)
