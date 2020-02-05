from __future__ import annotations

import numpy as np

from krummholz.exceptions import NotFittedError


class Estimator:
    """A base class for classification and regression models."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def fit(self, X, y) -> Estimator:
        """
        Train the model on a dataset.

        Parameters
        ----------
        X : array-like
          Features.
        y : array_like
          Targets.

        Returns
        -------
        self : Estimator
          Model fitted to the training data.

        """
        raise NotImplementedError

    def predict(self, X) -> np.array:
        """Predict targets for samples."""
        return np.array([self._predict(x) for x in X])

    def fit_predict(self, X, y, X_pred=None):
        """Convenience method to fit an estimator and predict."""
        X_pred = X_pred or X
        self.fit(X, y)
        return self.predict(X_pred)

    def predict_proba(self, X) -> np.array:
        """Predict probabilities.

        Predict probabilities is only implemented for classifiers.

        Parameters
        ----------
            X : array
              Features

        Returns
        -------
            proba : array
              Predicted probabilities of shape (n_samples, n_classes)

        """
        return np.array([self._predict_proba(x) for x in X])

    def _predict(self, x) -> float:
        """Predict target for a single sample."""
        raise NotImplementedError

    def _predict_proba(self, x):
        """Predict class probabilities for a single sample."""
        raise NotImplementedError


class BaselineClassifier(Estimator):
    """A baseline classifier making predictions solely based on observed class frequencies."""

    def __init__(self) -> None:
        super().__init__()
        self.proba = None
        self.pred = None

    def fit(self, X, y) -> BaselineClassifier:
        self.proba = np.bincount(y) / len(y)
        self.pred = np.argmax(self.proba)
        return self

    def _predict(self, x):
        if self.pred is None:
            raise NotFittedError

        return self.pred

    def _predict_proba(self, x):
        if self.pred is None:
            raise NotFittedError

        return self.proba


class BaselineRegressor(Estimator):
    """A baseline regressor making predictions solely based on the observed mean."""

    def __init__(self) -> None:
        super().__init__()
        self.pred = None

    def fit(self, X, y) -> BaselineRegressor:
        self.pred = np.mean(y)
        return self

    def _predict(self, x):
        if self.pred is None:
            raise NotFittedError
        return self.pred
