from __future__ import annotations
import numpy as np


class Model:
    """A base class for classification and regression models."""

    def fit(self, X, y) -> Model:
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
        self : Model
          Model fitted to the training data.

        """
        raise NotImplementedError

    def predict(self, X) -> np.array:
        """Predict targets for samples."""
        return np.array([self._predict(x) for x in X])

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

    def _predict(self, x):
        """Predict target for a single sample."""
        raise NotImplementedError

    def _predict_proba(self, x):
        """Predict class probabilities for a single sample."""
        raise NotImplementedError


class BaseClassifier(Model):
    """A baseline classifier making predictions solely based on observed class frequencies."""

    def __init__(self) -> None:
        super().__init__()
        self.proba = None
        self.pred = None

    def fit(self, X, y) -> BaseClassifier:
        self.proba = np.bincount(y) / len(y)
        self.pred = np.argmax(self.proba)
        return self

    def _predict(self, x):
        return self.pred

    def _predict_proba(self, x):
        return self.proba


class BaseRegressor(Model):
    """A baseline regressor making predictions solely based on the observed mean."""

    def __init__(self) -> None:
        super().__init__()
        self.pred = None

    def fit(self, X, y) -> BaseRegressor:
        self.pred = np.mean(y)
        return self

    def _predict(self, x):
        return self.pred
