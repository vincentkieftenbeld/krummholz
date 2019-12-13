from typing import List
from .models import Model


class Ensemble(Model):
    """Base class for an ensemble model."""

    def __init__(self, n_estimators: int) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.estimators: List[Model] = list()

    def _predict(self, x):
        y_pred = [estimator.predict(x) for estimator in self.estimators]
        return np.mean(y_pred, axis=0)
