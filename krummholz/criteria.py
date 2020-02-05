import numpy as np

from krummholz.metrics import mean_absolute_error, mean_squared_error


class Criterion:
    """
    Base class for a binary splitting criterion used by a decision tree.

    """

    def gain(self, x, y):
        values = np.concatenate([x, y])
        impurity = self.impurity(values)

        left_impurity = len(x) * self.impurity(x)
        right_impurity = len(y) * self.impurity(y)

        return impurity - (left_impurity + right_impurity)

    def impurity(self, x, y=None) -> float:
        assert x.ndim == 1

        y = y or self.predict(x)
        # assert y.ndim == 1

        return self.loss(x, y)

    def loss(self, x, y) -> float:
        raise NotImplementedError

    def predict(self, x) -> float:
        raise NotImplementedError

    def gradient(self, x) -> float:
        raise NotImplementedError

    def label(self) -> str:
        return "Criterion"


class ClassificationCriterion(Criterion):
    def predict(self, x) -> int:
        """A classification tree predicts the majority label at its leafs."""
        x = np.array(x)
        return np.argmax(np.bincount(x))


class Gini(ClassificationCriterion):
    """
    Gini criterion.

    """

    def __repr__(self) -> str:
        return "Gini()"

    def impurity(self, x) -> float:
        x = np.array(x)
        assert x.ndim == 1
        p = np.bincount(x) / len(x)
        return np.sum(p * (1 - p))

    def label(self) -> str:
        return "Gini impurity"


class MSE(Criterion):
    """
    Mean squared error criterion used to fit regression trees.

    MSE is minimized by the mean.
    """

    def __repr__(self) -> str:
        return "MSE()"

    def loss(self, y_true, y_pred) -> float:
        return 0.5 * mean_squared_error(y_true, y_pred)

    def predict(self, x: np.ndarray) -> float:
        return np.mean(x)

    def gradient(self, y_true, y_pred) -> float:
        return y_pred - y_true

    def label(self) -> str:
        return "Mean squared error"


class MAE(Criterion):
    """
    Mean absolute error criterion used to fit (robust) boosted trees.

    Mean absolute error loss is minimized by the median.
    """

    def __repr__(self) -> str:
        return "MAE()"

    def loss(self, x, y):
        return mean_absolute_error(x, y)

    def predict(self, x: np.ndarray):
        return np.median(x)

    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)

    def label(self) -> str:
        return "Mean absolute error"
