import numpy as np

from saplings.metrics import mean_squared_error


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

    def impurity(self, x):
        raise NotImplementedError

    def predict(self, x) -> float:
        raise NotImplementedError


class ClassificationCriterion(Criterion):
    def predict(self, x) -> int:
        """A classification tree predicts the majority label at its leafs."""
        x = np.array(x)
        return np.argmax(np.bincount(x))


class RegressionCriterion(Criterion):
    def predict(self, x: np.ndarray) -> float:
        """A regression tree predicts the mean at its leafs."""
        return np.mean(x)


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


class MSE(RegressionCriterion):
    """
    Mean squared error criterion used to fit regression trees.

    """

    def __repr__(self) -> str:
        return "MSE()"

    def impurity(self, x, y=None) -> float:
        assert x.ndim == 1

        y = y or self.predict(x)
        assert y.ndim == 1

        return mean_squared_error(x, y)
