import matplotlib.pyplot as plt
import numpy as np

from .decision_tree import DecisionTree
from .gradient_boosting import GradientBoostedTrees
from .metrics import accuracy_score


def plot_train_test(
    train, test, label, xlabel="Iterations", ylabel=None, skip=0, loc="upper right"
):
    """Plot training and validation."""
    assert len(train) == len(test)

    ylabel = ylabel or label.capitalize()
    xrange = range(skip, len(train))

    plt.plot(xrange, train[skip:], "b-", label=f"Training {label}")
    plt.plot(xrange, test[skip:], "r-", label=f"Test {label}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)


def plot_training(model: GradientBoostedTrees, X_test, y_test, skip: int = 1):
    """Plot training and validation loss of a gradient boosted tree estimator.

    Parameters
    ----------
    model : GradientBoostedTrees
        Gradient boosted tree model
    X_test, y_test : array_like
        Validation features and targets
    skip : int, optional (default: 1)
        Number of initial trees to skip
    """
    predictions = [model.estimators[0].predict(X_test)] + [
        model.learning_rate * estimator.predict(X_test)
        for estimator in model.estimators[1:]
    ]
    test_pred = np.cumsum(predictions, axis=0)
    test_loss = [model.criterion.loss(y_test, y_pred) for y_pred in test_pred]

    plot_train_test(
        model.losses,
        test_loss,
        "loss",
        xlabel="Number of trees",
        ylabel=model.criterion.label(),
    )
    plt.axhline(test_loss[-1], color="gray", linestyle="--")


def plot_tree_depth(criterion, max_depth, X_train, y_train, X_test, y_test):
    acc_train = []
    acc_test = []
    for depth in range(max_depth):
        model = DecisionTree(criterion, max_depth=depth).fit(X_train, y_train)
        acc_train.append(accuracy_score(y_train, model.predict(X_train)))
        acc_test.append(accuracy_score(y_test, model.predict(X_test)))
    plot_train_test(acc_train, acc_test, "accuracy", xlabel="Depth", loc="lower right")
