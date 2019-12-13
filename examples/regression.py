"""
This example fits a regression tree step-by-step.

"""
import numpy as np

from saplings.criteria import MSE
from saplings.decision_tree import DecisionTree


def create_data():
    # create a simple regression problem that can be fit perfectly by a tree
    # if x1 <= t1, then y = y1
    # if x1 > t1, then
    # .. if x2 <= t2, then y = y2
    # .. if x2 > t2, then y = y3
    t1, t2 = np.random.rand(2)
    y1, y2, y3 = np.random.rand(3)

    n_samples, n_features = 100, 2

    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples)

    y[X[:, 0] <= t1] = y1
    y[(X[:, 0] > t1) & (X[:, 1] <= t2)] = y2
    y[(X[:, 0] > t1) & (X[:, 1] > t2)] = y3

    return X, y


def best_split(X, y, criterion):
    """Finds the best binary split based on criterion."""
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    # consider every possible feature
    for feature in range(X.shape[1]):
        # levels are the sorted unique feature values
        values = X[:, feature]
        levels = np.unique(values)

        # candidate thresholds are the midpoints between levels
        thresholds = (levels[1:] + levels[:-1]) / 2.0

        # consider every possible split
        for threshold in thresholds:
            left, right = y[values <= threshold], y[values > threshold]
            gain = criterion.gain(left, right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def main():
    # X, y = create_data()

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    criterion = MSE()

    # tree of depth 0 predicts mean
    depth = 0
    loss = criterion.impurity(y)
    print(f"depth = {depth}, loss = {loss:.5f}")

    # find the best split
    feature, threshold, gain = best_split(X, y, criterion)

    print(f"{feature}, {threshold}, {gain}")

    clf = DecisionTree(MSE(), 1).fit(X, y)


if __name__ == "__main__":
    np.random.seed(42)
    main()
