"""

References
---------

Vander Plas, J. (2XXX). Python Data Science,
pp. 421-
"""

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from krummholz.criteria import Gini
from krummholz.decision_tree import DecisionTree
from krummholz.metrics import accuracy_score
from krummholz.training import train_test_split


def plot(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")


def visualize_tree(model, X, y, ax=None, cmap="rainbow"):
    ax = ax or plt.gca()

    ax.scatter(
        X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3
    )
    ax.axis("tight")
    ax.axis("off")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=cmap,
        clim=(y.min(), y.max()),
        zorder=1,
    )
    ax.set(xlim=xlim, ylim=ylim)


def main():
    X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # plot(X_train, y_train)

    clf = DecisionTree(Gini(), max_depth=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{acc:.4f}")


if __name__ == "__main__":
    main()
