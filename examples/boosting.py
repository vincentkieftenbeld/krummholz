import numpy as np

from krummholz.criteria import MSE
from krummholz.decision_tree import DecisionTree
from krummholz.metrics import mean_squared_error

n_samples = 10
X = np.sort(10 * np.random.random(n_samples)).reshape(10, 1)
y = np.random.random(n_samples)

estimators = []

predictions = np.zeros(len(y))
targets = y

# fit estimators
idx = 0

for idx in range(5):
    clf = DecisionTree(MSE(), max_depth=1)
    clf.fit(X, targets)

    predictions = predictions + clf.predict(X)
    targets = y - predictions

    print()
    # print(f"estimator {idx}")
    # print(f"y_true: {y}")
    # print(f"y_pred: {predictions}")
    # print(f"residuals: {targets}")
    print(f"loss: {mean_squared_error(y, predictions)}")

# use model
from krummholz.gradient_boosting import GradientBoostedTrees

clf = GradientBoostedTrees(5).fit(X, y)
print(clf.losses)

y_pred = clf.predict(X)
print(y)
print(y_pred)