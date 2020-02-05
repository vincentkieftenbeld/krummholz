from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from krummholz.criteria import Gini
from krummholz.random_forest import RandomForest
from krummholz.training import train_test_split

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = RandomForest(n_estimators=10, n_features=10, n_samples=1, criterion=Gini())
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print(cm)

# sns.heatmap(cm.T, square=True, annot=True, fmt="d", cbar=False)
# plt.xlabel("true")
# plt.ylabel("predicted")
