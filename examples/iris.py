from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from saplings.decision_tree import DecisionTree
from saplings.criteria import Gini


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    for clf in [DecisionTreeClassifier(criterion="gini"), DecisionTree(Gini())]:
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{acc}")


if __name__ == "__main__":
    main()
