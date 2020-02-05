from sklearn.datasets import load_iris

from krummholz.criteria import Gini
from krummholz.decision_tree import DecisionTree
from krummholz.metrics import accuracy_score
from krummholz.training import train_test_split



def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    dt = DecisionTree(criterion=Gini())
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{acc:.4f}")



if __name__ == "__main__":
    main()
