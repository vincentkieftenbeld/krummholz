import numpy as np

from krummholz.criteria import Gini

x = np.array([5, 22, 9, 3, 4, 7, 9, 3, 3, 1])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0])


def test_gini():
    criterion = Gini()
    assert criterion.impurity(np.zeros(5, dtype="int64")) == 0
