from collections import namedtuple
from typing import Optional, List, Union

# types to represent a binary tree
Leaf = namedtuple("Leaf", "value")
Split = namedtuple("Split", ["feature", "threshold", "left", "right"])
Tree = Union[Leaf, Split]


def _max_feature(tree: Tree) -> int:
    """Find the maximum feature index used in a tree."""
    if isinstance(tree, Leaf):
        return 0

    return max(tree.feature, _max_feature(tree.left), _max_feature(tree.right))


def print_tree(
    tree: Tree, feature_names: Optional[List[str]] = None, indent: int = 0
) -> None:
    """Print a text representation of a binary decision tree.

    Parameters
    ----------
      tree : Tree
        Binary decision tree consisting of leafs and splits
      feature_names : List[str], optional
        List of feature names, default is "feature 0", "feature 1", ...
      indent: int, optional
        Indentation level

    """
    spacing = " " * 2

    if feature_names is None:
        feature_names = [f"feature {index}" for index in range(_max_feature(tree) + 1)]

    node = tree
    if isinstance(node, Leaf):
        print(f"{spacing * indent}value = {node.value}")

    if isinstance(node, Split):
        print(f"{spacing * indent}{feature_names[node.feature]} <= {node.threshold}:")
        print_tree(node.left, feature_names, indent + 1)

        print(f"{spacing * indent}{feature_names[node.feature]} > {node.threshold}:")
        print_tree(node.right, feature_names, indent + 1)
